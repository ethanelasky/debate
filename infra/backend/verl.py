"""VERL backend. Drives verl as a worker pool — no RayPPOTrainer.

Targets verl main >= 0.9.0.dev (pin commit e961840 or later): uses
TinkerActorRolloutRefWorker (verl/workers/engine_workers_tinker.py), whose
decomposed optimizer_zero_grad / forward_backward / optimizer_step RPCs match
the Backend contract exactly (optimizer_step has zero_grad_on_exit=True, i.e.
"apply and clear"). Sampling goes through verl's rollout HTTP server
(LLMServerClient); weight sync via CheckpointEngineManager (colocated "naive"
mode: sleep FSDP-side, push over CUDA IPC, wake).

All verl imports are function-local so this module imports on machines without
verl/ray/torch (e.g. the dev laptop).

Loss mapping: LossSpec "ppo" -> verl policy loss "vanilla" with
clip_ratio_low = 1 - clip_low, clip_ratio_high = clip_high - 1.
"importance_sampling" is approximated by effectively-unclipped "vanilla".
Note verl's default aggregation is token-mean, while Tinker's built-in losses
are token sums — learning rates are NOT directly comparable across backends.

Memory model (colocated): sync_sampler() wakes the rollout engine with fresh
weights; the first forward/forward_backward after sampling puts it back to
sleep so FSDP has the GPU. Both transitions are automatic.
"""

from __future__ import annotations

import asyncio
import importlib.util
import math
import os
import uuid
from dataclasses import dataclass, field

from infra.backend.base import (
    Backend,
    Datum,
    LossSpec,
    OptimParams,
    Sample,
    SamplingParams,
    Tokens,
)


@dataclass
class VerlBackendConfig:
    model_path: str
    n_gpus: int = 1
    prompt_length: int = 4096
    response_length: int = 2048
    gpu_memory_utilization: float = 0.6
    rollout_tp: int = 1
    strategy: str = "fsdp2"  # "fsdp" | "fsdp2" | "megatron"
    max_token_len_per_gpu: int = 16384
    lora_rank: int = 32  # matches the Tinker backend; 0 = full finetune
    # verl's native differentiable in-loss KL vs the frozen ref (k3 /
    # low_var_kl, per minibatch). 0 = off (the campaign default; advantage-
    # space KL lives in infra/rl/kl.py). Set from training.kl_coef by
    # build_backend when training.kl_mechanism is "loss"; the train loop then
    # stamps datum.ref_logprobs and _pack forwards them as ref_log_prob.
    kl_loss_coef: float = 0.0
    lr: float = 1e-5  # initial; overridden per optim_step call
    grad_clip: float = 1.0
    checkpoint_dir: str = "checkpoints/verl"
    loss: LossSpec = field(default_factory=LossSpec)
    # Megatron-only parallelism (ignored on fsdp*). TinkerTrainingWorker is
    # engine-agnostic (it drives BaseEngine), so only config selection differs.
    megatron_tp: int = 1
    megatron_pp: int = 1
    megatron_cp: int = 1
    megatron_ep: int = 1
    # Escape hatch to verl's padded (non-rmpad) forward. NOT needed for
    # sliding-window models — the OLMo-3 long-context logprob skew was a
    # transformers YaRN regression (#39847, fixed >=5.13), not rmpad. Note the
    # engine reads this from the batch TensorDict (_pack stamps it), not from
    # the hydra model config.
    use_remove_padding: bool = True
    extra_overrides: tuple[str, ...] = ()

    @property
    def config_name(self) -> str:
        # ppo_megatron_trainer.yaml == ppo_trainer + `override model_engine:
        # megatron` (it also sets actor.strategy), so no strategy override needed.
        return "ppo_megatron_trainer" if self.strategy == "megatron" else "ppo_trainer"

    def _strategy_overrides(self) -> list[str]:
        if self.strategy == "megatron":
            overrides = [
                f"actor_rollout_ref.actor.megatron.tensor_model_parallel_size={self.megatron_tp}",
                f"actor_rollout_ref.actor.megatron.pipeline_model_parallel_size={self.megatron_pp}",
                f"actor_rollout_ref.actor.megatron.context_parallel_size={self.megatron_cp}",
                f"actor_rollout_ref.actor.megatron.expert_model_parallel_size={self.megatron_ep}",
            ]
            if self.lora_rank > 0:
                # Megatron reads the NESTED model.lora block (run-verified note
                # from the prior repo); default target modules adapt qkv/proj/fc.
                overrides += [
                    f"actor_rollout_ref.model.lora.rank={self.lora_rank}",
                    f"actor_rollout_ref.model.lora.alpha={2 * self.lora_rank}",
                    "actor_rollout_ref.model.lora.lora_A_init_method=kaiming",
                ]
            return overrides
        overrides = [f"actor_rollout_ref.actor.strategy={self.strategy}"]
        if self.lora_rank > 0:
            # The FSDP/dp engine reads the FLAT lora keys (run-verified on
            # verl 0.8.0 hardware); alpha = 2*rank matches Tinker parity.
            overrides += [
                f"actor_rollout_ref.model.lora_rank={self.lora_rank}",
                f"actor_rollout_ref.model.lora_alpha={2 * self.lora_rank}",
            ]
        return overrides

    def hydra_overrides(self) -> list[str]:
        # actor.clip_ratio_low/high feed the vanilla (ppo) loss AND gspo —
        # verl's compute_policy_loss_gspo reads them as 1±eps, falling back to
        # the generic clip_ratio default (PPO-scale ~0.2) when absent. GSPO's
        # sequence ratios concentrate near 1, so a PPO-scale clip is
        # effectively no clip: gspo therefore REQUIRES explicit paper-scale
        # bounds and gets its epsilons emitted. cispo reads them too (verl
        # clamps the IS weight to [1-eps_low, 1+eps_high] with stop-grad);
        # gpg alone ignores them.
        clip_overrides: list[str] = []
        if self.loss.kind == "gspo" and self.loss.clip_low < 0.99:
            raise ValueError(
                f"gspo on verl needs paper-scale clip bounds (e.g. clip_low 0.9997 / "
                f"clip_high 1.0004); got clip_low={self.loss.clip_low} — PPO-scale bounds "
                "leave sequence ratios effectively unclipped"
            )
        if self.loss.kind in ("ppo", "importance_sampling", "gspo", "cispo"):
            eps_low = round(1.0 - self.loss.clip_low, 6)
            eps_high = round(self.loss.clip_high - 1.0, 6)
            if self.loss.kind == "importance_sampling":
                eps_low, eps_high = 1.0, 1000.0  # effectively unclipped
            clip_overrides = [
                f"actor_rollout_ref.actor.clip_ratio_low={eps_low}",
                f"actor_rollout_ref.actor.clip_ratio_high={eps_high}",
            ]
        loss_mode = {
            "ppo": "vanilla",
            "importance_sampling": "vanilla",
            "gspo": "gspo",
            "cispo": "cispo",
            "reinforce": "gpg",  # verl's gpg IS -logp*adv (verified: core_algos.py:1723)
        }.get(self.loss.kind)
        if loss_mode is None:
            raise NotImplementedError(f"loss {self.loss.kind!r} not supported on verl (use tinker)")
        mode_override = (
            [f"actor_rollout_ref.actor.policy_loss.loss_mode={loss_mode}"] if loss_mode != "vanilla" else []
        )
        lora_ckpt_override = (
            # full-state checkpoints are ~55GB for a 7B (sharded model + optim)
            # and once filled a 60GB container disk mid-save; the adapter is
            # the only trained state, so save just it (~100-200MB). '+' because
            # the key is read via checkpoint_config.get() but not declared in
            # verl's hydra struct — the bare override is a ConfigCompositionException
            ["+actor_rollout_ref.actor.checkpoint.save_lora_only=True"] if self.lora_rank > 0 else []
        )
        # verl's model config defaults attn_implementation to flash_attention_2
        # and hard-errors at model load when the package is absent — it does
        # not fall back (sm100 pods provision with flash-attn skipped: no
        # cu130 wheel). find_spec, never an import: presence is all verl's
        # default keys on, and importing flash_attn can be slow or itself
        # broken. '+' because override_config is an open dict, not a declared
        # struct key. Stands down if the caller pinned any attn_implementation
        # via extra_overrides.
        attn_impl_override: list[str] = []
        if importlib.util.find_spec("flash_attn") is None and not any(
            "attn_implementation" in o for o in self.extra_overrides
        ):
            attn_impl_override = [
                "+actor_rollout_ref.model.override_config.attn_implementation=sdpa"
            ]
        return [
            *self._strategy_overrides(),
            *mode_override,
            *lora_ckpt_override,
            *attn_impl_override,
            f"actor_rollout_ref.model.path={self.model_path}",
            f"actor_rollout_ref.model.use_remove_padding={self.use_remove_padding}",
            "actor_rollout_ref.actor.use_dynamic_bsz=True",
            f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={self.max_token_len_per_gpu}",
            *clip_overrides,
            "actor_rollout_ref.actor.loss_agg_mode=token-mean",
            *(
                [
                    "actor_rollout_ref.actor.use_kl_loss=True",
                    f"actor_rollout_ref.actor.kl_loss_coef={self.kl_loss_coef}",
                    "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
                ]
                if self.kl_loss_coef > 0
                else ["actor_rollout_ref.actor.use_kl_loss=False"]
            ),
            "actor_rollout_ref.actor.entropy_coeff=0",
            f"actor_rollout_ref.actor.optim.lr={self.lr}",
            f"actor_rollout_ref.actor.optim.clip_grad={self.grad_clip}",
            "actor_rollout_ref.rollout.name=vllm",
            "actor_rollout_ref.rollout.mode=async",
            f"actor_rollout_ref.rollout.tensor_model_parallel_size={self.rollout_tp}",
            f"actor_rollout_ref.rollout.prompt_length={self.prompt_length}",
            f"actor_rollout_ref.rollout.response_length={self.response_length}",
            # engine KV sizing: without this vllm sizes for the MODEL'S native max
            f"actor_rollout_ref.rollout.max_model_len={self.prompt_length + self.response_length}",
            f"actor_rollout_ref.rollout.gpu_memory_utilization={self.gpu_memory_utilization}",
            # Pinned ON rather than trusting the engine default: debate contexts
            # are built for it (each seat's turn-t context strictly extends its
            # turn t-1 context; GRPO groups share the whole task prefix), and
            # the cache shares the existing KV pool via LRU — no extra memory
            # budget. If the pod's verl snapshot predates this key, hydra fails
            # at launch: drop the override there and upgrade verl.
            "actor_rollout_ref.rollout.enable_prefix_caching=True",
            "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True",
            f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={self.max_token_len_per_gpu}",
            "actor_rollout_ref.rollout.checkpoint_engine.backend=naive",
            f"trainer.n_gpus_per_node={self.n_gpus}",
            "trainer.nnodes=1",
            *self.extra_overrides,
        ]


def _reject_over_length_prompts(prompts: list[Tokens], prompt_length: int) -> None:
    """The rollout server truncates prompts beyond rollout.prompt_length
    without a word; the configs promise over-length rows fail loudly instead
    of training on a silently clipped context."""
    for prompt in prompts:
        if len(prompt) > prompt_length:
            raise ValueError(
                f"prompt is {len(prompt)} tokens but training.verl.prompt_length "
                f"is {prompt_length}; the engine would truncate it silently. "
                "Raise prompt_length or shorten the prompt."
            )


THINK_MARKERS = ("<think>", "</think>")


def visible_text(tokenizer, tokens: list[int]) -> str:
    """Decode for TEXT consumers (speech splicing, transcripts). Special-token
    strings must not survive into text that re-enters a chat template — they
    re-tokenize into real turn boundaries mid-message. Think markers stay:
    round.py splits on them."""
    text = tokenizer.decode(tokens)
    for special in getattr(tokenizer, "all_special_tokens", ()):
        if special not in THINK_MARKERS:
            text = text.replace(special, "")
    return text


def _token_weighted_loss_means(per_micro: list[tuple[dict, int]]) -> dict[str, float]:
    """(metrics, n_completion_tokens) per micro-batch -> loss/* means weighted
    by token count. The engine's metrics are token-means WITHIN a micro-batch,
    and with use_dynamic_bsz the micro-batches carry unequal token counts — an
    unweighted mean over them would let the small ones dominate."""
    sums: dict[str, float] = {}
    weights: dict[str, float] = {}
    for out_metrics, n_tokens in per_micro:
        for k, v in dict(out_metrics).items():
            try:
                val = float(v)
            except (TypeError, ValueError):
                continue
            sums[f"loss/{k}"] = sums.get(f"loss/{k}", 0.0) + val * n_tokens
            weights[f"loss/{k}"] = weights.get(f"loss/{k}", 0.0) + n_tokens
    return {k: v / weights[k] if weights[k] else 0.0 for k, v in sums.items()}


def _effective_grad_clip(config) -> float:
    """Last clip_grad in extra_overrides wins over config.grad_clip (hydra last-wins)."""
    for o in reversed(tuple(config.extra_overrides)):
        key, sep, val = str(o).partition("=")
        if sep and key.lstrip("+").endswith("actor.optim.clip_grad"):
            return float(val)
    return float(config.grad_clip)


def _grad_norm_metrics(grad_norm: float, clip_norm: float) -> dict[str, float]:
    """Engine reports the pre-clip norm; post-clip = min(pre, clip). Non-finite
    means the engine skipped the update, so no post-clip value exists."""
    if not math.isfinite(grad_norm):
        return {"optim/nonfinite_grad_step": 1.0}
    return {
        "optim/nonfinite_grad_step": 0.0,
        "optim/grad_norm_clipped": min(grad_norm, clip_norm),
    }


def _release_training_cache(*_args, **_kwargs):
    """Run inside one training worker; must stay module-level picklable.

    The driver dispatches this function to every worker via RayWorkerGroup's
    ``execute_all_sync`` API. Return the worker's global rank with MiB
    (allocated, reserved) before and after the flush so the driver can prove
    that every configured training rank completed before waking rollout.
    """
    import gc

    import torch

    mib = 1024 * 1024
    rank = int(os.environ["RANK"])
    before = (torch.cuda.memory_allocated() // mib, torch.cuda.memory_reserved() // mib)
    gc.collect()
    torch.cuda.empty_cache()
    after = (torch.cuda.memory_allocated() // mib, torch.cuda.memory_reserved() // mib)
    return {"rank": rank, "success": True, "before": before, "after": after}


def _validate_training_cache_release_stats(stats, expected_ranks: int) -> list[dict]:
    """Require one finite, successful allocator-flush record per rank."""
    if (
        not isinstance(expected_ranks, int)
        or isinstance(expected_ranks, bool)
        or expected_ranks < 1
    ):
        raise RuntimeError(f"invalid configured training rank count: {expected_ranks!r}")
    if not isinstance(stats, list):
        raise RuntimeError(
            "all-rank training cache release returned an invalid result container "
            f"{type(stats).__name__}; expected list"
        )
    if len(stats) != expected_ranks:
        raise RuntimeError(
            "all-rank training cache release cardinality mismatch: "
            f"expected {expected_ranks}, got {len(stats)}"
        )

    by_rank: dict[int, dict] = {}
    for index, record in enumerate(stats):
        if not isinstance(record, dict):
            raise RuntimeError(
                f"training cache release result {index} is not a mapping: {record!r}"
            )
        rank = record.get("rank")
        if not isinstance(rank, int) or isinstance(rank, bool):
            raise RuntimeError(
                f"training cache release result {index} has invalid rank: {rank!r}"
            )
        if rank < 0 or rank >= expected_ranks:
            raise RuntimeError(
                f"training cache release result {index} has out-of-range rank {rank}; "
                f"expected 0..{expected_ranks - 1}"
            )
        if rank in by_rank:
            raise RuntimeError(f"duplicate training cache release result for rank {rank}")
        if record.get("success") is not True:
            raise RuntimeError(f"training cache release failed on rank {rank}: {record!r}")

        for phase in ("before", "after"):
            values = record.get(phase)
            if not isinstance(values, (list, tuple)) or len(values) != 2:
                raise RuntimeError(
                    f"training cache release rank {rank} has invalid {phase} telemetry: "
                    f"{values!r}"
                )
            for value in values:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise RuntimeError(
                        f"training cache release rank {rank} has invalid {phase} telemetry: "
                        f"{values!r}"
                    )
                try:
                    finite = math.isfinite(value)
                except (OverflowError, TypeError, ValueError):
                    finite = False
                if not finite or value < 0:
                    raise RuntimeError(
                        f"training cache release rank {rank} has nonfinite or negative "
                        f"{phase} telemetry: {values!r}"
                    )
        by_rank[rank] = record

    missing = sorted(set(range(expected_ranks)) - set(by_rank))
    if missing:
        raise RuntimeError(f"missing training cache release results for ranks {missing}")
    return [by_rank[rank] for rank in range(expected_ranks)]


class VerlBackend(Backend):
    def __init__(self, config: VerlBackendConfig):
        import ray
        from hydra import compose, initialize_config_dir
        from transformers import AutoTokenizer

        import verl.trainer.config as verl_config_pkg
        from verl.checkpoint_engine import CheckpointEngineManager
        from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
        from verl.utils.config import omega_conf_to_dataclass
        from verl.workers.engine_workers_tinker import TinkerActorRolloutRefWorker
        from verl.workers.rollout.llm_server import LLMServerManager

        self.config = config
        self._grad_clip_norm = _effective_grad_clip(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)

        config_dir = os.path.join(os.path.dirname(verl_config_pkg.__file__))
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            self.verl_config = compose(config_name=config.config_name, overrides=config.hydra_overrides())

        if not ray.is_initialized():
            # expandable_segments lets the training allocator hand freed
            # segments back to CUDA, so the slept vLLM's wake-time re-pin can
            # claim them. Without it, MB-length steps (65k-token packing
            # units, ~79GB train peak) OOM inside wake_up at every
            # gpu_memory_utilization that can still serve a 65k KV
            # (2026-08-05 smoke + memprobe bisection).
            ray.init(
                runtime_env={
                    "env_vars": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
                }
            )

        pool = RayResourcePool(process_on_nodes=[config.n_gpus])
        self.wg = RayWorkerGroup(
            resource_pool=pool,
            ray_cls_with_init=RayClassWithInitArgs(
                cls=ray.remote(TinkerActorRolloutRefWorker),
                config=self.verl_config.actor_rollout_ref,
                role="actor_rollout",
            ),
        )
        self.wg.init_model()

        self.server_manager = LLMServerManager.create(config=self.verl_config, worker_group=self.wg)
        self.client = self.server_manager.get_client()
        self.checkpoint_manager = CheckpointEngineManager(
            config=omega_conf_to_dataclass(self.verl_config.actor_rollout_ref.rollout.checkpoint_engine),
            actor_wg=self.wg,
            replicas=self.server_manager.get_replicas(),
        )
        self.checkpoint_manager.sleep_replicas()

        self.wg.optimizer_zero_grad()
        self._rollout_awake = False
        self._global_step = 0
        self._last_temperature = 1.0
        self._pending_fwd_bwd: list = []

    # ------------------------------------------------------------- sampling

    def sync_sampler(self) -> None:
        # Flush every training allocator BEFORE the slept vLLM re-pins its
        # pools: wake-time create_and_map competes for the same physical
        # memory, and at MB lengths (65k packing units, ~79GB train peak) the
        # cached-but-free segments alone are the difference between wake and
        # OOM (2026-08-05 bisection). RayWorkerGroup.execute_all_sync invokes
        # the named worker method once per rank and resolves every Ray result.
        expected_ranks = self.config.n_gpus
        if getattr(self.wg, "world_size", None) != expected_ranks:
            raise RuntimeError(
                "training worker-group cardinality mismatch before cache release: "
                f"configured {expected_ranks}, worker group has "
                f"{getattr(self.wg, 'world_size', None)!r}"
            )
        execute_all_sync = getattr(self.wg, "execute_all_sync", None)
        if not callable(execute_all_sync):
            raise RuntimeError(
                "installed Verl RayWorkerGroup does not support execute_all_sync; "
                "refusing to wake rollout without an all-rank cache release"
            )
        stats = execute_all_sync("execute_func_rank_zero", _release_training_cache)
        stats = _validate_training_cache_release_stats(stats, expected_ranks)
        print(f"[verl] train-actor CUDA MiB (alloc/reserved) before->after flush: {stats}")
        # Wakes the rollout engine and pushes current FSDP weights (CUDA IPC).
        self.checkpoint_manager.update_weights(self._global_step)
        self._rollout_awake = True

    def _sleep_rollout(self) -> None:
        if self._rollout_awake:
            self.checkpoint_manager.sleep_replicas()
            self._rollout_awake = False

    def sample(
        self, prompts: list[Tokens], params: SamplingParams, n: int = 1
    ) -> list[list[Sample]]:
        # Before waking the engine: an over-length prompt must not cost a
        # weight sync just to fail.
        _reject_over_length_prompts(prompts, self.config.prompt_length)
        if not self._rollout_awake:
            self.sync_sampler()
        self._last_temperature = params.temperature
        sampling_params = {
            "temperature": params.temperature,
            "top_p": params.top_p,
            "max_tokens": params.max_tokens,
            "logprobs": True,  # bool at this boundary; forwarded as logprobs=0
        }
        if params.stop:
            sampling_params["stop"] = list(params.stop)
        if params.min_tokens:
            sampling_params["min_tokens"] = int(params.min_tokens)
        # Defensive EOS pin (run-verified in the prior repo): stops a natural
        # completion at EOS instead of padding out to max_tokens.
        if self.tokenizer.eos_token_id is not None:
            sampling_params["stop_token_ids"] = [int(self.tokenizer.eos_token_id)]

        async def _generate_all():
            return await asyncio.gather(
                *[
                    self.client.generate(
                        request_id=uuid.uuid4().hex,
                        prompt_ids=prompt,
                        sampling_params=dict(sampling_params),
                    )
                    for prompt in prompts
                    for _ in range(n)
                ]
            )

        outputs = asyncio.run(_generate_all())
        result: list[list[Sample]] = []
        for i in range(len(prompts)):
            group = []
            for out in outputs[i * n : (i + 1) * n]:
                tokens = list(out.token_ids)
                # verl's TokenOutput.stop_reason is 'completed' | 'aborted' |
                # None (replica.py:39) — it never carries "length", so it cannot
                # tell us whether generation was truncated. Hitting max_tokens
                # exactly is that signal, and it is the one the base contract
                # ("stop" | "length") asks for.
                truncated = params.max_tokens is not None and len(tokens) >= params.max_tokens
                group.append(
                    Sample(
                        tokens=tokens,
                        logprobs=list(out.log_probs) if out.log_probs is not None else [],
                        text=visible_text(self.tokenizer, tokens),
                        stop_reason="length" if truncated else "stop",
                    )
                )
            result.append(group)
        return result

    # ------------------------------------------------------------- training

    def _pack(self, data: list[Datum], for_update: bool):
        """Datums -> verl no-padding TensorDict (+ per-datum response lengths).

        Classic layout first (left-pad prompt, right-pad response), then verl's
        own left_right_2_no_padding converts to jagged. Datum.mask folds into
        response_mask, which becomes loss_mask.
        """
        import torch
        from verl import DataProto
        from verl.utils import tensordict_utils as tu
        from verl.utils.model import compute_position_id_with_mask
        from verl.workers.utils.padding import left_right_2_no_padding

        pad_id = self.tokenizer.pad_token_id or 0
        max_p = max(d.prompt_len for d in data)
        max_r = max(len(d.tokens) - d.prompt_len for d in data)
        B = len(data)

        prompts = torch.full((B, max_p), pad_id, dtype=torch.long)
        responses = torch.full((B, max_r), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((B, max_p + max_r), dtype=torch.long)
        response_mask = torch.zeros((B, max_r), dtype=torch.float32)
        old_log_probs = torch.zeros((B, max_r), dtype=torch.float32)
        advantages = torch.zeros((B, max_r), dtype=torch.float32)

        resp_lens = []
        for i, d in enumerate(data):
            p, r = d.prompt_len, len(d.tokens) - d.prompt_len
            resp_lens.append(r)
            prompts[i, max_p - p :] = torch.tensor(d.tokens[:p], dtype=torch.long)
            responses[i, :r] = torch.tensor(d.tokens[p:], dtype=torch.long)
            attention_mask[i, max_p - p : max_p + r] = 1
            mask = d.mask if d.mask is not None else [1.0] * r
            response_mask[i, :r] = torch.tensor(mask, dtype=torch.float32)
            old_log_probs[i, :r] = torch.tensor(d.sampler_logprobs, dtype=torch.float32)
            advantages[i, :r] = torch.tensor(d.completion_advantages, dtype=torch.float32)

        input_ids = torch.cat([prompts, responses], dim=1)
        position_ids = compute_position_id_with_mask(attention_mask)

        tensors = {
            "input_ids": input_ids,
            "prompts": prompts,
            "responses": responses,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "response_mask": response_mask,
        }
        if for_update:
            tensors["old_log_probs"] = old_log_probs
            tensors["advantages"] = advantages
            if any(d.ref_logprobs is not None for d in data):
                # In-loss KL (kl_mechanism "loss"): every datum must carry the
                # frozen-ref logprobs or none — a partial batch would pair
                # zeros against real logprobs inside verl's kl_loss term.
                if not all(d.ref_logprobs is not None for d in data):
                    raise ValueError(
                        "ref_logprobs set on some datums but not all; the "
                        "train loop stamps every datum or none"
                    )
                ref_lp = torch.zeros((B, max_r), dtype=torch.float32)
                for i, d in enumerate(data):
                    r = len(d.tokens) - d.prompt_len
                    ref_lp[i, :r] = torch.tensor(d.ref_logprobs, dtype=torch.float32)
                tensors["ref_log_prob"] = ref_lp

        meta = {
            "temperature": self._last_temperature,
            "global_token_num": attention_mask.sum(dim=-1).tolist(),
        }
        if not for_update:
            # infer_batch defaults compute_loss=True and would run the baked-in
            # ppo_loss (which requires the update keys) during pure logprob
            # extraction.
            meta["compute_loss"] = False
        proto = DataProto.from_single_dict(tensors, meta_info=meta)
        td = left_right_2_no_padding(proto.to_tensordict())
        # The engine branches packed-vs-padded on THIS batch key (default True),
        # not on the hydra model config — omit it and rmpad silently stays on.
        tu.assign_non_tensor(
            td, global_batch_size=B, use_remove_padding=self.config.use_remove_padding
        )
        return td, resp_lens

    def _pad_to_world_size(self, data: list[Datum]) -> list[Datum]:
        """nd dispatch chunks into world_size pieces with no auto-padding; pad
        with zero-mask copies of the last datum. No-op for the gradient
        DIRECTION (every loss term masks the row to zero); under seq-mean
        aggregation the padded row still counts in the stamped
        global_batch_size denominator, a uniform scale Adam absorbs."""
        ws = self.wg.world_size
        pad = (-len(data)) % ws
        if pad:
            d = data[-1]
            n = len(d.tokens) - d.prompt_len
            # All-zero mask IS safe at the pin: agg_loss's seq-mean branch
            # divides by (seq_mask + 1e-8), so a fully-masked row yields 0,
            # not NaN, and contributes nothing to ANY loss term — including
            # kl_loss, whose kld is masked by the same response_mask. (A
            # round-2 audit briefly swapped in one live token here; round 3
            # showed that token would carry a real KL gradient under
            # kl_mechanism 'loss' while the NaN it guarded against cannot
            # occur. Reverted.) ref_logprobs copied so the all-or-none check
            # in _pack stays consistent.
            filler = Datum(
                tokens=d.tokens,
                prompt_len=d.prompt_len,
                sampler_logprobs=d.sampler_logprobs,
                advantages=[0.0] * n,
                mask=[0.0] * n,
                ref_logprobs=d.ref_logprobs,
            )
            data = data + [filler] * pad
        return data

    def forward(self, data: list[Datum]) -> list[list[float]]:
        return self._forward(data, ref=False)

    def ref_logprobs(self, data: list[Datum]) -> list[list[float]]:
        """LoRA reference = the base weights: forward with the adapter
        disabled (TrainingWorker.infer_batch's no_lora_adapter flag)."""
        if self.config.lora_rank <= 0:
            raise NotImplementedError("verl ref_logprobs requires LoRA (ref = adapter-disabled base)")
        return self._forward(data, ref=True)

    def _forward(self, data: list[Datum], ref: bool) -> list[list[float]]:
        from verl.utils import tensordict_utils as tu
        from verl.workers.utils.padding import no_padding_2_padding

        self._sleep_rollout()
        n_real = len(data)
        padded = self._pad_to_world_size(data)
        td, resp_lens = self._pack(padded, for_update=False)
        if ref:
            tu.assign_non_tensor(td, no_lora_adapter=True)
        output = self.wg.compute_log_prob(td)
        log_probs = no_padding_2_padding(tu.get(output, "log_probs").cpu(), td)
        return [log_probs[i, : resp_lens[i]].tolist() for i in range(n_real)]

    def forward_backward(self, data: list[Datum], loss: LossSpec) -> dict[str, float]:
        assert loss.kind == self.config.loss.kind, (
            "verl bakes the loss into the worker at init; construct VerlBackend "
            f"with loss={loss.kind!r} (got {self.config.loss.kind!r})"
        )
        if data and (self.config.kl_loss_coef > 0) != (data[0].ref_logprobs is not None):
            # The loud direction (coef set, no ref) would KeyError inside verl;
            # THIS direction — ref stamped but coef 0 — would silently train
            # with no KL at all (round-3 audit). Both are config skew.
            raise RuntimeError(
                f"in-loss KL config skew: kl_loss_coef={self.config.kl_loss_coef} "
                f"but datums {'carry' if data[0].ref_logprobs is not None else 'lack'} "
                "ref_logprobs — kl_mechanism 'loss' needs both sides wired "
                "(build_backend does this from training.kl_mechanism)."
            )
        self._sleep_rollout()
        padded = self._pad_to_world_size(data)
        td, _ = self._pack(padded, for_update=True)
        n_tokens = sum(len(d.tokens) - d.prompt_len for d in data)
        future = self.wg.forward_backward(td)
        self._pending_fwd_bwd.append((future, n_tokens))
        return {}

    def optim_step(self, params: OptimParams) -> dict[str, float]:
        from verl.utils import tensordict_utils as tu

        metrics: dict[str, float] = _token_weighted_loss_means(
            [
                (dict(tu.get(future.get(), "metrics") or {}), n_tokens)
                for future, n_tokens in self._pending_fwd_bwd
            ]
        )
        self._pending_fwd_bwd.clear()

        # grad_clip is baked into the engine config; per-call override unsupported.
        step_metrics = self.wg.optimizer_step(
            {
                "lr": params.lr,
                "betas": tuple(params.betas),
                "eps": params.eps,
                "weight_decay": params.weight_decay,
            }
        )
        for m in step_metrics:
            for k, v in (m or {}).items():
                metrics[f"optim/{k}"] = float(v)
        if "optim/grad_norm" in metrics:
            metrics.update(_grad_norm_metrics(metrics["optim/grad_norm"], self._grad_clip_norm))
        self._global_step += 1
        return metrics

    # ------------------------------------------------------------ lifecycle

    def save(self, name: str) -> str:
        path = os.path.abspath(os.path.join(self.config.checkpoint_dir, name))
        self.wg.save_checkpoint(path, global_step=self._global_step)
        return path

    def load(self, path: str) -> None:
        self.wg.load_checkpoint(path)
