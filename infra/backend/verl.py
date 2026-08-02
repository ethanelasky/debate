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
    n_gpus: int = 8
    prompt_length: int = 2048
    response_length: int = 2048
    gpu_memory_utilization: float = 0.6
    rollout_tp: int = 1
    strategy: str = "fsdp2"  # "fsdp" | "fsdp2" | "megatron"
    max_token_len_per_gpu: int = 16384
    lora_rank: int = 0  # 0 = full finetune; 32 matches the Tinker backend
    lr: float = 1e-6  # initial; overridden per optim_step call
    grad_clip: float = 1.0
    checkpoint_dir: str = "checkpoints/verl"
    loss: LossSpec = field(default_factory=LossSpec)
    # Megatron-only parallelism (ignored on fsdp*). TinkerTrainingWorker is
    # engine-agnostic (it drives BaseEngine), so only config selection differs.
    megatron_tp: int = 1
    megatron_pp: int = 1
    megatron_cp: int = 1
    megatron_ep: int = 1
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
        eps_low = round(1.0 - self.loss.clip_low, 6)
        eps_high = round(self.loss.clip_high - 1.0, 6)
        if self.loss.kind == "importance_sampling":
            eps_low, eps_high = 1.0, 1000.0  # effectively unclipped
        return [
            *self._strategy_overrides(),
            f"actor_rollout_ref.model.path={self.model_path}",
            "actor_rollout_ref.model.use_remove_padding=True",
            "actor_rollout_ref.actor.use_dynamic_bsz=True",
            f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={self.max_token_len_per_gpu}",
            f"actor_rollout_ref.actor.clip_ratio_low={eps_low}",
            f"actor_rollout_ref.actor.clip_ratio_high={eps_high}",
            "actor_rollout_ref.actor.loss_agg_mode=token-mean",
            "actor_rollout_ref.actor.use_kl_loss=False",
            "actor_rollout_ref.actor.entropy_coeff=0",
            f"actor_rollout_ref.actor.optim.lr={self.lr}",
            f"actor_rollout_ref.actor.optim.clip_grad={self.grad_clip}",
            "actor_rollout_ref.rollout.name=vllm",
            "actor_rollout_ref.rollout.mode=async",
            f"actor_rollout_ref.rollout.tensor_model_parallel_size={self.rollout_tp}",
            f"actor_rollout_ref.rollout.prompt_length={self.prompt_length}",
            f"actor_rollout_ref.rollout.response_length={self.response_length}",
            f"actor_rollout_ref.rollout.gpu_memory_utilization={self.gpu_memory_utilization}",
            "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True",
            f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={self.max_token_len_per_gpu}",
            "actor_rollout_ref.rollout.checkpoint_engine.backend=naive",
            f"trainer.n_gpus_per_node={self.n_gpus}",
            "trainer.nnodes=1",
            *self.extra_overrides,
        ]


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
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)

        config_dir = os.path.join(os.path.dirname(verl_config_pkg.__file__))
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            self.verl_config = compose(config_name=config.config_name, overrides=config.hydra_overrides())

        if not ray.is_initialized():
            ray.init()

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
                # vLLM's stop_reason can be the matched stop STRING; normalize
                # to the base contract ("stop" | "length").
                raw_reason = str(out.stop_reason or "stop")
                group.append(
                    Sample(
                        tokens=tokens,
                        logprobs=list(out.log_probs) if out.log_probs is not None else [],
                        text=self.tokenizer.decode(tokens),
                        stop_reason="length" if raw_reason == "length" else "stop",
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
        tu.assign_non_tensor(td, global_batch_size=B)
        return td, resp_lens

    def _pad_to_world_size(self, data: list[Datum]) -> list[Datum]:
        """nd dispatch chunks into world_size pieces with no auto-padding; pad
        with zero-mask copies of the last datum (zero loss, zero tokens in the
        loss normalization — exact no-op for the gradient)."""
        ws = self.wg.world_size
        pad = (-len(data)) % ws
        if pad:
            d = data[-1]
            n = len(d.tokens) - d.prompt_len
            filler = Datum(
                tokens=d.tokens,
                prompt_len=d.prompt_len,
                sampler_logprobs=d.sampler_logprobs,
                advantages=[0.0] * n,
                mask=[0.0] * n,
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
        self._sleep_rollout()
        padded = self._pad_to_world_size(data)
        td, _ = self._pack(padded, for_update=True)
        n_tokens = sum(len(d.tokens) - d.prompt_len for d in data)
        future = self.wg.forward_backward(td)
        self._pending_fwd_bwd.append((future, n_tokens))
        return {}

    def optim_step(self, params: OptimParams) -> dict[str, float]:
        from verl.utils import tensordict_utils as tu

        metrics: dict[str, float] = {}
        for future, _ in self._pending_fwd_bwd:
            output = future.get()
            out_metrics = tu.get(output, "metrics") or {}
            for k, v in dict(out_metrics).items():
                try:
                    metrics[f"loss/{k}"] = metrics.get(f"loss/{k}", 0.0) + float(v)
                except (TypeError, ValueError):
                    pass
        n_micro = max(1, len(self._pending_fwd_bwd))
        metrics = {k: v / n_micro for k, v in metrics.items()}
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
        self._global_step += 1
        return metrics

    # ------------------------------------------------------------ lifecycle

    def save(self, name: str) -> str:
        path = os.path.abspath(os.path.join(self.config.checkpoint_dir, name))
        self.wg.save_checkpoint(path, global_step=self._global_step)
        return path

    def load(self, path: str) -> None:
        self.wg.load_checkpoint(path)
