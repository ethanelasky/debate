# Debate RL scaffold — skeleton

One abstraction: a `Backend` with four compute endpoints (sample, forward, backward-via-forward_backward, optim step) plus weight lifecycle. Envs produce rewarded trajectories; the train loop is ~100 lines of glue. Eval is not a separate path — it's the same `rollout()` on the test split.

Target size: ~4.5–6k lines total (of which ~2–3k is the salvaged debate protocol).

```
infra/  (python package; repo stays ~/code/debate)
  backend/
    base.py      # Backend ABC + Sample/Datum/LossSpec/OptimParams   (~150)
    tinker.py    # TinkerBackend                                     (~200)
    verl.py      # VerlBackend                                       (~400–600)
  envs/
    base.py      # Env ABC + Trajectory, Policy/Model adapter        (~150)
    tasks/       # task registry: TaskFamily per domain
      math.py        # MathEnv (CS285 HW4 port) + MathFamily         (~250)
      codecontests.py# CodeContestsEnv + CodeContestsFamily
    debate/      # SALVAGED from ~/ai-debate/ai_debate/debate/, slimmed
      protocol.py    # Protocol enum + TurnPlan (port ~as-is)        (~400)
      round.py       # DebateRound speaker loop, provenance/replay/
                     # view-projection layers stripped               (~500–700)
      judge.py       # verdict parsing + decision-token logit scan   (~350)
      rewards.py     # ONE reward ladder (unify the two diverged
                     #   copies) + judge/gt/binary/continuous modes  (~150)
      env.py         # Env wrapper: rounds -> Trajectories/Datums    (~200)
  prompts/       # YAML configs, copied from old repo (data, not code)
  rl/
    advantages.py# GRPO centering (port ~verbatim from old repo)     (~60)
    datums.py    # trajectory -> Datum packing per backend layout    (~120)
  train.py       # loop + evaluate()                                 (~180)
```

## Core types

```python
Tokens = list[int]

@dataclass
class SamplingParams:
    max_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0
    stop: list[str] | None = None

@dataclass
class Sample:
    tokens: Tokens
    logprobs: list[float]   # sampler's own per-token logprobs — this IS old_log_probs.
                            # Never recompute the behavior anchor with the trainer
                            # (under LoRA, FSDP-recompute biases the PPO ratio).
    text: str
    stop_reason: str        # "stop" | "length"

@dataclass
class Datum:
    """One training sequence, unpadded. Backends own padding/layout."""
    tokens: Tokens                   # prompt + completion
    prompt_len: int
    sampler_logprobs: list[float]    # len == len(tokens) - prompt_len
    advantages: list[float]          # same length (usually one scalar broadcast)
    mask: list[float] | None = None  # multi-turn: 0.0 on non-policy tokens

@dataclass
class LossSpec:
    kind: Literal["ppo", "importance_sampling", "cross_entropy"]
    clip_low: float = 0.8
    clip_high: float = 1.2

@dataclass
class OptimParams:
    lr: float
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.0
    grad_clip: float = 1.0
```

## Backend

```python
class Backend(ABC):
    """Owns the policy weights. Four compute endpoints + lifecycle. Nothing above this.

    Contract:
      - sample() reflects weights as of the last sync_sampler(), NOT live weights.
        (Tinker samplers are frozen snapshots; VERL syncs into vLLM. Making this
        explicit keeps both backends honest about which policy generated a rollout.)
      - forward_backward() ACCUMULATES gradients; optim_step() applies and clears.
        Microbatching/grad-accum = N forward_backward calls, then one optim_step.
        There is no separate backward(): Tinker's primitive is fused, and VERL's
        per-microbatch loss.backward() is the same shape. Fused is the honest op.
      - forward() returns per-token logprobs with no gradient side effects
        (ref-model KL, diagnostics). Not needed for plain GRPO; keep it anyway —
        it's ~free on both backends.
    """

    @abstractmethod
    def sync_sampler(self) -> None: ...

    @abstractmethod
    def sample(self, prompts: list[Tokens], params: SamplingParams,
               n: int = 1) -> list[list[Sample]]: ...

    @abstractmethod
    def forward(self, data: list[Datum]) -> list[list[float]]: ...

    @abstractmethod
    def forward_backward(self, data: list[Datum],
                         loss: LossSpec) -> dict[str, float]: ...

    @abstractmethod
    def optim_step(self, params: OptimParams) -> dict[str, float]: ...

    @abstractmethod
    def save(self, name: str) -> str: ...

    @abstractmethod
    def load(self, path: str) -> None: ...
```

### TinkerBackend (~200 lines)

Near-1:1 mapping:

| endpoint | tinker call |
|---|---|
| `sync_sampler` | `training_client.save_weights_and_get_sampling_client()` |
| `sample` | `sampling_client.sample(...)` (logprobs returned by default) |
| `forward` | `forward(data, "cross_entropy")` with all-zero weights → read `loss_fn_outputs[i]["logprobs"]` |
| `forward_backward` | built-in `"ppo"` / `"importance_sampling"` with `target_tokens`/`logprobs`/`advantages` (+ off-by-one shift: model_input = prompt + completion[:-1]) |
| `optim_step` | `optim_step(AdamParams(...))` — LR is per-call, scheduling is trivial |
| `save`/`load` | `save_state` / `load_state_with_optimizer` |

Notes: `mask` is folded into `advantages` (`adv *= mask`) or passed as explicit
`TensorData` (raw-list coercion doesn't know the key). Pipeline
`forward_backward` + `optim_step` futures before awaiting either.

### VerlBackend (~450 lines) — RUN-VERIFIED 2026-08-01

Live smoke test passed on 2× RTX 4090 (RunPod, Qwen2.5-0.5B-Instruct, FSDP2):
construct → sync_sampler (CUDA-IPC weight push) → sample (logprobs + fidelity)
→ forward → forward_backward ×2 → optim_step (grad_norm returned) → sharded
save → second sync/sample cycle. Env recipe in `scripts/provision_pod.sh`
(pins vllm==0.24.*, single-arch flash-attn build, MAX_JOBS=2); prebuilt
flash-attn wheel for torch2.11+cu130/sm89 saved at `wheels/`.

VERL used as a worker pool, not a trainer (`RayPPOTrainer` is deprecated
upstream anyway). Target **verl main ≥ 0.9.0.dev, pinned to a commit**: it
ships `TinkerTrainingWorker` (`verl/workers/engine_workers_tinker.py`), which
exposes exactly our decomposed primitives as Ray RPCs. The old
`fsdp_workers.py` / `dp_actor.py` API was deleted in v0.8.0 — everything the
prior repo integrated against is gone.

| endpoint | verl 0.9 mechanism |
|---|---|
| `sync_sampler` | `wg.update_weights(mode="naive")` (ZMQ+CUDA-IPC into vLLM server) + `rollout.sleep()/wake_up()` around it for colocated setups |
| `sample` | `LLMServerClient.generate(request_id, prompt_ids, sampling_params)` → `TokenOutput{token_ids, log_probs, stop_reason}`; `sampling_params["logprobs"]` is a **bool** at this boundary |
| `forward` | `wg.compute_log_prob(td)` / `TrainingWorker.infer_batch` |
| `forward_backward` | `TinkerTrainingWorker.forward_backward(td)` — accumulates; grad sync suppressed on non-final microbatches; normalization via DP-all-reduced `batch_num_tokens`, so caller must set `global_batch_size` correctly |
| `optim_step` | `TinkerTrainingWorker.optimizer_step(OptimStepParams)` — `zero_grad_on_exit=True`, so it applies **and clears**, matching the contract verbatim; per-call `lr/betas/eps/weight_decay` like Tinker |
| `save`/`load` | `TrainingWorker.save_checkpoint/load_checkpoint` |

Ray ceremony is small and torch.distributed-free on our side:
`RayResourcePool(process_on_nodes=[N])` → `RayWorkerGroup(pool,
RayClassWithInitArgs(ray.remote(TinkerTrainingWorker), config=...))` →
`wg.reset()` → `wg.set_loss_fn(partial(ppo_loss, config=...))`.
`TrainingWorkerConfig` is a plain dataclass — no Hydra.

Data layer: the engine now wants **nested/jagged TensorDicts (NO_PADDING)**,
not the classic left/right-padded DataProto — build with
`tu.get_tensordict(...)` and the converters in `verl/workers/utils/padding.py`
(`left_right_2_no_padding` / `no_padding_2_padding`, which also handles the
logprob off-by-one). Update keys: `input_ids`, `position_ids`, `temperature`
(required), `response_mask`, `old_log_probs`, `advantages`, `prompts`,
`responses`, `loss_mask`, non-tensor `global_batch_size` et al. Loss: shipped
`ppo_loss` (loss fn signature `(config, model_output, data, dp_group) ->
(loss, metrics)`); registry names are `vanilla` (=PPO clip), `gspo`, `cispo`, …

**Version risk:** `TinkerTrainingWorker` is main-only (not in v0.8.0) and
`BaseEngine` is marked subject to change. Mitigation: pin the commit
(`e9618406de5bad40041d7612554e465ec2003ec1`), and if needed vendor the
~100-line worker (thin wrapper over stable `BaseEngine` methods
`optimizer_zero_grad` / `forward_backward_batch` / `optimizer_step`).

**Megatron:** free. `TinkerTrainingWorker` drives `BaseEngine`, which FSDP and
Megatron both implement — zero backend-code changes. Only config differs:
compose `ppo_megatron_trainer` (= `ppo_trainer` + `override model_engine:
megatron`), parallelism under `actor_rollout_ref.actor.megatron.{tp,pp,cp,ep}`,
and LoRA via the NESTED `model.lora.rank/alpha` block instead of the flat
`model.lora_rank/lora_alpha` the FSDP engine reads (both key shapes
run-verified in the prior repo). `VerlBackendConfig(strategy="megatron",
megatron_tp=..., ...)` handles all of it. One to-verify on hardware: per-call
`OptimStepParams` lr overrides require the Megatron optimizer to expose
`param_groups` (the Tinker worker raises `NotImplementedError` otherwise).

## Env

```python
@dataclass
class Trajectory:
    datums: list[Datum]   # one per policy turn (debate: one per speech by a trained seat)
    reward: float         # advantages filled in later by GRPO centering
    info: dict            # correctness, verdict, transcript, ...

class Policy:
    """Thin: chat-template rendering over backend.sample(). Also usable against
    a plain OpenAI-compatible endpoint for offline eval of arbitrary models —
    same env code either way. Presents the old repo's `Model.predict` batch
    interface so the salvaged debate code runs unmodified (see DebateEnv)."""
    def predict(self, inputs: list[list[Message]], *, n: int = 1,
                max_new_tokens: int = ...) -> list[Sample]: ...

class Env(ABC):
    """ONE rollout path. Training calls rollout(train split, temp>0, group_size=G);
    evaluation is rollout(test split, temp 0, group_size=1) + mean(reward)/mean(info).
    No other eval machinery exists."""

    @abstractmethod
    def tasks(self, n: int, split: str = "train") -> list[Task]: ...

    @abstractmethod
    def rollout(self, tasks: list[Task], policy: Policy,
                group_size: int) -> list[list[Trajectory]]: ...
```

### MathEnv (~250 lines) — RLVR

Port of CS285 HW4 `MathHardTask`: Hendrycks MATH filtered to level 5, boxed-answer
parsing with numeric fallback, reward = correct(1.0) + format(0.1) shaping knobs.
Single-turn: one Datum per trajectory.

### DebateEnv — SALVAGED from `~/ai-debate/ai_debate/debate/`, slimmed

Not a rewrite. The protocol machinery encodes accumulated knowledge worth
keeping: protocol-as-data slot structure, per-round sequential/blind control
with final-round-always-blind, the judge logit scan with its BPE
surface-normalization gotcha, the reward ladder edge cases, and the prompt
YAML library. Port these, strip the apparatus.

**Port ~as-is:** the old repo's `protocol.py` (Protocol enum) + `turn_plan.py`
(compiled slot structure) — collapsed here into one `Protocol` dataclass, since
this repo has no separate format enum: the slot list IS the format;
`judge.py`'s verdict parsing + `scan_judge_logit`/
`score_decision_token`; `reward_ladder.py` — but unified into ONE module,
collapsing the two diverged copies (`experiments/verdict_verifier.py` vs
`debate/outcome.py`); the prompt YAMLs (`quality.yaml`,
`hendrycks_math.yaml`, `shared.yaml` + `_extends` resolution).

**Port the core, strip the layers:** `debate_round.py`'s speaker loop
(`run_round` driven by `get_next_expected_speaker`, batched transcripts) minus
its four separable add-on layers: generation-ledger provenance (~450 lines),
frozen-prefix resume/replay (~250), per-reader view projection (~200), and
`transcript_codec.py` entirely (transcripts serialize as plain
dataclass→JSON). Tool-use loops (~350) stay out until an env needs them.

**Interface decision that makes the salvage cheap:** the old debate code is
written against `Model.predict(inputs: list[list[ModelInput]], ...) ->
list[ModelResponse]`. Keep that as the env-facing policy interface — `Policy`
becomes a ~50-line `Model` adapter over `backend.sample()` (the new-world
`GeneratePortModelBridge`, minus its ceremony). This lets the salvaged round
code run unmodified, lets the judge stay a provider-gate OpenAI-compat
`Model`, and lets offline eval point the same env at any provider.

Semantics preserved from the skeleton:
- Two seats, same policy in both (self-play, one LoRA). Side = answer;
  models stay fixed to seats (the side-flip must swap *answers*, not models).
- Each speech by a trained seat = one Datum (transcript-so-far as prompt);
  seat's reward broadcast across that seat's speeches.
- Reward modes: `judge_continuous` (2·confidence − 1), `judge_binary`
  (±1 ladder), `gt` (RLVR fast path — skips speeches+judge, scores the
  proposal directly).
- Fidelity asserts per sample, reject trajectory on failure:
  len(tokens)==len(logprobs), decode(tokens)==text, stop_reason present.

`envs/debate/env.py` is the only new code: wraps round construction +
execution into `Env.tasks()/rollout()` and harvests per-speech Datums.

## Train loop (~180 lines, the whole thing)

```python
def train(env: Env, backend: Backend, cfg: Config):
    policy = Policy(backend, renderer, cfg.sampling)
    for step in range(cfg.steps):
        backend.sync_sampler()
        groups = env.rollout(env.tasks(cfg.batch_size), policy, cfg.group_size)
        datums = grpo_pack(groups)          # center rewards within group,
                                            # drop zero-variance groups,
                                            # broadcast advantages into datums
        for micro in chunks(datums, cfg.micro_batch):
            backend.forward_backward(micro, cfg.loss)
        metrics = backend.optim_step(cfg.adam)
        if step % cfg.eval_every == 0:
            metrics |= evaluate(env, policy, n=cfg.eval_n)   # same rollout path
        if step % cfg.save_every == 0:
            backend.save(f"step-{step}")
        log(metrics)                        # wandb.log(), nothing else

def evaluate(env, policy, n) -> dict:
    groups = env.rollout(env.tasks(n, split="test"),
                         policy.greedy(), group_size=1)
    return aggregate(t.reward, t.info for g in groups for t in g)
```

## Deliberately absent

Provenance envelopes, science digests, monitoring outboxes, attempt journals,
durable-facts state machines, transcript codecs, registries, port protocols
beyond the Backend ABC itself. Checkpoint-resume = `backend.load(path)` and
restart the loop. Logging = wandb.log of the metrics dicts. Config = one flat
dataclass per run.

## Salvage from ~/ai-debate (run-verified, despite doc headers claiming otherwise)

- `debate/` protocol machinery — the big salvage; see the DebateEnv section
  for the port/strip split (protocol, turn_plan, judge + logit scan, unified
  reward ladder, prompt YAMLs; speaker loop minus provenance/replay/view
  layers and transcript_codec)
- `train/core/advantages.py` (GRPO, 135 lines) — port near-verbatim
- `train/ppo_loss.py` (dual-clip PPO/GSPO, 424) — reference for a custom `set_loss_fn` loss if verl's shipped `ppo_loss` doesn't fit
- `models/provider_gate.py` (~300) — for Judge/offline-eval client concurrency
- ~~`train/backends/verl/datums.py`~~ — its left/right-padded layout targets
  the ≤0.7.1 API deleted in verl v0.8.0; verl 0.9 wants jagged TensorDicts
  with in-tree converters (`verl/workers/utils/padding.py`). Don't port; keep
  only as reference if pinning old verl.

## Resolved / open questions

1. ~~VERL decomposition~~ **Resolved**: `TinkerTrainingWorker` on verl main
   exposes `optimizer_zero_grad`/`forward_backward`/`optimizer_step` as
   separate RPCs; `optimizer_step(zero_grad_on_exit=True)` matches our
   contract verbatim. Pin a main commit (e.g. `e961840`); vendor the ~100-line
   worker if the pin must move.
2. Judge serving: co-hosted vLLM instance vs external API endpoint. Leaning external/OpenAI-compat first — simplest, and Policy/Judge share the client. (Verl's rollout server also speaks plain OpenAI HTTP at `http://{addr}/v1`, so the judge could hit a frozen replica.)
3. Ref-model KL: skip initially (GRPO without KL penalty); `forward()` exists if we want it.
4. Verl's `experimental/agent_loop` (AgentLoopBase + response_mask machinery) exists but we deliberately don't use it — multi-turn stays in our `Env` so the same rollout code runs on both backends; our `Datum.mask` plays the role of their `response_mask`.
