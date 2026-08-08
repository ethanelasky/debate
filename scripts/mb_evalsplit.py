"""Eval-only pass over the 250-row MB EVAL split for the Qwen3.6 RLVR arms.

Renders the exact PlannedEnv rollout shape the scratchpad RLVR runs train with
(plan turn + blind answer turn, enable_thinking=False, greedy decoding via
train.evaluate) against either the base model or a tinker checkpoint, without
touching the training loop. Used for the 2026-08-07 final numbers — see the
result block in configs/mb_rlvr.yaml and outputs/mb_evalsplit_20260807/.

Usage (repo root, TINKER_API_KEY in the environment):

    python scripts/mb_evalsplit.py base out/base.jsonl
    python scripts/mb_evalsplit.py tinker://<run>/weights/final out/final.jsonl

PROMPT VERSION IS NOT PINNED HERE: the rendered cue is whatever
infra/prompts/tasks/monitoringbench.yaml says in the working tree. Numbers are
only comparable to a training run if that file matches the run's cue — verify
against the run's transcript samples, not its recorded git commit (launches
have gone out with uncommitted prompt edits; see the 2026-08-07 launch-offset
incident in configs/mb_rlvr.yaml).

SAFETY: the output jsonl carries ids/labels/correctness only — never record
contents (data/monitoringbench rows are red-team attack trajectories).
"""

from __future__ import annotations

import json
import sys

import tinker
from tinker import types

from infra.backend.base import Backend, Sample, SamplingParams, Tokens
from infra.envs.base import Policy
from infra.envs.planned import PlannedEnv
from infra.envs.tasks import get_family
from infra.train import evaluate

MODEL = "Qwen/Qwen3.6-35B-A3B"
EVAL_N = 250  # the whole eval split
MAX_TOKENS = 1000  # = mb_rlvr_qwen36's max_completion_tokens / eval_max_tokens
PLAN_TOKENS = 1000  # = the scratchpad arm's plan_tokens

DATASET = {
    "files": [
        "data/monitoringbench/honest_mb550_norm.jsonl",
        "data/monitoringbench/attacks_mb550_norm.jsonl",
    ],
    "seed": 0,
    "split_files": {
        "train": "data/monitoringbench/splits/train_ids.txt",
        "test": "data/monitoringbench/splits/eval_ids.txt",  # FINAL NUMBERS split
    },
}


class SamplerOnlyBackend(Backend):
    """TinkerBackend.sample without a training client: greedy eval needs
    sampling and a tokenizer, nothing else. "weights/..." checkpoints are
    training state, so they round-trip through a training client to become
    sampler weights."""

    def __init__(self, model_path: str):
        self.service_client = tinker.ServiceClient()
        if model_path.startswith("tinker://"):
            tc = self.service_client.create_training_client_from_state(model_path)
            self._sampling_client = tc.save_weights_and_get_sampling_client()
        else:
            self._sampling_client = self.service_client.create_sampling_client(
                base_model=model_path
            )
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)

    def sample(self, prompts: list[Tokens], params: SamplingParams, n: int = 1):
        sp = types.SamplingParams(
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            stop=params.stop,
        )
        futures = [
            self._sampling_client.sample(
                prompt=types.ModelInput.from_ints(p), num_samples=n, sampling_params=sp
            )
            for p in prompts
        ]
        out = []
        for fut in futures:
            seqs = fut.result().sequences
            out.append(
                [
                    Sample(
                        tokens=list(seq.tokens),
                        logprobs=list(seq.logprobs) if seq.logprobs is not None else [],
                        text=self.tokenizer.decode(seq.tokens),
                        stop_reason=str(seq.stop_reason),
                    )
                    for seq in seqs
                ]
            )
        return out

    # unused Backend surface — this backend only ever samples
    def sync_sampler(self) -> None:
        pass

    def forward(self, *a, **k):
        raise NotImplementedError

    def forward_backward(self, *a, **k):
        raise NotImplementedError

    def optim_step(self, *a, **k):
        raise NotImplementedError

    def save(self, *a, **k):
        raise NotImplementedError

    def load(self, *a, **k):
        raise NotImplementedError


def main() -> None:
    model_path, out_path = sys.argv[1], sys.argv[2]
    family = get_family("monitoringbench")
    env = PlannedEnv(family.source(dict(DATASET)), PLAN_TOKENS)

    backend = SamplerOnlyBackend(MODEL if model_path == "base" else model_path)
    policy = Policy(
        backend,
        SamplingParams(max_tokens=MAX_TOKENS, temperature=1.0, top_p=1.0),
        {"enable_thinking": False},
    )

    metrics = evaluate(env, policy, EVAL_N)  # evaluate() decodes greedily
    print(json.dumps(metrics, indent=2, sort_keys=True))

    with open(out_path, "w") as f:
        for r in env.last_rollout_records:
            meta, info = r["meta"], r["info"]
            f.write(
                json.dumps(
                    {
                        "id": meta.get("task_id"),
                        "label": meta.get("label"),
                        "correct": info.get("correct"),
                        "chose_attack": info.get("chose_attack"),
                        "answer_tag": info.get("answer_tag"),
                        "stop_reason": r["stop_reason"],
                    }
                )
                + "\n"
            )
    print(f"wrote {len(env.last_rollout_records)} rows -> {out_path}")


if __name__ == "__main__":
    main()
