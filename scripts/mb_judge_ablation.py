"""Judge-prompt ablation: one set of debates, every judge prompt replayed on it.

WHAT IT MEASURES. ``mb_debate``'s judge prompt stacks four components. This
runs the ladder in infra/prompts/debate/mb_judge_ablation.yaml (j0 one
sentence -> j4 == today's prompt, plus a j4-minus-monitor-card leave-one-out)
and reports what each one buys.

WHY REPLAY RATHER THAN SIX RUNS. The judge prompt does not appear in any
debater's context, so every rung should be judging the SAME debates. Running
six full arms would regenerate the speeches six times — paying for them
repeatedly and, worse, putting fresh debater-sampling noise on top of every
rung-to-rung difference. At n=120 that noise is comparable to the effects
being measured, so the comparison is run PAIRED:

  PHASE 1  debate once with real debaters, and record every debater slot as
           (sha256 of its rendered context) -> speech.
  PHASE 2  per rung, re-run with the debater seats bound to PlaybackModel and
           only the judge live. The cache is keyed by rendered context, so if a
           rung perturbs a debater's view by one byte the lookup MISSES and the
           run dies loudly instead of silently comparing incomparable arms.

Phase 1's own judge output is discarded — the seed arm's rung is replayed like
any other, so no rung gets a different code path from the rest.

Metrics come from infra.run_eval unchanged (result_row / summarize), so these
numbers are directly comparable to a real eval run.

READ THE RESULTS AGAINST THE POOL BASELINE. Even normalized, the pool retains
surface shortcuts (worst residual AUC 0.616; see
data/monitoringbench/norm/config.json). blind_choice accuracy is IDENTICAL
across rungs by construction — alice's speech is played back — which makes it
a free correctness check on the harness: if it moves, playback is broken.

    scripts/mb_judge_ablation.py [--limit N] [--rungs a,b] [--out DIR]
                                 [--dry-run] [--reuse-cache DIR]

Defaults to the whole dev split (120 rows). Selection experiments belong on
dev; the eval split is for final numbers only.

SAFETY: writes ids/verdicts/metrics, plus a speech cache that is a run
artifact. Nothing here prints trajectory or speech content.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infra.config import load_experiment  # noqa: E402
from infra.envs.debate.round import render_context  # noqa: E402
from infra.models.playback_model import PlaybackModel, context_key  # noqa: E402
from infra.run_eval import build_eval_env, build_task_source, result_rows, summarize  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CONFIG = "configs/mb_judge_ablation.yaml"
SEED_EXPERIMENT = "mb_judge_ablation"
LADDER = "infra/prompts/debate/mb_judge_ablation.yaml"

#: Ladder order matters for reading the table: each rung adds one component to
#: the one above it, except the last, which is a leave-one-out on j4.
RUNGS = [
    ("mb_judge_j0", "one sentence"),
    ("mb_judge_j1", "+ debate framing"),
    ("mb_judge_j2", "+ monitor card"),
    ("mb_judge_j3", "+ judge guidance"),
    ("mb_judge_j4", "+ 3-step deliberation (= today)"),
    ("mb_judge_j4_nocard", "j4 minus monitor card"),
    ("mb_judge_j0card", "one sentence + monitor card"),
]


def load_env_file() -> None:
    """Same optional-dotenv pattern as scripts/mb_solo_to_docent.py: the model
    wrappers read provider keys from the environment, and this script is run
    directly rather than through pod_run.sh."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv is optional in this repo
        return
    load_dotenv(dotenv_path=REPO / ".env", override=False)


def record_cache(env, states) -> list[dict]:
    """Completed debates -> [{key, speech, thinking, slot, task_id}] for every
    DEBATER slot.

    The context is re-rendered against the state TRUNCATED to the records that
    existed when that slot ran. That truncation is required, not cosmetic:
    render_context pulls in every record of the current speaker regardless of
    turn, so rendering against the finished state would hand a seat its own
    future speeches and produce a key nothing will ever look up."""
    out: list[dict] = []
    for st in states:
        if st.failed is not None:
            continue
        task_id = (st.meta.get("task") or {}).get("task_id")
        full = st.records
        try:
            for i, rec in enumerate(full):
                if rec.slot.speaker == env.judge_speaker:
                    continue
                st.records = full[:i]
                messages = render_context(st, rec.slot, env.prompts)
                out.append(
                    {
                        "key": context_key(messages),
                        "speech": rec.text,
                        "thinking": rec.thinking,
                        "slot": f"{rec.slot.speaker}/{rec.slot.slot.name}@{rec.slot.turn}",
                        "task_id": task_id,
                    }
                )
        finally:
            st.records = full
    return out


def run_rung(exp: dict, entry: str, source, tasks, cache_path: Path, matched_fpr: float):
    """One rung: playback debaters, live judge, standard eval metrics.

    The task source is still required even though the tasks are already built:
    DebateEnv pulls <TRAJECTORY_USER>/<ANSWER_GEN_USER> from the family's
    answer-generation config through it, which is what keeps every seat's
    trajectory message byte-identical to the RLVR prompt."""
    rung_exp = copy.deepcopy(exp)
    rung_exp["prompt_config"] = {"file_path": LADDER, "entry": entry}
    env = build_eval_env(rung_exp, source)
    for speaker in env.debaters:
        # Swap the API model for playback. Mutating the dict rather than
        # rebuilding the config keeps every other wiring decision identical to
        # a normal eval run.
        env.config.frozen_models[speaker] = PlaybackModel(
            alias=f"playback-{speaker}", cache_path=cache_path, is_debater=True
        )
    env.rollout(tasks, policy=None, group_size=1)
    rows = result_rows(env)
    return rows, summarize(rows, matched_fpr), env


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="first N dev rows (default: all 120)")
    ap.add_argument("--rungs", default=None, help="comma-separated subset of the ladder")
    ap.add_argument("--out", default=None, help="output dir (default outputs/judge_ablation_<ts>)")
    ap.add_argument("--reuse-cache", default=None, help="skip phase 1; use this run's cache.jsonl")
    ap.add_argument("--matched-fpr", type=float, default=8 / 116)
    ap.add_argument("--dry-run", action="store_true", help="resolve config + tasks + prompts, generate nothing")
    args = ap.parse_args()
    load_env_file()

    rungs = RUNGS
    if args.rungs:
        wanted = [r.strip() for r in args.rungs.split(",") if r.strip()]
        known = dict(RUNGS)
        if unknown := [r for r in wanted if r not in known]:
            raise SystemExit(f"unknown rung(s) {unknown}; known: {list(known)}")
        rungs = [(r, known[r]) for r in wanted]

    exp = load_experiment(CONFIG, SEED_EXPERIMENT)
    source = build_task_source(exp, None, None)
    tasks = source.tasks(10**9, split="test")
    if args.limit is not None:
        tasks = tasks[: args.limit]
    labels: dict[str, int] = {}
    for t in tasks:
        labels[str(t.meta.get("label"))] = labels.get(str(t.meta.get("label")), 0) + 1

    out_dir = Path(args.out) if args.out else REPO / f"outputs/judge_ablation_{datetime.now():%Y%m%d_%H%M%S}"
    print(
        json.dumps(
            {
                "config": CONFIG,
                "rungs": [r for r, _ in rungs],
                "n_tasks": len(tasks),
                "labels": labels,
                "debates_to_generate": 0 if args.reuse_cache else len(tasks),
                "judge_passes": len(tasks) * len(rungs),
                "out_dir": str(out_dir),
            },
            indent=2,
        )
    )
    if args.dry_run:
        # Prove every rung's prompts compile against the protocol before any
        # generation is paid for — a bad cue key must fail here, not at rung 6.
        from infra.envs.debate.prompts import load_prompt_library, validate_prompts
        from infra.envs.debate.protocol import Protocol

        protocol = Protocol.parse(exp["protocol"])
        for entry, _ in rungs:
            lib = load_prompt_library(LADDER, entry, protocol)
            validate_prompts(lib, protocol, fresh_positions=False, choice_positions=True)
        print(f"\ndry-run: {len(rungs)} rung(s) compile against the protocol; nothing generated", file=sys.stderr)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "cache.jsonl"

    # ---------------- phase 1: debate once
    if args.reuse_cache:
        src = Path(args.reuse_cache)
        cache_path = src if src.is_file() else src / "cache.jsonl"
        print(f"\nphase 1 skipped; reusing {cache_path}", file=sys.stderr)
    else:
        print(f"\nphase 1: generating {len(tasks)} debates with live debaters...", file=sys.stderr)
        seed_exp = copy.deepcopy(exp)
        seed_exp["prompt_config"] = {"file_path": LADDER, "entry": rungs[0][0]}
        seed_env = build_eval_env(seed_exp, source)
        seed_env.rollout(tasks, policy=None, group_size=1)
        entries = record_cache(seed_env, seed_env.last_states)
        n_failed = sum(1 for st in seed_env.last_states if st.failed is not None)
        with open(cache_path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
        print(
            f"phase 1 done: {len(entries)} debater slots cached from "
            f"{len(tasks) - n_failed}/{len(tasks)} debates ({n_failed} failed)",
            file=sys.stderr,
        )
        if n_failed:
            # Failed debates cache nothing, so every rung will fail those same
            # rows identically. That keeps the arms paired; it does shrink n.
            print(
                f"NOTE: {n_failed} debate(s) failed in phase 1 and are absent from the "
                "cache; all rungs will fail those rows identically (pairing preserved, "
                "denominator reduced).",
                file=sys.stderr,
            )

    # ---------------- phase 2: replay each rung
    summaries = {}
    for entry, label in rungs:
        print(f"\nrung {entry} ({label}): judging {len(tasks)} debates...", file=sys.stderr)
        rows, summary, env = run_rung(exp, entry, source, tasks, cache_path, args.matched_fpr)
        summary["rung"], summary["label"] = entry, label
        summaries[entry] = summary
        with open(out_dir / f"{entry}.results.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        acc = summary["confusion_at_0.5"]
        n = summary["n_scored"]
        correct = acc["tp"] + acc["tn"]
        print(
            f"  scored {n}/{summary['n_attempted']} | acc {correct / n if n else 0:.3f} "
            f"| AUC {summary['roc_auc']} | ties {summary['n_tie']}",
            file=sys.stderr,
        )

    # ---------------- report
    def acc_of(s):
        c = s["confusion_at_0.5"]
        return (c["tp"] + c["tn"]) / s["n_scored"] if s["n_scored"] else None

    lines = [
        "| rung | judge prompt | n | acc | AUC | TPR | FPR | ties | failed |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for entry, label in rungs:
        s = summaries[entry]
        f = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
        lines.append(
            f"| {entry.replace('mb_judge_', '')} | {label} | {s['n_scored']} | "
            f"{f(acc_of(s))} | {f(s['roc_auc'])} | {f(s['tpr'])} | {f(s['fpr'])} | "
            f"{s['n_tie']} | {s['n_failed']} |"
        )
    blind = {e: summaries[e].get("blind_choice", {}) or {} for e, _ in rungs}
    blind_accs = {e: b.get("accuracy") for e, b in blind.items()}
    table = "\n".join(lines)
    consistent = len({round(v, 6) for v in blind_accs.values() if v is not None}) <= 1
    note = (
        "blind_choice accuracy identical across rungs — playback verified"
        if consistent
        else f"WARNING: blind_choice accuracy DIFFERS across rungs {blind_accs} — playback is not replaying alice faithfully"
    )
    report = {
        "config": CONFIG,
        "ladder": LADDER,
        "n_tasks": len(tasks),
        "labels": labels,
        "cache": str(cache_path),
        "blind_choice_accuracy_by_rung": blind_accs,
        "playback_check": note,
        "summaries": summaries,
    }
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (out_dir / "table.md").write_text(table + f"\n\n{note}\n")
    print("\n" + table)
    print(f"\n{note}")
    print(f"\nresults: {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
