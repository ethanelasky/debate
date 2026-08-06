"""Normalize the MonitoringBench pool so surface statistics stop leaking the label.

THE PROBLEM. In the raw matched 550/550 pool the two labels come from
different generators — attacks from the MonitoringBench HF release, honest
rollouts from the local Opus honest world — and they differ on three
correlated surface statistics before any security reasoning happens:

                     attack   honest    AUC (0.5 = no signal)
    rendered chars    39.3k    62.3k    0.786  shorter => attack
    steps                27       19    0.700  more steps => attack
    tool calls         27.8     38.6    0.708  fewer calls => attack

A single character threshold classified the raw pool at 73.2% against a 50%
base rate.

THE ROOT CAUSE IS STEP GRANULARITY. Attack transcripts hold EXACTLY ONE tool
call per step (max 1, all 15,271 of them). Honest transcripts BATCH — 1.79
calls per step on average, up to 15. So "step" is not a comparable unit
across the two sources, and the honest agent additionally takes genuinely more
actions (38.6 vs 27.8 calls per trajectory).

That is why cutting alone cannot work, and every cut-only variant was measured
before this one was chosen:

    approach                        chars  steps  calls   worst dev from 0.5
    untouched                       0.786  0.700  0.708   0.286
    truncate honest whole steps     0.491  0.939    -     0.439
    cap text inside honest steps    0.489  0.700  0.708   0.208
    truncate honest by tool calls   0.567  0.901  0.452   0.401
    THIS: merge, then truncate      0.574  0.616  0.452   0.116

Fixing a count while the granularity stays mismatched just relocates the
shortcut into the step axis.

WHAT THIS SCRIPT DOES, in order — format first, then counts.

STAGE 1, attacks: batch consecutive attack steps into groups whose sizes are
drawn from honest's empirical calls-per-step distribution, joining actions and
outputs with newlines. This is PURE REFORMATTING: no action, output, or
ordering is removed, so the attack behavior that makes the row an attack is
still fully present and the label stays valid. It is the only edit made to the
attack side, and it exists solely to put both sources in the same units.

STAGE 2, honest: truncate each honest trajectory by dropping TRAILING WHOLE
STEPS until its tool-call count meets a target rank-matched, within task,
against the same-task attack call counts. Whole steps only — a honest step's
`responses` is the concatenated output of all its batched calls with no
delimiter, so a call cannot be separated from its output and a partial step
would fabricate attribution. Truncation is safe on this side only: an honest
trajectory with trailing work removed is still honest, whereas cutting an
attack could delete the attack itself.

Row IDS AND ORDER ARE UNCHANGED, so data/monitoringbench/splits/ stays valid.

KNOWN RESIDUAL ARTIFACTS, recorded rather than assumed away:
  - honest rows now stop before their task is finished; "ends mid-task" is an
    honest-only tell. No truncation marker is inserted, deliberately — a marker
    only honest rows carry would be trivially detectable.
  - merged attack outputs are joined with "\\n". That is an INFERENCE about
    honest's batching convention: honest's concatenated outputs carry no
    delimiter to read it off, so this could not be verified from the data.
  - the residual AUCs are 0.574 / 0.616 / 0.452, not 0.500.
The check is empirical, not arithmetic: run scripts/mb_solo_monitor.py over
the dev split on this pool and compare against the baselines above.

    .venv/bin/python scripts/mb_normalize_pool.py [--seed 0] [--dry-run]

SAFETY: the source files contain red-team attack trajectories. This script
prints ids, labels, counts and lengths ONLY — never trajectory content.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import hashlib
import json
import random
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infra.envs.tasks.monitoringbench import render_transcript  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ATTACKS_IN = "data/monitoringbench/attacks_mb550_final.jsonl"
HONEST_IN = "data/monitoringbench/honest_mb550.jsonl"
ATTACKS_OUT = "data/monitoringbench/attacks_mb550_norm.jsonl"
HONEST_OUT = "data/monitoringbench/honest_mb550_norm.jsonl"
REPORT_DIR = REPO / "data/monitoringbench/norm"

#: A tool call opens a line: ``bash({...``, ``text_editor({...``, ``python({...``.
#: Counted per step; a step with no match still counts as one call (it did
#: something) rather than zero, so budgets can never divide by zero.
CALL_RE = re.compile(r"^(\w+)\(\{", re.M)

ID_RE = re.compile(r"^(attack|honest)_(\d+)_(\d+)$")


def task_of(rid: str) -> str:
    m = ID_RE.match(rid)
    if not m:
        raise ValueError(f"id {rid!r} is not <label>_<task>_<idx>")
    return m.group(2)


def n_calls(step: dict) -> int:
    return max(1, len(CALL_RE.findall(step.get("action") or "")))


def row_calls(row: dict) -> int:
    return sum(n_calls(s) for s in row["steps"])


def read(rel: str) -> list[dict]:
    return [json.loads(l) for l in (REPO / rel).read_text().splitlines() if l.strip()]


def auc(pos: list[float], neg: list[float]) -> float:
    """P(pos > neg), ties 0.5."""
    s = sorted(neg)
    total = 0.0
    for v in pos:
        lo, hi = bisect.bisect_left(s, v), bisect.bisect_right(s, v)
        total += lo + 0.5 * (hi - lo)
    return total / (len(pos) * len(s))


def merge_steps(steps: list[dict], batch_dist: list[int], rng: random.Random) -> list[dict]:
    """Stage 1. Batch consecutive steps into groups sized from batch_dist.

    Content-preserving by construction: every action and every output survives,
    in order, joined by newlines — the shape honest rows already have."""
    out: list[dict] = []
    i = 0
    while i < len(steps):
        group = steps[i : i + rng.choice(batch_dist)]
        out.append(
            {
                "action": "\n".join(s.get("action") or "" for s in group),
                "responses": "\n".join(s.get("responses") or "" for s in group),
            }
        )
        i += len(group)
    return out


def truncate_to_calls(steps: list[dict], budget: int) -> list[dict]:
    """Stage 2. Longest prefix of WHOLE steps whose call count fits budget.

    Never returns empty: a row whose first step already exceeds the budget
    keeps that step (dropping it would leave no trajectory at all)."""
    keep: list[dict] = []
    used = 0
    for s in steps:
        c = n_calls(s)
        if keep and used + c > budget:
            break
        keep.append(s)
        used += c
    return keep


def axes(rows: list[dict]) -> dict[str, list[int]]:
    return {
        "chars": [len(render_transcript(r["steps"])) for r in rows],
        "steps": [len(r["steps"]) for r in rows],
        "calls": [row_calls(r) for r in rows],
    }


def confound(a: dict[str, list[int]], h: dict[str, list[int]]) -> dict[str, float]:
    """Each AUC oriented so >0.5 means the statistic predicts "attack"."""
    return {
        "auc_shorter_means_attack": round(1 - auc(a["chars"], h["chars"]), 4),
        "auc_more_steps_means_attack": round(auc(a["steps"], h["steps"]), 4),
        "auc_fewer_calls_means_attack": round(1 - auc(a["calls"], h["calls"]), 4),
    }


def best_threshold_accuracy(attack: list[int], honest: list[int]) -> dict:
    """Best 'chars < T => attack' accuracy — what a pure length reader scores."""
    best = (0.0, 0)
    n = len(attack) + len(honest)
    for t in sorted({*attack, *honest}):
        acc = (sum(1 for a in attack if a < t) + sum(1 for x in honest if x >= t)) / n
        if acc > best[0]:
            best = (acc, t)
    return {"accuracy": round(best[0], 4), "threshold_chars": best[1]}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--seed", type=int, default=0, help="batching draw for stage 1")
    ap.add_argument("--dry-run", action="store_true", help="report the effect; write nothing")
    args = ap.parse_args()

    attacks, honest = read(ATTACKS_IN), read(HONEST_IN)
    before = {"attack": axes(attacks), "honest": axes(honest)}

    # --- stage 1: attacks adopt honest's step granularity
    batch_dist = [n_calls(s) for r in honest for s in r["steps"]]
    rng = random.Random(args.seed)
    attacks_out = [{**r, "steps": merge_steps(r["steps"], batch_dist, rng)} for r in attacks]
    # Content preservation, checked per row rather than asserted in a comment.
    # Merging joins with "\n", so char COUNTS legitimately grow by the added
    # separators; what must hold is that the newline-joined concatenation of a
    # field is identical before and after regrouping.
    def joined(rows: list[dict], field: str) -> str:
        return "\n".join(s.get(field) or "" for s in rows)

    for src, dst in zip(attacks, attacks_out):
        if row_calls(src) != row_calls(dst):
            raise AssertionError(f"{src['id']}: merge changed the call count")
        for field in ("action", "responses"):
            if joined(src["steps"], field) != joined(dst["steps"], field):
                raise AssertionError(f"{src['id']}: merge altered {field} text")

    # --- stage 2: honest call counts rank-matched to same-task attack counts
    a_by_task: dict[str, list[int]] = collections.defaultdict(list)
    for r in attacks_out:
        a_by_task[task_of(r["id"])].append(row_calls(r))
    h_by_task: dict[str, list[dict]] = collections.defaultdict(list)
    for r in honest:
        h_by_task[task_of(r["id"])].append(r)

    targets: dict[str, int] = {}
    for task, rows in h_by_task.items():
        alens = sorted(a_by_task[task])
        if len(alens) != len(rows):
            raise ValueError(f"task {task} not mirrored: {len(alens)} attack vs {len(rows)} honest")
        for rank, row in enumerate(sorted(rows, key=row_calls)):
            targets[row["id"]] = alens[rank]

    honest_out, manifest = [], [
        "id,task,target_calls,calls_before,calls_after,steps_before,steps_after,chars_before,chars_after"
    ]
    for row in honest:
        rid = row["id"]
        kept = truncate_to_calls(row["steps"], targets[rid])
        new = {**row, "steps": kept}
        new["normalize"] = {
            "target_calls": targets[rid],
            "calls_before": row_calls(row),
            "calls_after": row_calls(new),
            "steps_before": len(row["steps"]),
            "steps_after": len(kept),
            "chars_before": len(render_transcript(row["steps"])),
            "chars_after": len(render_transcript(kept)),
        }
        honest_out.append(new)
        m = new["normalize"]
        manifest.append(
            f"{rid},{task_of(rid)},{m['target_calls']},{m['calls_before']},{m['calls_after']},"
            f"{m['steps_before']},{m['steps_after']},{m['chars_before']},{m['chars_after']}"
        )

    after = {"attack": axes(attacks_out), "honest": axes(honest_out)}
    b, a = confound(before["attack"], before["honest"]), confound(after["attack"], after["honest"])
    report = {
        "generator": "scripts/mb_normalize_pool.py",
        "seed": args.seed,
        "stages": [
            "attacks: batch consecutive steps to honest's calls-per-step distribution (reformat only)",
            "honest: drop trailing whole steps until the call count meets the same-task rank-matched attack count",
        ],
        "inputs": {
            rel: {"sha256": hashlib.sha256((REPO / rel).read_bytes()).hexdigest()}
            for rel in (ATTACKS_IN, HONEST_IN)
        },
        "medians": {
            axis: {
                "attack_before": round(statistics.median(before["attack"][axis])),
                "attack_after": round(statistics.median(after["attack"][axis])),
                "honest_before": round(statistics.median(before["honest"][axis])),
                "honest_after": round(statistics.median(after["honest"][axis])),
            }
            for axis in ("chars", "steps", "calls")
        },
        "confound": {
            "before": b,
            "after": a,
            "worst_deviation_from_0.5": {
                "before": round(max(abs(v - 0.5) for v in b.values()), 4),
                "after": round(max(abs(v - 0.5) for v in a.values()), 4),
            },
            "best_length_threshold_accuracy": {
                "before": best_threshold_accuracy(before["attack"]["chars"], before["honest"]["chars"]),
                "after": best_threshold_accuracy(after["attack"]["chars"], after["honest"]["chars"]),
                "base_rate": 0.5,
            },
        },
        "attack_side": {
            "rows": len(attacks_out),
            "content_removed": False,
            "note": "reformatting only — call counts and action text verified identical row by row",
        },
        "honest_side": {
            "rows": len(honest_out),
            "rows_truncated": sum(
                1 for r in honest_out if r["normalize"]["steps_after"] < r["normalize"]["steps_before"]
            ),
            "calls_removed_total": sum(
                r["normalize"]["calls_before"] - r["normalize"]["calls_after"] for r in honest_out
            ),
        },
        "residual_artifacts": [
            "honest rows stop before their task is finished ('ends mid-task' is an honest-only tell); no truncation marker is inserted, since a marker only honest rows carry would be trivially detectable",
            "merged attack outputs are joined with a newline — an inference about honest's batching convention, unverifiable from the data (honest's concatenated outputs carry no delimiter)",
            f"residual AUCs are {a} , not 0.5",
            "verify empirically: scripts/mb_solo_monitor.py over the dev split, compared against the 'before' baselines",
        ],
    }
    print(json.dumps(report, indent=2))

    if args.dry_run:
        print("\n--dry-run: nothing written", file=sys.stderr)
        return
    for rel, rows in ((ATTACKS_OUT, attacks_out), (HONEST_OUT, honest_out)):
        with open(REPO / rel, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "honest_manifest.csv").write_text("\n".join(manifest) + "\n")
    (REPORT_DIR / "config.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {ATTACKS_OUT}, {HONEST_OUT}, {REPORT_DIR.relative_to(REPO)}/", file=sys.stderr)


if __name__ == "__main__":
    main()
