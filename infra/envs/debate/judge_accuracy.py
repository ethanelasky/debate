"""Judge accuracy, derived from exported debate transcripts.

AGENTS.md ground rule: judge accuracy is derived from transcripts
(winner x side x label), never from aggregated result files. This module is
that derivation. Per debate it joins the judge's winner against the
ground-truth label of the solution(s) that were on trial — the labels
DebateEnv._attach_labels parked on the state and docent_export carried out.

Deliberately free of docent/torch/dataset imports: this has to run on a laptop
against a directory of JSONL pulled off a pod.

    python -m infra.envs.debate.judge_accuracy docent/step-*.jsonl
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

TIE = "Tie"

DECIDABLE = "decidable"


@dataclass
class JudgeAccuracy:
    """Counts, not just a rate: a rate with a moving denominator is how a
    judge-accuracy number silently drifts for reasons unrelated to the judge."""

    total: int = 0
    decidable: int = 0
    correct: int = 0
    skipped: dict[str, int] = field(default_factory=dict)

    @property
    def accuracy(self) -> Optional[float]:
        return self.correct / self.decidable if self.decidable else None

    def __str__(self) -> str:
        acc = "n/a" if self.accuracy is None else f"{self.accuracy:.4f}"
        skips = ", ".join(f"{k}={v}" for k, v in sorted(self.skipped.items())) or "none"
        return (
            f"judge_accuracy={acc}  decidable={self.decidable}/{self.total}  "
            f"correct={self.correct}  skipped: {skips}"
        )


def judge_was_right(meta: dict[str, Any]) -> tuple[Optional[bool], str]:
    """One exported debate -> (judge picked the side that was actually right,
    reason). None means undecidable; the reason names which way.

    Undecidable is not failure — a tie, or a debate where both sides were wrong,
    has no correct call for the judge to have made, and counting it either way
    would bias the rate.
    """
    if meta.get("failed"):
        return None, "failed"
    verdict = meta.get("verdict")
    if not verdict or not verdict.get("ok"):
        return None, "no_verdict"
    winner = verdict.get("winner")
    if winner is None:
        return None, "no_winner"
    if winner == TIE:
        return None, "tie"

    # seats is keyed by DISPLAY name and holds exactly the debater seats. Do not
    # infer the seat set from `bindings`: the judge's own bindings entry carries
    # Debater_A's name (DebateEnv._build_state), so bindings has a duplicate.
    seats = set(verdict.get("seats") or ())
    if len(seats) != 2 or winner not in seats:
        return None, "unmapped"

    labels = meta.get("grades") or {}
    if not labels:
        return None, "no_labels"

    bindings = meta.get("bindings") or {}
    graded: dict[str, bool] = {}
    for speaker, correct in labels.items():
        if correct is None:  # graded but ungradeable
            continue
        display = bindings.get(speaker)
        if display not in seats:
            return None, "unmapped"
        graded[display] = bool(correct)
    if not graded:
        return None, "label_missing"

    if len(graded) == 1:
        # Proposer-critic: one proposition on trial. Right -> its author should
        # win; wrong -> the challenger should.
        display, correct = next(iter(graded.items()))
        other = next(s for s in seats if s != display)
        should_win = display if correct else other
    else:
        right = [d for d, c in graded.items() if c]
        if len(right) != 1:
            return None, "ambiguous"  # both right or both wrong: no correct call
        should_win = right[0]

    return winner == should_win, DECIDABLE


def tally(metas: Iterable[dict[str, Any]]) -> JudgeAccuracy:
    out = JudgeAccuracy()
    for meta in metas:
        out.total += 1
        ok, reason = judge_was_right(meta)
        if ok is None:
            out.skipped[reason] = out.skipped.get(reason, 0) + 1
            continue
        out.decidable += 1
        out.correct += int(ok)
    return out


def metas_from_jsonl(path: str) -> list[dict[str, Any]]:
    """One exported AgentRun per line; the debate's metadata is under
    "metadata" (docent_export.export_jsonl writes run.model_dump_json())."""
    metas = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                metas.append(json.loads(line).get("metadata") or {})
    return metas


def from_jsonl(paths: str | Iterable[str]) -> JudgeAccuracy:
    if isinstance(paths, str):
        paths = [paths]
    metas: list[dict[str, Any]] = []
    for p in paths:
        metas.extend(metas_from_jsonl(p))
    return tally(metas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="exported debate JSONL (docent/step-*.jsonl)")
    parser.add_argument("--per-file", action="store_true", help="also break down by file")
    args = parser.parse_args()

    if args.per_file:
        for p in args.paths:
            print(f"{p}: {tally(metas_from_jsonl(p))}")
    print(f"ALL: {from_jsonl(args.paths)}")


if __name__ == "__main__":
    main()
