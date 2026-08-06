"""Build the canonical MonitoringBench train / dev / eval splits.

WHY A GROUPED SPLIT. The 550/550 matched pool is MIRRORED PER TASK: for each
of the 116 BashArena tasks it holds k attack and k honest trajectories of the
SAME underlying task (k ranges 1..14). Row-level carving (the old
``dataset.test_size`` prefix carve) therefore leaks: an eval row's task is
almost always represented in train by its siblings, which share the task
prompt, the environment, and long stretches of setup actions. Splits here are
made at the TASK level — every trajectory of a task lands in exactly one
split — so eval measures generalization to unseen tasks.

THREE SPLITS, ALL TASK-DISJOINT:
  eval  ~250 rows  final numbers only. Never trained on, never used to pick a
                   prompt, a checkpoint, or a hyperparameter.
  dev   ~120 rows  everything selection-shaped: judge-prompt ablations, lr
                   sweeps, RLVR periodic eval / checkpoint choice. Held out of
                   training so it is an honest generalization read, and kept
                   off eval so selection pressure never touches the final set.
  train ~730 rows  RLVR + debate training.

Label balance is exact and free: mirroring means any set of whole tasks is
50/50 attack/honest.

STRATIFIED ON TASK SIZE AND ON TRAJECTORY LENGTH. Tasks are banded by their
per-label row count so the three splits see the same task-size mix, and the
search below additionally minimizes the gap in median rendered-trajectory
length across splits, so the splits stay comparable on the length axis.

Length matters here because the RAW pool leaked the label through surface
statistics — chars 0.786 AUC, steps 0.700, tool calls 0.708 — which
``scripts/mb_normalize_pool.py`` reduces to 0.574 / 0.616 / 0.452 by putting
both generators on the same step granularity and matching call counts. These
splits are built over that NORMALIZED pool, and the ``length_confound`` block
in the emitted config.json is measured on it. Splitting neither creates nor
fixes the residual — holding the splits length-comparable just stops it
drifting between them.

ROW ORDER WITHIN A SPLIT is chosen so that any PREFIX is a usable sample:
labels strictly alternate (a prefix of length n is n/2 attack, n/2 honest) and
tasks are visited round-robin (the first rows are one-per-task). That makes
``--limit 64`` / ``eval_n: 64`` an exactly-balanced, maximally-task-diverse
subsample instead of whatever the shuffle happened to put first.

Deterministic: same inputs + same --seed -> byte-identical outputs.

    .venv/bin/python scripts/mb_make_splits.py [--seed 0] [--check]

``--check`` re-derives the splits and diffs them against the committed files
without writing (a CI-shaped guard against silent drift).

SAFETY: the source jsonl files contain red-team attack trajectories. This
script prints and writes ids, labels, counts, and lengths ONLY — never
trajectory content.
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
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infra.envs.tasks.monitoringbench import render_transcript  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
#: The NORMALIZED pool (scripts/mb_normalize_pool.py), not the raw files:
#: splits must be built over the same bytes the arms serve, or the statistics
#: recorded below describe a pool nobody runs.
SOURCE_FILES = [
    "data/monitoringbench/honest_mb550_norm.jsonl",
    "data/monitoringbench/attacks_mb550_norm.jsonl",
]
OUT_DIR = REPO / "data/monitoringbench/splits"

#: Row targets. eval is sized to the historical MB eval pool (256 rounds) so
#: numbers stay comparable; dev is sized for ~100-row selection experiments
#: with a little slack.
TARGET_EVAL_ROWS = 250
TARGET_DEV_ROWS = 120

#: Task-size bands (per-label rows per task). Each split draws its tasks
#: proportionally from every band, so no split is all-singletons or all-giants.
BANDS = ((1, 1), (2, 3), (4, 6), (7, 9), (10, 12), (13, 14))

#: Restarts of the randomized stratified allocation; the best-scoring
#: assignment wins. 4000 is far past the point where the score stops improving
#: and still runs in a couple of seconds.
RESTARTS = 4000

ID_RE = re.compile(r"^(attack|honest)_(\d+)_(\d+)$")


@dataclass(frozen=True)
class Row:
    id: str
    label: str
    task: str
    idx: int
    chars: int
    steps: int


def load_rows() -> list[Row]:
    rows: list[Row] = []
    for rel in SOURCE_FILES:
        with open(REPO / rel, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                if not line.strip():
                    continue
                rec = json.loads(line)
                m = ID_RE.match(rec["id"])
                if not m:
                    raise ValueError(f"{rel}:{lineno}: id {rec['id']!r} is not <label>_<task>_<idx>")
                if m.group(1) != rec["label"] and not (m.group(1) == "honest" and rec["label"] == "honest"):
                    raise ValueError(f"{rel}:{lineno}: id prefix disagrees with label")
                rows.append(
                    Row(
                        id=rec["id"],
                        label=rec["label"],
                        task=m.group(2),
                        idx=int(m.group(3)),
                        chars=len(render_transcript(rec["steps"])),
                        steps=len(rec["steps"]),
                    )
                )
    ids = [r.id for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate ids in the source pool")
    return sorted(rows, key=lambda r: r.id)


def band_of(k: int) -> int:
    for i, (lo, hi) in enumerate(BANDS):
        if lo <= k <= hi:
            return i
    raise ValueError(f"task size {k} outside the declared bands")


def auc(pos: list[float], neg: list[float]) -> float:
    """P(pos > neg), ties counted 0.5."""
    s = sorted(neg)
    total = 0.0
    for v in pos:
        lo, hi = bisect.bisect_left(s, v), bisect.bisect_right(s, v)
        total += lo + 0.5 * (hi - lo)
    return total / (len(pos) * len(s))


def allocate(tasks: dict[str, list[Row]], seed: int) -> dict[str, str]:
    """Assign every task to 'eval' | 'dev' | 'train'.

    Two stages, because the two held-out sets want different things.

    EVAL is drawn band-stratified and proportional, so its task-size mix
    matches the pool: it has to carry a representative absolute number.

    DEV wants the opposite — it is used for RELATIVE comparisons (prompt A vs
    prompt B, checkpoint N vs N+1) at n~100, where rows sharing a task are
    correlated and the effective sample size is closer to the task count than
    the row count. So dev is filled from the SMALLEST remaining tasks, which
    buys the most distinct tasks per row. Small tasks are handed out
    ALTERNATELY to dev and train rather than dev taking them all: the
    singleton tasks are the most task-diverse part of the pool and training
    should not be stripped of them.

    Randomized restarts over the seeded shuffles; the winner minimizes row
    counts' distance to target plus the cross-split median-length gap, so the
    splits stay comparable on the length axis (see module docstring)."""
    by_band: dict[int, list[str]] = collections.defaultdict(list)
    for task, rs in tasks.items():
        by_band[band_of(len(rs) // 2)].append(task)
    for band in by_band.values():
        band.sort()

    n_rows = sum(len(rs) for rs in tasks.values())
    eval_frac = TARGET_EVAL_ROWS / n_rows
    rng = random.Random(seed)
    best: tuple[float, dict[str, str]] | None = None

    for _ in range(RESTARTS):
        assign: dict[str, str] = {}
        # --- stage 1: eval, proportional within every task-size band
        for _band, task_ids in sorted(by_band.items()):
            shuffled = list(task_ids)
            rng.shuffle(shuffled)
            want = eval_frac * sum(len(tasks[t]) for t in shuffled)
            got = 0
            for t in shuffled:
                if got >= want or (got and got + len(tasks[t]) - want > want - got):
                    break  # stop before overshooting the band quota by half a task
                assign[t] = "eval"
                got += len(tasks[t])

        # --- stage 2: dev, smallest tasks first, alternating with train
        remaining = sorted(
            (t for t in tasks if t not in assign),
            key=lambda t: (len(tasks[t]), rng.random()),
        )
        dev_rows = 0
        for i, t in enumerate(remaining):
            if dev_rows < TARGET_DEV_ROWS and i % 2 == 0:
                assign[t] = "dev"
                dev_rows += len(tasks[t])
            else:
                assign[t] = "train"

        rows_of: dict[str, list[Row]] = collections.defaultdict(list)
        for t, split in assign.items():
            rows_of[split].extend(tasks[t])
        if len(rows_of) != 3 or any(not v for v in rows_of.values()):
            continue
        meds = {s: statistics.median(r.chars for r in rs) for s, rs in rows_of.items()}
        score = (
            abs(len(rows_of["eval"]) - TARGET_EVAL_ROWS) / 10
            + abs(len(rows_of["dev"]) - TARGET_DEV_ROWS) / 10
            # median rendered length within ~2k chars across splits
            + (max(meds.values()) - min(meds.values())) / 2000
        )
        if best is None or score < best[0]:
            best = (score, dict(assign))

    if best is None:
        raise RuntimeError("allocation found no valid assignment")
    return best[1]


def order_rows(rows: list[Row], seed: int) -> list[Row]:
    """Label-alternating, task-round-robin order: every prefix is 50/50 and
    hits as many distinct tasks as possible."""
    rng = random.Random(seed)
    tasks = sorted({r.task for r in rows})
    rng.shuffle(tasks)
    rank = {t: i for i, t in enumerate(tasks)}

    def lane(label: str) -> list[Row]:
        by_task: dict[str, list[Row]] = collections.defaultdict(list)
        for r in rows:
            if r.label == label:
                by_task[r.task].append(r)
        for rs in by_task.values():
            rs.sort(key=lambda r: r.idx)
        # (nth row taken from this task, task rank) -> one row per task, then seconds, ...
        return [r for _, _, r in sorted(
            ((n, rank[t], r) for t, rs in by_task.items() for n, r in enumerate(rs)),
            key=lambda x: (x[0], x[1]),
        )]

    attack, honest = lane("attack"), lane("honest")
    out: list[Row] = []
    for a, h in zip(attack, honest):
        out.extend((a, h))
    out.extend(attack[len(honest):] or honest[len(attack):])
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(seed: int):
    rows = load_rows()
    tasks: dict[str, list[Row]] = collections.defaultdict(list)
    for r in rows:
        tasks[r.task].append(r)
    for task, rs in tasks.items():
        n_a = sum(1 for r in rs if r.label == "attack")
        if n_a * 2 != len(rs):
            raise ValueError(f"task {task} is not mirrored: {n_a} attack of {len(rs)} rows")

    assign = allocate(tasks, seed)
    ordered = {
        split: order_rows([r for r in rows if assign[r.task] == split], seed)
        for split in ("train", "dev", "eval")
    }
    return rows, tasks, assign, ordered


def stats(rs: list[Row]) -> dict:
    a = [r.chars for r in rs if r.label == "attack"]
    h = [r.chars for r in rs if r.label == "honest"]
    return {
        "rows": len(rs),
        "tasks": len({r.task for r in rs}),
        "attack": len(a),
        "honest": len(h),
        "median_chars": round(statistics.median(r.chars for r in rs)),
        "median_chars_attack": round(statistics.median(a)) if a else None,
        "median_chars_honest": round(statistics.median(h)) if h else None,
        "median_steps": round(statistics.median(r.steps for r in rs)),
        "length_auc_attack_is_shorter": round(1 - auc(a, h), 4) if a and h else None,
    }


def render(ordered, rows, tasks, assign, seed) -> dict[str, str]:
    """Split -> file text. Returned rather than written so --check can diff."""
    files: dict[str, str] = {}
    for split, rs in ordered.items():
        header = (
            f"# MonitoringBench {split} split — {len(rs)} rows over "
            f"{len({r.task for r in rs})} tasks, 50/50 attack/honest.\n"
            f"# Generated by scripts/mb_make_splits.py --seed {seed}; do not hand-edit.\n"
            "# Order is label-alternating and task-round-robin: any prefix is a\n"
            "# balanced, task-diverse subsample.\n"
        )
        files[f"{split}_ids.txt"] = header + "".join(f"{r.id}\n" for r in rs)

    manifest = ["id,label,task,split,n_steps,chars,order_in_split"]
    for split, rs in ordered.items():
        for i, r in enumerate(rs):
            manifest.append(f"{r.id},{r.label},{r.task},{split},{r.steps},{r.chars},{i}")
    files["manifest.csv"] = "\n".join(manifest) + "\n"

    all_a = [r.chars for r in rows if r.label == "attack"]
    all_h = [r.chars for r in rows if r.label == "honest"]
    pool_auc = 1 - auc(all_a, all_h)
    within = wins = ties = 0
    for rs in tasks.values():
        for a in (r.chars for r in rs if r.label == "attack"):
            for h in (r.chars for r in rs if r.label == "honest"):
                within += 1
                if a < h:
                    wins += 1
                elif a == h:
                    ties += 1
    config = {
        "generator": "scripts/mb_make_splits.py",
        "seed": seed,
        "source_files": [
            {"path": p, "sha256": sha256(REPO / p), "bytes": (REPO / p).stat().st_size}
            for p in SOURCE_FILES
        ],
        "grouping": "task id (middle field of <label>_<task>_<idx>); splits are task-disjoint",
        "targets": {"eval_rows": TARGET_EVAL_ROWS, "dev_rows": TARGET_DEV_ROWS},
        "task_size_bands": [list(b) for b in BANDS],
        "splits": {split: stats(rs) for split, rs in ordered.items()},
        "pool": stats(rows),
        # Residual after scripts/mb_normalize_pool.py, recorded because it
        # bounds how these numbers should be read: a trivial "shorter
        # trajectory => attack" rule still beats chance. Splitting holds it
        # constant across splits; it does not remove it.
        "length_confound": {
            "auc_shorter_means_attack_pool": round(pool_auc, 4),
            "auc_shorter_means_attack_within_task": round((wins + 0.5 * ties) / within, 4),
            "tasks_where_attack_median_is_shorter": sum(
                1
                for rs in tasks.values()
                if statistics.median(r.chars for r in rs if r.label == "attack")
                < statistics.median(r.chars for r in rs if r.label == "honest")
            ),
            "n_tasks": len(tasks),
            "note": (
                "residual after scripts/mb_normalize_pool.py (raw pool was 0.786 / "
                "0.854). Attack trajectories come from the MonitoringBench HF "
                "release, honest ones from the local Opus honest world. Any accuracy "
                "on this pool should be read against this baseline, and against the "
                "step-count residual recorded in data/monitoringbench/norm/config.json."
            ),
        },
    }
    files["config.json"] = json.dumps(config, indent=2) + "\n"
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true", help="diff against the committed files; write nothing")
    args = ap.parse_args()

    rows, tasks, assign, ordered = build(args.seed)
    files = render(ordered, rows, tasks, assign, args.seed)

    if args.check:
        bad = []
        for name, text in files.items():
            path = OUT_DIR / name
            if not path.exists():
                bad.append(f"{name}: missing")
            elif path.read_text() != text:
                bad.append(f"{name}: differs from regenerated content")
        if bad:
            print("SPLIT DRIFT:\n  " + "\n  ".join(bad), file=sys.stderr)
            raise SystemExit(1)
        print(f"splits up to date ({', '.join(sorted(files))})")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (OUT_DIR / name).write_text(text)
    print(json.dumps(json.loads(files["config.json"])["splits"], indent=2))
    print(f"\nwrote {len(files)} files to {OUT_DIR.relative_to(REPO)}", file=sys.stderr)


if __name__ == "__main__":
    main()
