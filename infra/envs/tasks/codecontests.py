"""CodeContests task family: competitive programming problems verified by
running the candidate program against stdin/stdout test cases.

Ported from the old repo (ai_debate/data_loader/codecontests_loader.py and
ai_debate/experiments/verifiers/codecontests_verifier.py), slimmed: rows are
held in memory (no lazy-JSONL dict) and there is no container sandbox — the
runner is a plain subprocess.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Optional

from infra.envs.base import Env, Policy, Task, Trajectory, datum_from_sample
from infra.envs.tasks.base import TaskFamily, reject_unknown_keys

# Each verifier subprocess may use up to 4GB (RLIMIT_AS in the runner script),
# so cap concurrency to keep total memory bounded.
_MAX_CONCURRENT_VERIFIERS = int(os.environ.get("MAX_CONCURRENT_VERIFIERS", "8"))
_verifier_semaphore = threading.Semaphore(_MAX_CONCURRENT_VERIFIERS)

# The last two sentences are verbatim from the old repo's shared.yaml
# "Direct Code Generation" prompt (its only battle-tested code wording).
SYSTEM_PROMPT = (
    "Solve the competitive programming problem.\n"
    "The program reads from stdin and writes to stdout.\n"
    "Return the complete program as exactly one ```python code block.\n"
    "Your goal is to write code that is consistent with the specification in "
    "order to pass the hidden test cases. Do not include functionality or "
    "error handling beyond what is described."
)

# Descriptions containing these admit multiple valid outputs, which exact
# output comparison would mis-grade. Ported verbatim from the old loader.
_MULTI_ANSWER_PHRASES = [
    "output any",
    "print any",
    "any valid",
    "any such",
    "any of them",
    "any one of",
    "any suitable",
    "any optimal",
    "any correct",
    "if there are multiple",
    "if multiple",
    "any possible",
    "you can output any",
    "you may output any",
    "you can print any",
    "you may print any",
    "any order",
    "in some order",
    "arbitrary order",
    "you may output",
]

# C++ detection: models asked for Python sometimes emit C++, which the Python
# runner would report as a syntax error. Ported verbatim; grading-relevant.
CPP_PATTERNS = [
    r"\busing\s+namespace\s+std\b",
    r"#include\s*<",
    r"\bint\s+main\s*\(",
    r"\bcout\s*<<",
    r"\bcin\s*>>",
    r"\bstd::\w+",
    r"\bvector\s*<",
    r"\bprintf\s*\(",
    r"\bscanf\s*\(",
    r"\bgetline\s*\(",
    r"\b(endl|nullptr|NULL)\b",
    r"\bchar\s+\w+\s*\[\s*\d+\s*\]",  # char arrays
    r"\bclass\s+\w+\s*{[^}]*public:",  # C++ class with public
    r"\btemplate\s*<",
    r"\bauto\s+\w+\s*=",  # C++ auto type inference
]

PYTHON_CODE_BLOCK_PATTERN = re.compile(r"```python\s*(.*?)```", re.IGNORECASE | re.DOTALL)
GENERIC_BLOCK_PATTERN = re.compile(r"```\s*(.*?)```", re.DOTALL)
_OPEN_FENCE_PATTERN = re.compile(r"```[ \t]*(?:python)?[ \t]*\r?\n", re.IGNORECASE)


def is_cpp_code(code: str, threshold: int = 2) -> bool:
    if not code:
        return False
    return sum(1 for p in CPP_PATTERNS if re.search(p, code, re.IGNORECASE)) >= threshold


def row_eligible(entry: dict[str, Any]) -> bool:
    """Verifiable (non-empty test I/O) and not a multi-answer problem."""
    if not entry.get("inputs") or not entry.get("outputs"):
        return False
    desc_lower = (entry.get("description", "") or "").lower()
    return not any(phrase in desc_lower for phrase in _MULTI_ANSWER_PHRASES)


def extract_code(text: str, relaxed: bool = True) -> Optional[str]:
    """Last ```python fence, else last generic fence. When `relaxed`, a
    trailing UNCLOSED fence is accepted too (generations truncated at the
    token cap routinely end mid-block)."""
    if not text:
        return None
    for pattern in (PYTHON_CODE_BLOCK_PATTERN, GENERIC_BLOCK_PATTERN):
        matches = list(pattern.finditer(text))
        if matches:
            return matches[-1].group(1).strip() or None
    if not relaxed:
        return None
    opens = list(_OPEN_FENCE_PATTERN.finditer(text))
    if not opens:
        return None
    return text[opens[-1].end():].strip() or None


def has_closed_fence(text: str) -> bool:
    return bool(text) and bool(
        PYTHON_CODE_BLOCK_PATTERN.search(text) or GENERIC_BLOCK_PATTERN.search(text)
    )


# ------------------------------------------------------------------ verifier


def run_stdin_tests(
    solution_code: str,
    inputs: list[str],
    outputs: list[str],
    timeout: int = 90,
) -> dict[str, Any]:
    """Run every stdin/stdout case in one subprocess (the runner execs the
    solution once per case in-process, so we pay one interpreter start per
    solution rather than one per test)."""
    t0 = time.perf_counter()
    tmpdir = tempfile.mkdtemp(prefix="codecontests_test_")

    _verifier_semaphore.acquire()
    try:
        solution_path = os.path.join(tmpdir, "solution.py")
        with open(solution_path, "w") as f:
            f.write(solution_code)

        test_cases_path = os.path.join(tmpdir, "test_cases.json")
        with open(test_cases_path, "w") as f:
            json.dump({"inputs": inputs, "outputs": outputs}, f)

        runner_path = os.path.join(tmpdir, "runner.py")
        with open(runner_path, "w") as f:
            f.write(_TEST_RUNNER_SCRIPT)

        try:
            proc = subprocess.run(
                [sys.executable, runner_path, solution_path, test_cases_path],
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return _failure_result("timeout", len(inputs), "Total execution timed out.", t0, timeout=True)
        except Exception as exc:  # noqa: BLE001
            return _failure_result("error", len(inputs), repr(exc), t0)

        elapsed = time.perf_counter() - t0
        try:  # runner prints its result JSON on the last line of stdout
            result = json.loads(proc.stdout.strip().split("\n")[-1])
            result["execution_time_seconds"] = elapsed
            return result
        except (json.JSONDecodeError, IndexError):
            return _failure_result(
                "error", len(inputs), (proc.stderr or proc.stdout or "")[:1000], t0
            )
    except Exception as exc:  # noqa: BLE001
        return _failure_result("error", len(inputs), repr(exc), t0)
    finally:
        _verifier_semaphore.release()
        shutil.rmtree(tmpdir, ignore_errors=True)


def _failure_result(
    status: str, tests_total: int, stderr: str, t0: float, timeout: bool = False
) -> dict[str, Any]:
    return {
        "status": status,
        "passed": False,
        "tests_passed": 0,
        "tests_total": tests_total,
        "timeout": timeout,
        "first_failure": {"test_idx": 0, "expected": "", "actual": "", "stderr": stderr},
        "execution_time_seconds": time.perf_counter() - t0,
    }


# Runner script: execs the solution against each case with stdin/stdout
# swapped to StringIO, then prints one JSON line.
_TEST_RUNNER_SCRIPT = r'''
import json
import sys
import io
import re
import resource
import traceback

# Cap memory at 4GB to prevent OOM when many solutions run concurrently
try:
    mem_limit = 4 * 1024 * 1024 * 1024  # 4 GB
    resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))
except (ValueError, resource.error):
    pass  # Not available on all platforms

def normalize_output(s):
    lines = s.split("\n")
    normalized = []
    for line in lines:
        line = line.rstrip()
        def normalize_float(match):
            try:
                return f"{float(match.group(0)):g}"
            except (ValueError, OverflowError):
                return match.group(0)
        line = re.sub(r"-?\d+\.\d+(?:[eE][+-]?\d+)?", normalize_float, line)
        normalized.append(line)
    while normalized and not normalized[-1]:
        normalized.pop()
    return "\n".join(normalized).lower()

def main():
    solution_path = sys.argv[1]
    test_cases_path = sys.argv[2]

    with open(solution_path) as f:
        solution_code = f.read()

    # Compile once, run many times
    try:
        compiled = compile(solution_code, solution_path, "exec")
    except SyntaxError as e:
        print(json.dumps({
            "status": "error", "passed": False,
            "tests_passed": 0, "tests_total": 0, "timeout": False,
            "first_failure": {"test_idx": 0, "expected": "", "actual": "",
                              "stderr": f"SyntaxError: {e}"},
        }))
        return

    with open(test_cases_path) as f:
        cases = json.load(f)

    inputs = cases["inputs"]
    outputs = cases["outputs"]
    tests_passed = 0
    tests_total = len(inputs)
    first_failure = None

    for i, (test_input, expected_output) in enumerate(zip(inputs, outputs)):
        old_stdin = sys.stdin
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        captured_stdout = io.StringIO()
        captured_stderr = io.StringIO()

        try:
            sys.stdin = io.StringIO(test_input)
            sys.stdout = captured_stdout
            sys.stderr = captured_stderr
            # Each exec gets a fresh namespace so globals don't leak.
            # __name__ must be "__main__" so if __name__ == "__main__" guards work.
            exec(compiled, {"__builtins__": __builtins__, "__name__": "__main__"})
        except SystemExit:
            pass  # Some solutions call exit()
        except Exception:
            captured_stderr.write(traceback.format_exc())
        finally:
            sys.stdin = old_stdin
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        actual = captured_stdout.getvalue()
        stderr = captured_stderr.getvalue()

        if normalize_output(actual) == normalize_output(expected_output):
            tests_passed += 1
        else:
            if first_failure is None:
                first_failure = {
                    "test_idx": i,
                    "expected": normalize_output(expected_output)[:500],
                    "actual": normalize_output(actual)[:500],
                    "stderr": stderr[:500],
                }

    passed = tests_passed == tests_total
    print(json.dumps({
        "status": "passed" if passed else "failed",
        "passed": passed,
        "tests_passed": tests_passed,
        "tests_total": tests_total,
        "timeout": False,
        "first_failure": first_failure,
    }))

if __name__ == "__main__":
    main()
'''


# ----------------------------------------------------------------------- env


def _load_rows(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if not row_eligible(entry):
                continue
            rows.append(
                {
                    "problem": str(entry.get("description", "")).strip(),
                    "name": str(entry.get("name", "")).strip(),
                    "inputs": list(entry["inputs"]),
                    "outputs": list(entry["outputs"]),
                }
            )
    return rows


class CodeContestsEnv(Env):
    def __init__(
        self,
        path: str,
        test_path: Optional[str] = None,
        seed: int = 0,
        eval_subset_size: int = 128,
        timeout_seconds: int = 90,
        correct_reward: float = 1.0,
        format_reward: float = 0.1,
    ):
        self.rng = random.Random(seed)
        self.timeout_seconds = timeout_seconds
        self.correct_reward = correct_reward
        self.format_reward = format_reward

        self.train_rows = _load_rows(path)
        if test_path is not None:
            self.test_rows = _load_rows(test_path)
        else:
            rng = random.Random(seed + 22222)
            rows = list(self.train_rows)
            rng.shuffle(rows)
            n_test = min(max(64, len(rows) // 10), max(0, len(rows) - 2))
            self.test_rows, self.train_rows = rows[:n_test], rows[n_test:]
        self.test_rows = self.test_rows[:eval_subset_size]
        if len(self.train_rows) < 2 or not self.test_rows:
            raise RuntimeError(
                f"codecontests env: too few rows after filtering "
                f"(train={len(self.train_rows)}, test={len(self.test_rows)})"
            )

    def _task(self, row: dict[str, Any], split: str) -> Task:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": row["problem"] + "\n\nAnswer with exactly one ```python code block.",
            },
        ]
        # meta["question"] is what DebateEnv binds as the debate TOPIC.
        # inputs/outputs are verifier test cases only — they are NOT public
        # examples and must never be rendered into any prompt or transcript.
        return Task(
            messages=messages,
            meta={
                "question": row["problem"],
                "name": row["name"],
                "inputs": row["inputs"],
                "outputs": row["outputs"],
                "split": split,
            },
        )

    def tasks(self, n: int, split: str = "train") -> list[Task]:
        if split == "train":
            return [self._task(self.rng.choice(self.train_rows), split) for _ in range(n)]
        return [self._task(row, split) for row in self.test_rows[:n]]

    def reward(self, task: Task, text: str) -> tuple[float, dict[str, Any]]:
        code = extract_code(text, relaxed=True)
        info: dict[str, Any] = {
            "correct": 0.0,
            "has_code": float(code is not None),
            "tests_passed_frac": 0.0,
        }
        if code is None:
            return 0.0, info
        if is_cpp_code(code):
            info["cpp_code"] = 1.0
            return self.format_reward, info

        result = run_stdin_tests(
            code, task.meta["inputs"], task.meta["outputs"], timeout=self.timeout_seconds
        )
        total = result.get("tests_total") or len(task.meta["inputs"])
        info["correct"] = float(bool(result.get("passed")))
        info["tests_passed_frac"] = result.get("tests_passed", 0) / total if total else 0.0
        return self.format_reward + self.correct_reward * info["correct"], info

    def rollout(self, tasks: list[Task], policy: Policy, group_size: int) -> list[list[Trajectory]]:
        results = policy.predict([t.messages for t in tasks], n=group_size)
        n_dropped = 0
        pending: list[tuple[int, Any, Any]] = []  # (group idx, sample, future)
        n_groups = len(tasks)
        # Grading shells out to a subprocess per sample, so threads overlap.
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, _MAX_CONCURRENT_VERIFIERS)
        ) as pool:
            for gi, (task, samples) in enumerate(zip(tasks, results)):
                for s in samples:
                    if not s.fidelity_ok():
                        n_dropped += 1
                        continue
                    pending.append((gi, s, pool.submit(self.reward, task, s.text)))

            groups: list[list[Trajectory]] = [[] for _ in range(n_groups)]
            for gi, s, fut in pending:
                reward, info = fut.result()
                groups[gi].append(Trajectory(datums=[datum_from_sample(s)], reward=reward, info=info))
        if n_dropped and groups and groups[0]:
            groups[0][0].info["samples_dropped_fidelity"] = float(n_dropped)
        return groups


# -------------------------------------------------------------------- family


class CodeContestsFamily(TaskFamily):
    def __init__(self, timeout_seconds: int = 90):
        # grade() is called by DebateEnv with only meta, so the per-problem
        # timeout has to live on the family instance.
        self.timeout_seconds = timeout_seconds

    def source(self, ds: dict) -> Env:
        reject_unknown_keys(
            ds,
            {
                "path",
                "test_path",
                "seed",
                "eval_subset_size",
                "timeout_seconds",
                "correct_reward",
                "format_reward",
            },
            "codecontests",
        )
        path = ds.get("path")
        if not path:
            raise ValueError(
                "codecontests requires dataset.path: the preprocessed CodeContests JSONL "
                "(rows with name/description/inputs/outputs). Known source: the old repo's "
                "ai_debate/data/codecontests/{train,test}.jsonl"
            )
        self.timeout_seconds = int(ds.get("timeout_seconds", self.timeout_seconds))
        return CodeContestsEnv(
            path=str(path),
            test_path=(str(ds["test_path"]) if ds.get("test_path") else None),
            seed=int(ds.get("seed", 0)),
            eval_subset_size=int(ds.get("eval_subset_size", 128)),
            timeout_seconds=self.timeout_seconds,
            correct_reward=float(ds.get("correct_reward", 1.0)),
            format_reward=float(ds.get("format_reward", 0.1)),
        )

    def extractor(self, relaxed: bool) -> Callable[[str], Any]:
        return lambda text: extract_code(text, relaxed=relaxed)

    def grade(self, meta: dict[str, Any], solution: Any) -> Optional[bool]:
        if solution is None:
            return None
        inputs, outputs = meta.get("inputs"), meta.get("outputs")
        if not inputs or not outputs:
            return None
        if is_cpp_code(solution):
            return False
        return bool(run_stdin_tests(solution, inputs, outputs, timeout=self.timeout_seconds)["passed"])

    def format_flags(self, text: str) -> dict[str, float]:
        # strict flag: a properly CLOSED fence, independent of the (possibly
        # relaxed) extractor used for position binding
        return {"code_fence": float(has_closed_fence(text))}
