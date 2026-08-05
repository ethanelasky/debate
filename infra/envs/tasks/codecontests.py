"""CodeContests task family: competitive programming problems verified by
running the candidate program against stdin/stdout test cases.

Ported from the old repo (ai_debate/data_loader/codecontests_loader.py and
ai_debate/experiments/verifiers/codecontests_verifier.py), slimmed: rows are
held in memory (no lazy-JSONL dict) and there is no container sandbox — the
runner is a plain subprocess.

===========================================================================
HOW THE TWO ARMS USE THIS FILE
===========================================================================

The experiment compares two ways of rewarding the same proposer model.

RLVR arm (the baseline).  CodeContestsEnv.reward() below IS the reward: the
model writes a program, we execute it against that problem's `rlvr_tests`,
and reward = format_reward + correct_reward * (all tests passed). Execution
decides the gradient.

Debate arm.  DebateEnv never calls reward(). Its reward is judge-only: two
debaters argue about the program and a judge picks a winner. NO test is run
to produce that reward — the judge cannot execute code and never sees a test
case. Execution appears only as a MEASUREMENT, via grade() below, which
labels whether the proposer was actually right. That label lands in the
transcript, never in the reward.

That asymmetry is the whole experiment. If the verifier leaked into debate
reward we would no longer be measuring whether debate produces a
correct-detecting judge.

===========================================================================
TWO TEST SUITES PER PROBLEM, AND WHY
===========================================================================

Each row carries two disjoint suites, built by scripts/build_codecontests_rlvr.py:

  rlvr_tests   <=10 cases sampled (seeded) from DeepMind's public_tests +
               private_tests. The RLVR arm's REWARD. Deliberately small:
               median 4 cases, and 27% of problems have exactly 1, because
               that is all DeepMind publishes for them.

  truth_tests  the cases the sample did NOT take. Ground truth for measuring
               proposer accuracy. Disjoint from rlvr_tests by construction,
               so we never score accuracy on a case the RLVR arm optimized
               against. EMPTY for ~66% of problems (the sample consumed them
               all); grade() returns None there rather than falling back to
               rlvr_tests, which would be measuring on the training signal.

Neither suite is ever rendered into a prompt. Models see the problem
statement only. _task() puts test cases in Task.meta, and DebateEnv binds
only meta["question"] into prompts; docent_export filters meta to scalars, so
the list-valued suites cannot leak into a transcript either.

One thing that looks like a leak and is not: some rlvr_tests cases DO appear
verbatim in the problem statement (~16% of non-trivial cases across the test
split). Those are the PUBLIC tests — DeepMind's public_tests are exactly the
worked examples printed in a Codeforces problem, so a contestant sees them
too. The model reading them off the statement is the intended setting, and it
is part of why this reward is weak. Private tests never appear there.

===========================================================================
OFFLINE ACCURACY (the primary ground truth)
===========================================================================

truth_tests is a cheap in-run signal that exists for only a third of
problems. The real accuracy pass happens AFTER a run, locally, with no GPU or
API cost:

  1. Training writes every rollout to docent/step-NNNNN.jsonl
     (run_debate.py wires on_rollout -> docent_export). Each record contains
     the full text of every speech, so the proposer's program is in there
     verbatim, plus meta["name"] as a stable problem key.
  2. Offline, re-extract the program from each proposal, run it against
     CodeContests-O's corner cases, and recompute proposer and judge accuracy.

This is worth the extra step because CCO's tests are much stronger, and
running them during training would be slow (see the dataset note below).

===========================================================================
HOW THE VERIFIER WORKS
===========================================================================

run_stdin_tests() writes the program, the cases, and a runner script to a
tempdir and spawns `python runner.py` in its OWN PROCESS GROUP so a timeout
kills grandchildren too. The runner compiles the program once and exec()s it
per case with stdin/stdout swapped to StringIO — one interpreter start per
solution, not per test.

The verdict is read from a FILE the runner writes, never from stdout: the
untrusted program shares the runner's stdout and could otherwise print a
forged {"passed": true} from an atexit hook after the StringIO swap unwinds.
The runner also blanks sys.argv so the program cannot find and forge that
file. Limits: RLIMIT_AS 4GB, RLIMIT_NPROC 256, RLIMIT_FSIZE 64MB (the last
two are best-effort; macOS skips them).

Comparison is whitespace-normalized with floats collapsed via %g, so
"3.0000000000" and "3" match. Problems whose statements admit multiple valid
outputs are dropped at BUILD time (_MULTI_ANSWER_PHRASES) because exact
comparison would mis-grade them.

===========================================================================
TEST-CASE SELECTION AND TRUNCATION (all at build time, not here)
===========================================================================

scripts/build_codecontests_rlvr.py applies, per problem:
  - drop if the statement matches a multi-answer phrase
  - drop if it is not stdin/stdout (file-based I/O cannot be piped)
  - drop any problem with a single case above 500 KB
  - sample <=10 cases into rlvr_tests, subject to a 2 MB total budget
    (ten 500 KB cases through a subprocess on every rollout is not viable)
  - remainder becomes truth_tests
Seed, caps, counts and sha256 are recorded in the dataset's manifest.json, so
the exact split is reproducible.

===========================================================================
WHICH DATASET, AND WHY THREE OF THEM EXIST
===========================================================================

  deepmind/code_contests  — the original. public_tests (median 2) are what a
      contestant sees; private_tests (median 2, but 40% of problems have
      none) are held back; generated_tests (median 94) are mutations of
      existing cases and stay SMALL — median input 45 bytes. Our two suites
      are drawn from public+private, at pinned revision 802411c3010c.

  CodeContests-O (caijanfeng)  — regenerates test cases with an iterative
      feedback loop. ~34 cases per problem, and crucially the inputs are at
      the problem's real constraint limits: median input 518 bytes but
      reaching 690 KB on problems where scale matters, vs DeepMind's 45.
      That is why it catches wrong-but-plausible solutions that pass small
      tests — and why it costs ~1.3 MB per problem to run. Used offline only.

  CodeContests+ (ByteDance-Seed)  — a third regeneration, ~27 cases with
      validated true-positive/true-negative rates. NOT used: its test data is
      ~11.5 MB per problem (130 GB for the config), and it is not what the
      earlier inference runs used. See docs/codecontests-dataset-provenance.md.

The short version: DeepMind = small and free, CCO = strong and expensive, and
the design trains on the cheap one while measuring with the strong one.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Optional

from infra.envs.base import Env, SingleTurnEnv, Task
from infra.envs.task_prompts import load_generation_prompts, resolve_prompt_file
from infra.envs.tasks.base import TaskFamily, reject_unknown_keys

PROMPT_FILE = "codecontests.yaml"

logger = logging.getLogger(__name__)

# Concurrent verifier subprocesses. This is the limit that BINDS: the thread
# pool in SingleTurnEnv.rollout also gates on this semaphore, so raising pool
# workers alone changes nothing (measured three times before we noticed).
#
# Was 8, derived from "RLIMIT_AS is 4GB, so 8 x 4GB = 32GB". That arithmetic
# mixed currencies: RLIMIT_AS bounds VIRTUAL address space, while the thing
# worth budgeting is resident memory — measured at ~21MB per verifier process,
# three orders of magnitude less. 4GB is still the right backstop against a
# runaway allocation; it was never a sane basis for a concurrency budget.
#
# 32 measured 2.87x faster wallclock on 632 real completions with 632/632
# identical verdicts. The floor is the timeout itself, not concurrency: a
# fan-out cannot finish before its slowest item.
_MAX_CONCURRENT_VERIFIERS = int(os.environ.get("MAX_CONCURRENT_VERIFIERS", "32"))
_verifier_semaphore = threading.Semaphore(_MAX_CONCURRENT_VERIFIERS)

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

# Strong anchors: at least one must match (case-sensitively) before the weak
# CPP_PATTERNS can classify text as C++. Without this, plain Python such as
# `vector = [0]*n` plus a variable named `null` (NULL under IGNORECASE) scored
# >= 2 weak hits and was graded incorrect WITHOUT execution.
CPP_STRONG_ANCHORS = [
    r"#include\b",
    r"\busing\s+namespace\b",
    r"\bstd::",
    r"\bint\s+main\s*\(",
]

PYTHON_CODE_BLOCK_PATTERN = re.compile(r"```python\s*(.*?)```", re.IGNORECASE | re.DOTALL)
GENERIC_BLOCK_PATTERN = re.compile(r"```\s*(.*?)```", re.DOTALL)
_OPEN_FENCE_PATTERN = re.compile(r"```[ \t]*(?:python)?[ \t]*\r?\n", re.IGNORECASE)


def is_cpp_code(code: str, threshold: int = 2) -> bool:
    if not code:
        return False
    if not any(re.search(p, code) for p in CPP_STRONG_ANCHORS):
        return False
    return sum(1 for p in CPP_PATTERNS if re.search(p, code)) >= threshold


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

        # The runner writes its verdict to this file, never to stdout: the
        # untrusted solution shares the runner's stdout (e.g. via atexit after
        # the StringIO swap is undone) and could forge a result JSON line
        # there. stdout/stderr are captured for diagnostics only.
        result_path = os.path.join(tmpdir, "result.json")

        try:
            proc = subprocess.Popen(
                [sys.executable, runner_path, solution_path, test_cases_path, result_path],
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,  # own process group, so timeout kills grandchildren too
            )
        except Exception as exc:  # noqa: BLE001
            return _failure_result("error", len(inputs), repr(exc), t0)
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.communicate()  # reap
            return _failure_result("timeout", len(inputs), "Total execution timed out.", t0, timeout=True)
        except Exception as exc:  # noqa: BLE001
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            proc.communicate()
            return _failure_result("error", len(inputs), repr(exc), t0)

        elapsed = time.perf_counter() - t0
        try:  # verdict comes ONLY from the runner-written result file
            with open(result_path) as f:
                result = json.load(f)
            result["execution_time_seconds"] = elapsed
            return result
        except (OSError, json.JSONDecodeError, ValueError):
            return _failure_result(
                "error", len(inputs), (stderr or stdout or "")[:1000], t0
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
# swapped to StringIO, then writes one JSON result to the file given as
# argv[3] (atomically, via os.replace) — never to stdout, which the solution
# code shares and could forge.
_TEST_RUNNER_SCRIPT = r'''
import json
import os
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
try:
    resource.setrlimit(resource.RLIMIT_NPROC, (256, 256))
except (ValueError, resource.error):
    pass  # best-effort (macOS)
try:
    fsize_limit = 64 * 1024**2
    resource.setrlimit(resource.RLIMIT_FSIZE, (fsize_limit, fsize_limit))
except (ValueError, resource.error):
    pass  # best-effort (macOS)

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
    result_path = sys.argv[3]

    def write_result(obj):
        # temp name + os.replace: the parent never sees a half-written file
        tmp_path = result_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(obj, f)
        os.replace(tmp_path, result_path)

    # Hide the result path from the solution (it runs in-process and could
    # read sys.argv, then forge the verdict file from an atexit hook).
    sys.argv = [solution_path]

    with open(solution_path) as f:
        solution_code = f.read()

    # Compile once, run many times
    try:
        compiled = compile(solution_code, solution_path, "exec")
    except SyntaxError as e:
        write_result({
            "status": "error", "passed": False,
            "tests_passed": 0, "tests_total": 0, "timeout": False,
            "first_failure": {"test_idx": 0, "expected": "", "actual": "",
                              "stderr": f"SyntaxError: {e}"},
        })
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
    write_result({
        "status": "passed" if passed else "failed",
        "passed": passed,
        "tests_passed": tests_passed,
        "tests_total": tests_total,
        "timeout": False,
        "first_failure": first_failure,
    })

if __name__ == "__main__":
    main()
'''


# ----------------------------------------------------------------------- env


def _load_rows(path: str) -> list[dict[str, Any]]:
    """Read the built dataset. Eligibility (multi-answer, stdin/stdout, size
    caps) was already applied by scripts/build_codecontests_rlvr.py, so the
    only checks here are structural: a usable reward suite, and inputs paired
    with outputs. The runner zips the two lists, so a length mismatch would
    silently grade on a truncated suite rather than erroring."""
    rows: list[dict[str, Any]] = []
    n_total = 0
    n_no_rlvr = 0
    n_len_mismatch = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            e = json.loads(line)
            ri, ro = e.get("rlvr_inputs") or [], e.get("rlvr_outputs") or []
            ti, to = e.get("truth_inputs") or [], e.get("truth_outputs") or []
            if not ri or not ro:
                n_no_rlvr += 1
                continue
            if len(ri) != len(ro) or len(ti) != len(to):
                n_len_mismatch += 1
                continue
            rows.append(
                {
                    "problem": str(e.get("problem", "")).strip(),
                    "name": str(e.get("name", "")).strip(),
                    "rlvr_inputs": list(ri),
                    "rlvr_outputs": list(ro),
                    "truth_inputs": list(ti),
                    "truth_outputs": list(to),
                    "cf_rating": e.get("cf_rating"),
                    "difficulty": e.get("difficulty"),
                }
            )
    n_truth = sum(1 for r in rows if r["truth_inputs"])
    logger.info(
        "codecontests rows from %s: total=%d kept=%d dropped=%d "
        "(no_rlvr_suite=%d len_mismatch=%d); %d/%d have a non-empty truth "
        "suite (the rest are gradeable only by the offline CCO pass)",
        path, n_total, len(rows), n_no_rlvr + n_len_mismatch,
        n_no_rlvr, n_len_mismatch, n_truth, len(rows),
    )
    return rows


class CodeContestsEnv(SingleTurnEnv):
    # Grading shells out to a subprocess per sample, so threads overlap.
    grade_workers = max(1, _MAX_CONCURRENT_VERIFIERS)

    def __init__(
        self,
        path: str,
        test_path: Optional[str] = None,
        seed: int = 0,
        eval_subset_size: int = 128,
        timeout_seconds: int = 90,
        correct_reward: float = 1.0,
        format_reward: float = 0.1,
        prompt_file: Optional[str] = None,
        soft_token_budget: Optional[int] = None,
        overshoot_penalty: float = 0.0,
    ):
        self.rng = random.Random(seed)
        # Flat penalty past the budget (see SingleTurnEnv). Inert unless the
        # backend's response_length is set above the budget.
        self.soft_token_budget = soft_token_budget
        self.overshoot_penalty = overshoot_penalty
        self.prompts = load_generation_prompts(resolve_prompt_file(prompt_file, PROMPT_FILE))
        self.timeout_seconds = timeout_seconds
        self.correct_reward = correct_reward
        self.format_reward = format_reward

        self.train_rows = _load_rows(path)
        if test_path is not None:
            self.test_rows = _load_rows(test_path)
            overlap = {r["name"] for r in self.train_rows if r["name"]} & {
                r["name"] for r in self.test_rows if r["name"]
            }
            if overlap:
                logger.warning(
                    "codecontests: %d problem name(s) present in BOTH train (%s) "
                    "and test (%s); eval on the overlap measures memorization",
                    len(overlap),
                    path,
                    test_path,
                )
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
        # meta["question"] is what DebateEnv binds as the debate TOPIC — the
        # problem statement, and the ONLY field that reaches a prompt.
        #
        # Both suites ride in meta for the graders to read. They are verifier
        # inputs, not public examples: nothing renders them. `name` is the
        # stable key the offline CCO accuracy pass joins on, so it must stay a
        # scalar (docent_export keeps scalars and drops lists — which is also
        # what keeps the suites out of exported transcripts).
        return Task(
            messages=self.prompts.messages(row["problem"]),
            meta={
                "question": row["problem"],
                "name": row["name"],
                "rlvr_inputs": row["rlvr_inputs"],
                "rlvr_outputs": row["rlvr_outputs"],
                "truth_inputs": row["truth_inputs"],
                "truth_outputs": row["truth_outputs"],
                "cf_rating": row.get("cf_rating"),
                "difficulty": row.get("difficulty"),
                "split": split,
            },
        )

    def tasks(self, n: int, split: str = "train") -> list[Task]:
        if split == "train":
            return [self._task(self.rng.choice(self.train_rows), split) for _ in range(n)]
        return [self._task(row, split) for row in self.test_rows[:n]]

    def reward(self, task: Task, text: str) -> tuple[float, dict[str, Any]]:
        """THE RLVR ARM'S REWARD. Runs the program against rlvr_tests only —
        never truth_tests, which exist to measure this arm, not to train it.
        The debate arm does not call this at all (its reward is judge-only)."""
        code = extract_code(text, relaxed=True)
        # every key present in EVERY branch, so eval-time averages are means
        # over all samples rather than over the branch that happened to set it
        info: dict[str, Any] = {
            "correct": 0.0,
            "has_code": float(code is not None),
            "tests_passed_frac": 0.0,
            "cpp_code": 0.0,
            "exec_timeout": 0.0,
            "exec_error": 0.0,
        }
        if code is None:
            return 0.0, info
        if is_cpp_code(code):
            info["cpp_code"] = 1.0
            return self.format_reward, info

        result = run_stdin_tests(
            code, task.meta["rlvr_inputs"], task.meta["rlvr_outputs"],
            timeout=self.timeout_seconds,
        )
        total = result.get("tests_total") or len(task.meta["rlvr_inputs"])
        info["correct"] = float(bool(result.get("passed")))
        info["tests_passed_frac"] = result.get("tests_passed", 0) / total if total else 0.0
        # verifier breakage must be distinguishable from wrong answers
        info["exec_timeout"] = float(bool(result.get("timeout")))
        info["exec_error"] = float(result.get("status") == "error")
        return self.format_reward + self.correct_reward * info["correct"], info


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
                "prompt_file",
                "soft_token_budget",
                "overshoot_penalty",
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
            prompt_file=(str(ds["prompt_file"]) if ds.get("prompt_file") else None),
            soft_token_budget=(int(ds["soft_token_budget"]) if ds.get("soft_token_budget") else None),
            overshoot_penalty=float(ds.get("overshoot_penalty", 0.0)),
        )

    def extractor(self, relaxed: bool) -> Callable[[str], Any]:
        return lambda text: extract_code(text, relaxed=relaxed)

    def grade(self, meta: dict[str, Any], solution: Any) -> Optional[bool]:
        """GROUND-TRUTH LABEL, never a reward. DebateEnv calls this to record
        whether the proposer was actually right; the label goes to the
        transcript and to info["solution_correct"], and the judge never sees it.

        Uses truth_tests, which the RLVR arm was NOT trained on. Returns None
        when a problem has no truth suite (~66% of them — the <=10 sample
        consumed every case). Deliberately no fallback to rlvr_tests: on an
        RLVR-trained policy that would be scoring the training signal and
        would read as inflated accuracy. Those problems get their label from
        the offline CodeContests-O pass instead (see the module docstring)."""
        if solution is None:
            return None
        inputs, outputs = meta.get("truth_inputs"), meta.get("truth_outputs")
        if not inputs or not outputs:
            return None
        if is_cpp_code(solution):
            return False
        return bool(run_stdin_tests(solution, inputs, outputs, timeout=self.timeout_seconds).get("passed"))

    def format_flags(self, text: str) -> dict[str, float]:
        # strict flag: a properly CLOSED fence, independent of the (possibly
        # relaxed) extractor used for position binding
        return {"code_fence": float(has_closed_fence(text))}
