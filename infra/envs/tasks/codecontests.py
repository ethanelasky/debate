"""CodeContests task family: competitive programming problems verified by
running the candidate program against stdin/stdout test cases.

Ported from the old repo (ai_debate/data_loader/codecontests_loader.py and
ai_debate/experiments/verifiers/codecontests_verifier.py), slimmed: rows are
held in memory (no lazy-JSONL dict). The default local runner is a plain
subprocess; datasets may instead select a remote Piston execution service.

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
TRAINING AND PAIRED EVALUATION SUITES
===========================================================================

Training rows carry two disjoint suites built by
scripts/build_codecontests_rlvr.py:

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

The held-out artifact is built by scripts/build_codecontests_paired_eval.py.
Each row is self-contained and carries stable problem metadata plus explicitly
named ``gdm_inputs/gdm_outputs`` and ``cco_inputs/cco_outputs`` suites. The
runtime never joins a sidecar.

No suite is ever rendered into a prompt or exported in a transcript. Models
see the problem statement only. _task() keeps test cases in private Task.meta
for reward/measurement, while CodeContestsEnv.export_meta() uses an explicit
safe scalar allowlist before a rollout record can reach Docent or W&B.

One thing that looks like a leak and is not: some rlvr_tests cases DO appear
verbatim in the problem statement (~16% of non-trivial cases across the test
split). Those are the PUBLIC tests — DeepMind's public_tests are exactly the
worked examples printed in a Codeforces problem, so a contestant sees them
too. The model reading them off the statement is the intended setting, and it
is part of why this reward is weak. Private tests never appear there.

===========================================================================
THE TWO EVAL SUITES (both graded in-run)
===========================================================================

Evaluation runs the held-out split against TWO independent test suites and
reports each separately, so the same policy is measured by a weak in-
distribution suite and a strong out-of-distribution one on the same axis:

  gdm_eval   the deterministic <=10-case sample of DeepMind public+private
             tests. Same KIND of test the RLVR arm trains on, but evaluated on
             contest-disjoint problems it never saw.

  cco_eval   a deterministic <=10-case sample of CodeContests-O corner cases
             (drawn from ~34/problem, with inputs at the real constraint
             limits). Much stronger: it catches
             wrong-but-plausible solutions that pass DeepMind's small cases.
             The paired artifact covers 456 of 501 held-out problems after
             four conflicting-output problems are removed.

The gap between the two curves is the quantity of interest. A policy that
climbs on gdm_eval while flat on cco_eval is learning to satisfy small tests
rather than to solve problems.

The same greedy completion is graded against both during the run, so
eval/gdm_correct and eval/cco_correct are live in wandb and overlayable. Both
suites use the same <=10-case, 500 KB/case, 2 MB/problem caps.

===========================================================================
HOW THE VERIFIER WORKS
===========================================================================

run_stdin_tests() is the trusted supervisor. It executes a fresh Python process
for EVERY case, gives it only the solution and that case's stdin, and compares
the output in the parent. Expected outputs and verifier results never enter
the candidate process. The default local backend captures output in bounded
regular files, kills the whole process group on timeout, and applies
AS/CPU/file-size/process/fd/core limits best-effort before executing the
candidate. The optional Piston backend replaces only that per-case execution
call; validation, the shared deadline, comparison, and verdicts stay here.
Piston receives the source directly, so unlike the local bootstrap it cannot
distinguish ``os._exit(0)`` from an ordinary top-level return.

This is process isolation for grading integrity, not a malicious-host sandbox.
A same-UID process can still attack its parent or host; hostile candidates need
an OS/container boundary.

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
      tests. We deterministically cap it to <=10 cases and 2 MB per problem,
      pair it with GDM at build time, and use it only for held-out evaluation.
      It never affects the training reward.

  CodeContests+ (ByteDance-Seed)  — a third regeneration, ~27 cases with
      validated true-positive/true-negative rates. NOT used: 951 GB across
      five configs (46 GB for the smallest), and it is not what the earlier
      inference runs used. See docs/codecontests-dataset-provenance.md.

The short version: DeepMind = small and free, CCO = strong and affordable at
eval scale. The design trains on the cheap suite and measures with both.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
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
from pathlib import Path
from typing import Any, Optional

from infra.envs.base import Env, SingleTurnEnv, Task
from infra.envs.task_prompts import load_generation_prompts, resolve_prompt_file
from infra.envs.tasks import piston
from infra.envs.tasks.base import (
    AnswerParse,
    GraderInfrastructureError,
    TaskFamily,
    reject_unknown_keys,
)

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

# Piston queues above its own capacity without charging that queue time to the
# candidate's run limit or cancelling jobs whose HTTP clients disconnect. Keep
# requests out of that hidden queue. The default matches deploy/piston; if
# several trainer processes share one service, their configured totals must not
# exceed the service's PISTON_MAX_CONCURRENT_JOBS.
_MAX_CONCURRENT_PISTON_VERIFIERS = int(
    os.environ.get("MAX_CONCURRENT_PISTON_VERIFIERS", "4")
)
if not 1 <= _MAX_CONCURRENT_PISTON_VERIFIERS <= piston.MAX_CONCURRENT_JOBS:
    raise ValueError(
        "MAX_CONCURRENT_PISTON_VERIFIERS must be between 1 and "
        f"{piston.MAX_CONCURRENT_JOBS} for {piston.PROTOCOL_ID}"
    )
_piston_verifier_semaphore = threading.Semaphore(
    _MAX_CONCURRENT_PISTON_VERIFIERS
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
_EXACT_PISTON_PYTHON_VERSION_PATTERN = re.compile(
    r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z]+(?:[.-][0-9A-Za-z]+)*)?"
    r"(?:\+[0-9A-Za-z]+(?:[.-][0-9A-Za-z]+)*)?"
)


def _validate_verifier_settings(
    verifier: str,
    piston_url: Optional[str],
    piston_python_version: Optional[str],
) -> None:
    if not isinstance(verifier, str) or verifier not in {"local", "piston"}:
        raise ValueError(
            f"codecontests verifier must be 'local' or 'piston' (got {verifier!r})"
        )
    if verifier == "local":
        if piston_url is not None or piston_python_version is not None:
            raise ValueError(
                "codecontests local verifier does not accept Piston-only settings"
            )
        return
    if not isinstance(piston_url, str) or not piston_url.strip():
        raise ValueError(
            "codecontests verifier='piston' requires a nonempty piston_url"
        )
    if (
        not isinstance(piston_python_version, str)
        or not _EXACT_PISTON_PYTHON_VERSION_PATTERN.fullmatch(
            piston_python_version
        )
    ):
        raise ValueError(
            "codecontests verifier='piston' requires an exact semantic "
            "piston_python_version (for example '3.10.0'); wildcards and "
            "version ranges are not allowed"
        )
    try:
        piston.validate_settings(
            base_url=piston_url,
            runtime_version=piston_python_version,
        )
    except GraderInfrastructureError as exc:
        raise ValueError(f"invalid CodeContests Piston settings: {exc}") from exc


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


def parse_code_answers(text: str) -> AnswerParse:
    """Extract the strict envelope and the relaxed reward candidate once.

    Strict means a non-empty, closed Python or generic Markdown fence.  The
    relaxed candidate additionally accepts the trailing unclosed fence used by
    the historical RLVR reward when a generation is truncated at its cap.
    """
    return AnswerParse(
        strict=extract_code(text, relaxed=False),
        relaxed=extract_code(text, relaxed=True),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _suite_projection(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The executable suite bytes, retaining row and case order."""
    suite_keys = (
        "name",
        "rlvr_inputs",
        "rlvr_outputs",
        "truth_inputs",
        "truth_outputs",
        "gdm_inputs",
        "gdm_outputs",
        "cco_inputs",
        "cco_outputs",
    )
    return [{key: row.get(key) for key in suite_keys} for row in rows]


def _manifest_for(path: Path) -> Optional[Path]:
    """Find the builder manifest associated with an artifact, if present."""
    candidates = (path.with_suffix(".manifest.json"), path.parent / "manifest.json")
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _optional(value: Any) -> str:
    return "none" if value is None else str(value)


# ------------------------------------------------------------------ verifier


def run_stdin_tests(
    solution_code: str,
    inputs: list[str],
    outputs: list[str],
    timeout: int = 90,
    *,
    verifier: str = "local",
    piston_url: Optional[str] = None,
    piston_python_version: Optional[str] = None,
) -> dict[str, Any]:
    """Run cases in fresh candidate processes under one solution deadline.

    This function is the trusted supervisor: expected outputs remain only in
    this process, while each candidate receives its source path and the
    current case on stdin.  Candidate failures are ordinary ``False``
    verdicts; failures of process launch/communication or this supervisor's
    result contract invalidate grading via ``GraderInfrastructureError``.
    """
    t0 = time.perf_counter()
    tmpdir: Optional[str] = None
    acquired = False
    try:
        if len(inputs) != len(outputs):
            raise GraderInfrastructureError(
                "codecontests verifier received mismatched input/output suites"
            )
        if timeout <= 0:
            raise GraderInfrastructureError(
                "codecontests verifier requires a positive total timeout"
            )
        if not all(isinstance(case, str) for case in [*inputs, *outputs]):
            raise GraderInfrastructureError(
                "codecontests verifier cases must be strings"
            )
        _validate_verifier_settings(verifier, piston_url, piston_python_version)

        if verifier == "local":
            # Compile with the same interpreter that executes the local
            # candidate. SyntaxError and its IndentationError/TabError
            # subclasses are candidate failures, classified here by trusted
            # Python state rather than by grepping a runtime traceback (which
            # would misclassify ``raise SyntaxError``). Piston source must be
            # parsed by its configured runtime instead: the trainer and judge
            # Python versions need not match.
            try:
                compile(solution_code.encode("utf-8"), "solution.py", "exec")
            except SyntaxError as exc:
                return {
                    "status": "candidate_error",
                    "passed": False,
                    "tests_passed": 0,
                    "tests_total": len(inputs),
                    "timeout": False,
                    "first_failure": {
                        "test_idx": 0,
                        "expected": "",
                        "actual": "",
                        "stderr": f"{type(exc).__name__}: {exc}",
                    },
                    "execution_time_seconds": time.perf_counter() - t0,
                }

        if verifier == "local":
            tmpdir = tempfile.mkdtemp(prefix="codecontests_test_")
        verifier_semaphore = (
            _piston_verifier_semaphore
            if verifier == "piston"
            else _verifier_semaphore
        )
        verifier_semaphore.acquire()
        acquired = True
        # One candidate-execution budget covers all cases for this solution.
        # Local execution is supervised directly against a monotonic deadline.
        # Piston is a trusted remote supervisor, so its validated isolate wall
        # time is accumulated instead; HTTP/tunnel latency and trusted server
        # setup must not consume candidate execution time.
        deadline = time.perf_counter() + timeout if verifier == "local" else None
        candidate_seconds_remaining = float(timeout)
        tests_passed = 0
        first_failure: Optional[dict[str, Any]] = None
        saw_candidate_error = False
        timed_out = False

        for i, (test_input, expected_output) in enumerate(zip(inputs, outputs)):
            remaining = (
                deadline - time.perf_counter()
                if deadline is not None
                else candidate_seconds_remaining
            )
            if remaining <= 0:
                case = {
                    "returncode": -signal.SIGKILL,
                    "timed_out": True,
                    "output_limited": False,
                    "stdout": "",
                    "stderr": "Total solution execution timed out.",
                }
                if verifier == "piston":
                    case["candidate_time_seconds"] = 0.0
            else:
                if verifier == "piston":
                    case = piston.run_python_case(
                        base_url=piston_url,
                        runtime_version=piston_python_version,
                        solution_code=solution_code,
                        test_input=test_input,
                        remaining_seconds=remaining,
                    )
                else:
                    case = _run_candidate_case(
                        solution_code=solution_code,
                        test_input=test_input,
                        remaining=remaining,
                        tmpdir=tmpdir,
                    )
            _validate_case_result(
                case, require_candidate_time=verifier == "piston"
            )
            if verifier == "piston":
                candidate_time = case["candidate_time_seconds"]
                if not case["timed_out"] and candidate_time > remaining:
                    raise GraderInfrastructureError(
                        "Piston reported a non-timeout execution beyond the "
                        "remaining candidate budget"
                    )
                candidate_seconds_remaining -= candidate_time

            actual = _normalize_output(case["stdout"])
            expected = _normalize_output(expected_output)
            case_ok = (
                case["returncode"] == _NORMAL_RETURN_CODE
                and not case["timed_out"]
                and not case["output_limited"]
                and actual == expected
            )
            if case_ok:
                tests_passed += 1
            else:
                # Compile failures were classified before launch. Output-limit
                # failures remain candidate errors; runtime exceptions and
                # signals are simply failed solutions.
                if case["output_limited"]:
                    saw_candidate_error = True
                if first_failure is None:
                    stderr = case["stderr"]
                    if case["output_limited"]:
                        stderr = "Candidate exceeded the output limit.\n" + stderr
                    if case["timed_out"]:
                        stderr = "Total solution execution timed out.\n" + stderr
                    first_failure = {
                        "test_idx": i,
                        "expected": expected[:500],
                        "actual": actual[:500],
                        "stderr": stderr[:500],
                    }
            if case["timed_out"]:
                timed_out = True
                break

        passed = tests_passed == len(inputs) and not timed_out
        status = (
            "passed"
            if passed
            else "timeout"
            if timed_out
            else "candidate_error"
            if saw_candidate_error
            else "failed"
        )
        return {
            "status": status,
            "passed": passed,
            "tests_passed": tests_passed,
            "tests_total": len(inputs),
            "timeout": timed_out,
            "first_failure": first_failure,
            "execution_time_seconds": time.perf_counter() - t0,
        }
    except GraderInfrastructureError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise GraderInfrastructureError(
            "codecontests verifier failed outside candidate execution"
        ) from exc
    finally:
        if acquired:
            verifier_semaphore.release()
        if tmpdir is not None:
            shutil.rmtree(tmpdir, ignore_errors=True)


_OUTPUT_LIMIT_BYTES = 8 * 1024 * 1024
# The isolated bootstrap reports that candidate top-level execution returned
# normally with this exit status. A direct ``os._exit(0)`` bypasses it and is
# therefore distinguishable from an ordinary successful script termination.
_NORMAL_RETURN_CODE = 120


def _normalize_output(value: str) -> str:
    normalized = []
    for line in value.split("\n"):
        line = line.rstrip()

        def normalize_float(match: re.Match[str]) -> str:
            try:
                return f"{float(match.group(0)):g}"
            except (ValueError, OverflowError):
                return match.group(0)

        normalized.append(
            re.sub(r"-?\d+\.\d+(?:[eE][+-]?\d+)?", normalize_float, line)
        )
    while normalized and not normalized[-1]:
        normalized.pop()
    return "\n".join(normalized).lower()


def _minimal_candidate_env() -> dict[str, str]:
    """No credentials or experiment controls cross into candidate code."""
    env = {"PATH": os.defpath}
    for key in ("LANG", "LC_ALL"):
        value = os.environ.get(key)
        if value:
            env[key] = value
    return env


def _kill_and_reap_process_group(proc: subprocess.Popen) -> None:
    """Kill candidate descendants and ensure the direct child is reaped."""
    pid = getattr(proc, "pid", None)
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise GraderInfrastructureError(
            "codecontests verifier worker exposed an invalid pid"
        )
    try:
        os.killpg(pid, signal.SIGKILL)  # start_new_session makes pgid == pid
    except ProcessLookupError:
        pass
    except OSError as exc:
        raise GraderInfrastructureError(
            "codecontests verifier could not terminate a candidate process group"
        ) from exc
    try:
        proc.communicate(timeout=2)
    except subprocess.TimeoutExpired as exc:
        raise GraderInfrastructureError(
            "codecontests verifier could not reap a candidate subprocess"
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise GraderInfrastructureError(
            "codecontests verifier lost communication while reaping a candidate"
        ) from exc


def _run_candidate_case(
    *,
    solution_code: str,
    test_input: str,
    remaining: float,
    tmpdir: str,
) -> dict[str, Any]:
    """Launch one candidate process. Gold never crosses this boundary.

    Launcher bytes are supplied afresh through Python's ``-c`` argument, and
    the source lives in a newly created per-case directory. Nothing a prior
    candidate can rewrite is executed by a later case.
    """
    case_dir = tempfile.mkdtemp(prefix="case_", dir=tmpdir)
    solution_path = os.path.join(case_dir, "solution.py")
    try:
        # Exclusive creation avoids following a path planted by an escaped
        # descendant. The directory is private and did not exist previously.
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(solution_path, flags, 0o400)
        with os.fdopen(fd, "w", encoding="utf-8") as solution_file:
            solution_file.write(solution_code)

        with tempfile.TemporaryFile(mode="w+b", dir=case_dir) as stdout_file, tempfile.TemporaryFile(
            mode="w+b", dir=case_dir
        ) as stderr_file:
            result = _supervise_candidate_process(
                solution_path=solution_path,
                test_input=test_input,
                remaining=remaining,
                cwd=case_dir,
                stdout_file=stdout_file,
                stderr_file=stderr_file,
            )
            return result
    except GraderInfrastructureError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise GraderInfrastructureError(
            "codecontests verifier failed while preparing a candidate case"
        ) from exc
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def _supervise_candidate_process(
    *,
    solution_path: str,
    test_input: str,
    remaining: float,
    cwd: str,
    stdout_file,
    stderr_file,
) -> dict[str, Any]:
    """Start, bound, reap, and collect one already-prepared candidate."""
    try:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-B",
                "-c",
                _RESOURCE_BOOTSTRAP,
                solution_path,
                str(max(1, int(remaining + 0.999))),
            ],
            stdin=subprocess.PIPE,
            stdout=stdout_file,
            stderr=stderr_file,
            env=_minimal_candidate_env(),
            cwd=cwd,
            close_fds=True,
            start_new_session=True,
        )
    except Exception as exc:  # noqa: BLE001
        raise GraderInfrastructureError(
            "codecontests verifier failed to start a candidate subprocess"
        ) from exc

    timed_out = False
    communication_error: Optional[BaseException] = None
    try:
        proc.communicate(input=test_input.encode("utf-8"), timeout=remaining)
    except subprocess.TimeoutExpired:
        timed_out = True
    except Exception as exc:  # noqa: BLE001
        communication_error = exc
    finally:
        _kill_and_reap_process_group(proc)
    if communication_error is not None:
        raise GraderInfrastructureError(
            "codecontests verifier lost communication with a candidate subprocess"
        ) from communication_error

    returncode = getattr(proc, "returncode", None)
    if isinstance(returncode, bool) or not isinstance(returncode, int):
        raise GraderInfrastructureError(
            "codecontests verifier worker exposed an invalid return code"
        )
    try:
        stdout_size = os.fstat(stdout_file.fileno()).st_size
        stderr_size = os.fstat(stderr_file.fileno()).st_size
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout = stdout_file.read(_OUTPUT_LIMIT_BYTES + 1).decode(
            "utf-8", errors="replace"
        )
        stderr = stderr_file.read(_OUTPUT_LIMIT_BYTES + 1).decode(
            "utf-8", errors="replace"
        )
    except OSError as exc:
        raise GraderInfrastructureError(
            "codecontests verifier could not read captured candidate output"
        ) from exc
    output_limited = (
        stdout_size >= _OUTPUT_LIMIT_BYTES
        or stderr_size >= _OUTPUT_LIMIT_BYTES
        or returncode == -getattr(signal, "SIGXFSZ", 25)
    )
    return {
        "returncode": returncode,
        "timed_out": timed_out,
        "output_limited": output_limited,
        "stdout": stdout,
        "stderr": stderr,
    }


def _validate_case_result(
    result: Any, *, require_candidate_time: bool = False
) -> None:
    """Treat a broken trusted-supervisor contract as fatal, never wrong."""
    if not isinstance(result, dict):
        raise GraderInfrastructureError(
            "codecontests verifier supervisor returned a non-object case result"
        )
    expected_fields = {
        "returncode",
        "timed_out",
        "output_limited",
        "stdout",
        "stderr",
    }
    if require_candidate_time:
        expected_fields.add("candidate_time_seconds")
    if set(result) != expected_fields:
        raise GraderInfrastructureError(
            "codecontests verifier supervisor returned an invalid case schema"
        )
    if (
        isinstance(result["returncode"], bool)
        or not isinstance(result["returncode"], int)
        or not isinstance(result["timed_out"], bool)
        or not isinstance(result["output_limited"], bool)
        or not isinstance(result["stdout"], str)
        or not isinstance(result["stderr"], str)
    ):
        raise GraderInfrastructureError(
            "codecontests verifier supervisor returned invalid case field types"
        )
    if require_candidate_time:
        candidate_time = result["candidate_time_seconds"]
        if (
            isinstance(candidate_time, bool)
            or not isinstance(candidate_time, (int, float))
            or not math.isfinite(candidate_time)
            or candidate_time < 0
        ):
            raise GraderInfrastructureError(
                "codecontests verifier supervisor returned invalid candidate time"
            )


# Limits are installed inside the isolated child, avoiding preexec_fn (unsafe
# when verifier launches overlap in threads). The bootstrap has no gold,
# result path, or control descriptor; it only distinguishes a normal top-level
# return from hard exits. Candidate argv contains only its own source path.
_RESOURCE_BOOTSTRAP = r'''
import os
import resource
import sys

def set_limit(name, requested):
    kind = getattr(resource, name, None)
    if kind is None:
        return
    try:
        _soft, hard = resource.getrlimit(kind)
        value = requested if hard == resource.RLIM_INFINITY else min(requested, hard)
        resource.setrlimit(kind, (value, value))
    except (OSError, ValueError):
        pass

set_limit("RLIMIT_AS", 4 * 1024 * 1024 * 1024)
# Keep the CPU backstop slightly beyond the parent's wall-clock deadline so a
# busy loop is classified as a timeout rather than an ambiguous signal exit.
set_limit("RLIMIT_CPU", max(1, int(sys.argv[2]) + 2))
set_limit("RLIMIT_FSIZE", 8 * 1024 * 1024)
set_limit("RLIMIT_NPROC", 256)
set_limit("RLIMIT_NOFILE", 64)
set_limit("RLIMIT_CORE", 0)

solution = sys.argv[1]
sys.argv = [solution]
if hasattr(sys, "orig_argv"):
    sys.orig_argv = [sys.executable, solution]
with open(solution, "rb") as source_file:
    source = source_file.read()
compiled = compile(source, solution, "exec")

class CandidateHardExit(BaseException):
    pass

hard_exit_attempted = False

def candidate_hard_exit(code=0):
    global hard_exit_attempted
    hard_exit_attempted = True
    raise CandidateHardExit(code)

# Distinguish Python's ordinary os._exit API from a top-level return. This is
# grading-integrity instrumentation, not a hostile-code sandbox: native
# syscalls remain part of the documented same-UID/container limitation.
os._exit = candidate_hard_exit
try:
    import posix
    posix._exit = candidate_hard_exit
except ImportError:
    pass

try:
    exec(compiled, {"__name__": "__main__", "__file__": solution})
except CandidateHardExit:
    raise SystemExit(1)
except SystemExit as exc:
    # ``exit()``/``sys.exit()`` with a success code is common in valid contest
    # programs and counts as a normal top-level return. Nonzero exits remain
    # candidate failures.
    if exc.code not in (None, 0):
        raise SystemExit(1)
if hard_exit_attempted:
    raise SystemExit(1)
raise SystemExit(120)
'''


# ----------------------------------------------------------------------- env


def _load_paired_rows(path: str) -> list[dict[str, Any]]:
    """Read a self-contained GDM+CCO held-out evaluation artifact.

    Both suites are required on every row. Keeping the join at build time makes
    the evaluated population inspectable and prevents a missing/stale runtime
    sidecar from silently changing it.
    """
    rows: list[dict[str, Any]] = []
    n_bad = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            name = str(e.get("name", "")).strip()
            problem = str(e.get("problem", "")).strip()
            rating = e.get("cf_rating")
            gi, go = e.get("gdm_inputs") or [], e.get("gdm_outputs") or []
            ci, co = e.get("cco_inputs") or [], e.get("cco_outputs") or []
            if (
                not name
                or not problem
                or isinstance(rating, bool)
                or not isinstance(rating, (int, float))
                or not gi
                or not ci
                or len(gi) != len(go)
                or len(ci) != len(co)
            ):
                n_bad += 1
                continue
            rows.append(
                {
                    "problem": problem,
                    "name": name,
                    "gdm_inputs": list(gi),
                    "gdm_outputs": list(go),
                    "cco_inputs": list(ci),
                    "cco_outputs": list(co),
                    "cf_contest_id": e.get("cf_contest_id"),
                    "cf_index": e.get("cf_index"),
                    "cf_rating": rating,
                    "difficulty": e.get("difficulty"),
                    "source": e.get("source"),
                }
            )
    if n_bad:
        logger.warning("paired eval suite: dropped %d malformed rows from %s", n_bad, path)
    return rows


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
        "suite (the rest are covered by cco_eval)",
        path, n_total, len(rows), n_no_rlvr + n_len_mismatch,
        n_no_rlvr, n_len_mismatch, n_truth, len(rows),
    )
    return rows


def _filter_cf_rating(
    rows: list[dict[str, Any]],
    min_cf_rating: Optional[int],
    max_cf_rating: Optional[int],
) -> list[dict[str, Any]]:
    """Keep rows inside an inclusive Codeforces rating range.

    GDM uses ``0`` for problems without a matched Codeforces rating.  Requiring
    a numeric value inside the requested interval therefore also keeps those
    unrated rows from leaking into an "easy" slice.
    """
    if min_cf_rating is None and max_cf_rating is None:
        return rows
    kept: list[dict[str, Any]] = []
    for row in rows:
        rating = row.get("cf_rating")
        if isinstance(rating, bool) or not isinstance(rating, (int, float)):
            continue
        if min_cf_rating is not None and rating < min_cf_rating:
            continue
        if max_cf_rating is not None and rating > max_cf_rating:
            continue
        kept.append(row)
    return kept


class CodeContestsEnv(SingleTurnEnv):
    # Grading shells out to a subprocess per sample, so threads overlap.
    grade_workers = max(1, _MAX_CONCURRENT_VERIFIERS)

    def __init__(
        self,
        path: str,
        test_path: Optional[str] = None,
        paired_test_path: Optional[str] = None,
        seed: int = 0,
        eval_subset_size: int = 128,
        expected_eval_size: Optional[int] = None,
        timeout_seconds: int = 90,
        correct_reward: float = 1.0,
        format_reward: float = 0.1,
        prompt_file: Optional[str] = None,
        soft_token_budget: Optional[int] = None,
        overshoot_penalty: float = 0.0,
        min_cf_rating: Optional[int] = None,
        max_cf_rating: Optional[int] = None,
        verifier: str = "local",
        piston_url: Optional[str] = None,
        piston_python_version: Optional[str] = None,
    ):
        _validate_verifier_settings(verifier, piston_url, piston_python_version)
        if (
            min_cf_rating is not None
            and max_cf_rating is not None
            and min_cf_rating > max_cf_rating
        ):
            raise ValueError(
                "codecontests min_cf_rating must be <= max_cf_rating "
                f"(got {min_cf_rating} > {max_cf_rating})"
            )
        self.rng = random.Random(seed)
        # Flat penalty past the budget (see SingleTurnEnv). Inert unless the
        # backend's response_length is set above the budget.
        self.soft_token_budget = soft_token_budget
        self.overshoot_penalty = overshoot_penalty
        self.prompts = load_generation_prompts(resolve_prompt_file(prompt_file, PROMPT_FILE))
        self.timeout_seconds = timeout_seconds
        self.correct_reward = correct_reward
        self.format_reward = format_reward
        self.verifier = verifier
        self.piston_url = piston_url
        self.piston_python_version = piston_python_version
        self.grade_workers = (
            _MAX_CONCURRENT_PISTON_VERIFIERS
            if verifier == "piston"
            else _MAX_CONCURRENT_VERIFIERS
        )

        loaded_train_rows = _load_rows(path)
        self.train_rows = _filter_cf_rating(
            loaded_train_rows, min_cf_rating, max_cf_rating
        )
        if test_path is not None and paired_test_path is not None:
            raise ValueError("codecontests accepts only one of test_path and paired_test_path")
        if test_path is not None or paired_test_path is not None:
            loaded_test_rows = (
                _load_paired_rows(paired_test_path)
                if paired_test_path is not None
                else _load_rows(test_path)
            )
            self.test_rows = _filter_cf_rating(
                loaded_test_rows, min_cf_rating, max_cf_rating
            )
            if min_cf_rating is not None or max_cf_rating is not None:
                logger.info(
                    "codecontests cf_rating filter [%s, %s]: train=%d/%d test=%d/%d",
                    min_cf_rating,
                    max_cf_rating,
                    len(self.train_rows),
                    len(loaded_train_rows),
                    len(self.test_rows),
                    len(loaded_test_rows),
                )
            overlap = {r["name"] for r in self.train_rows if r["name"]} & {
                r["name"] for r in self.test_rows if r["name"]
            }
            if overlap:
                logger.warning(
                    "codecontests: %d problem name(s) present in BOTH train (%s) "
                    "and test (%s); eval on the overlap measures memorization",
                    len(overlap),
                    path,
                    paired_test_path or test_path,
                )
        else:
            rng = random.Random(seed + 22222)
            rows = list(self.train_rows)
            rng.shuffle(rows)
            n_test = min(max(64, len(rows) // 10), max(0, len(rows) - 2))
            self.test_rows, self.train_rows = rows[:n_test], rows[n_test:]
        if expected_eval_size is not None and len(self.test_rows) != expected_eval_size:
            raise RuntimeError(
                "codecontests paired eval population changed: "
                f"expected {expected_eval_size}, got {len(self.test_rows)}"
            )
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
        # Suites ride in meta for the graders to read. They are verifier inputs,
        # not public examples: nothing renders them, and export_meta() removes
        # them before any transcript record is constructed.
        return Task(
            messages=self.prompts.render({"PROBLEM": row["problem"]}),
            meta={
                "question": row["problem"],
                "name": row["name"],
                "rlvr_inputs": row.get("rlvr_inputs"),
                "rlvr_outputs": row.get("rlvr_outputs"),
                "truth_inputs": row.get("truth_inputs"),
                "truth_outputs": row.get("truth_outputs"),
                "gdm_inputs": row.get("gdm_inputs"),
                "gdm_outputs": row.get("gdm_outputs"),
                "cco_inputs": row.get("cco_inputs"),
                "cco_outputs": row.get("cco_outputs"),
                "cf_rating": row.get("cf_rating"),
                "difficulty": row.get("difficulty"),
                "split": split,
            },
        )

    def tasks(self, n: int, split: str = "train") -> list[Task]:
        if split == "train":
            return [self._task(self.rng.choice(self.train_rows), split) for _ in range(n)]
        return [self._task(row, split) for row in self.test_rows[:n]]

    def export_meta(self, task: Task) -> dict[str, Any]:
        """Expose problem identity, never private verifier suites.

        This explicit allowlist is intentionally narrower than the generic
        SingleTurnEnv default.  Adding a suite-like field to Task.meta cannot
        silently make it into ``last_rollout_records`` and downstream Docent
        or W&B payloads.
        """
        safe_keys = ("question", "name", "cf_rating", "difficulty", "split")
        scalar_types = (str, int, float, bool)
        return {
            key: value
            for key in safe_keys
            if key in task.meta
            and ((value := task.meta[key]) is None or isinstance(value, scalar_types))
        }

    def reward(self, task: Task, text: str) -> tuple[float, dict[str, Any]]:
        """GDM-only RLVR reward plus paired held-out measurement.

        Training tasks use ``rlvr_inputs``. Paired held-out tasks use the
        explicitly named ``gdm_inputs`` and additionally run the SAME extracted
        program through CCO. CCO metrics never enter the returned reward.
        """
        parsed = parse_code_answers(text)
        code = parsed.relaxed
        # every key present in EVERY branch, so eval-time averages are means
        # over all samples rather than over the branch that happened to set it
        info: dict[str, Any] = {
            "answer_format_valid": float(parsed.answer_format_valid),
            "correct_strict": 0.0,
            "correct_relaxed": 0.0,
            "gdm_correct": 0.0,
            "gdm_tests_passed_frac": 0.0,
            "cpp_code": 0.0,
            "exec_timeout": 0.0,
            "exec_error": 0.0,
        }
        has_cco = bool(task.meta.get("cco_inputs"))
        if has_cco:
            info["cco_correct"] = 0.0
            info["cco_tests_passed_frac"] = 0.0
        if code is None:
            return 0.0, info
        if is_cpp_code(code):
            info["cpp_code"] = 1.0
            return self.format_reward, info

        verifier_kwargs: dict[str, Any] = {}
        if self.verifier == "piston":
            verifier_kwargs = {
                "verifier": self.verifier,
                "piston_url": self.piston_url,
                "piston_python_version": self.piston_python_version,
            }

        if has_cco:
            cco = run_stdin_tests(
                code, task.meta["cco_inputs"], task.meta["cco_outputs"],
                timeout=self.timeout_seconds,
                **verifier_kwargs,
            )
            cco_total = cco.get("tests_total") or len(task.meta["cco_inputs"])
            info["cco_correct"] = float(bool(cco.get("passed")))
            info["cco_tests_passed_frac"] = (
                cco.get("tests_passed", 0) / cco_total if cco_total else 0.0
            )

        gdm_inputs = task.meta.get("gdm_inputs") or task.meta["rlvr_inputs"]
        gdm_outputs = task.meta.get("gdm_outputs") or task.meta["rlvr_outputs"]
        result = run_stdin_tests(
            code, gdm_inputs, gdm_outputs,
            timeout=self.timeout_seconds,
            **verifier_kwargs,
        )
        total = result.get("tests_total") or len(gdm_inputs)
        gdm_correct = float(bool(result.get("passed")))
        gdm_passed_frac = result.get("tests_passed", 0) / total if total else 0.0
        info["gdm_correct"] = info["correct_relaxed"] = gdm_correct
        info["correct_strict"] = (
            gdm_correct if parsed.strict is not None else 0.0
        )
        info["gdm_tests_passed_frac"] = gdm_passed_frac
        # verifier breakage must be distinguishable from wrong answers
        info["exec_timeout"] = float(bool(result.get("timeout")))
        info["exec_error"] = float(result.get("status") == "candidate_error")
        # Historical reward semantics deliberately use the relaxed candidate:
        # a truncated fence still earns format reward and can earn correctness.
        return self.format_reward + self.correct_reward * gdm_correct, info


# -------------------------------------------------------------------- family


class CodeContestsFamily(TaskFamily):
    def __init__(
        self,
        timeout_seconds: int = 90,
        verifier: str = "local",
        piston_url: Optional[str] = None,
        piston_python_version: Optional[str] = None,
    ):
        _validate_verifier_settings(verifier, piston_url, piston_python_version)
        # grade() is called by DebateEnv with only meta, so the per-problem
        # verifier settings have to live on the family instance.
        self.timeout_seconds = timeout_seconds
        self.verifier = verifier
        self.piston_url = piston_url
        self.piston_python_version = piston_python_version
        if verifier == "piston":
            self.grade_workers = _MAX_CONCURRENT_PISTON_VERIFIERS
        self._protocol_identity: dict[str, str] = {}

    def source(self, ds: dict) -> Env:
        reject_unknown_keys(
            ds,
            {
                "path",
                "test_path",
                "paired_test_path",
                "seed",
                "eval_subset_size",
                "expected_eval_size",
                "timeout_seconds",
                "correct_reward",
                "format_reward",
                "prompt_file",
                "soft_token_budget",
                "overshoot_penalty",
                "min_cf_rating",
                "max_cf_rating",
                "verifier",
                "piston_url",
                "piston_python_version",
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
        verifier = ds.get("verifier", self.verifier)
        piston_url = ds.get("piston_url", self.piston_url)
        piston_python_version = ds.get(
            "piston_python_version", self.piston_python_version
        )
        _validate_verifier_settings(verifier, piston_url, piston_python_version)
        self.verifier = verifier
        self.piston_url = piston_url
        self.piston_python_version = piston_python_version
        self.grade_workers = (
            _MAX_CONCURRENT_PISTON_VERIFIERS
            if verifier == "piston"
            else TaskFamily.grade_workers
        )
        env = CodeContestsEnv(
            path=str(path),
            test_path=(str(ds["test_path"]) if ds.get("test_path") else None),
            paired_test_path=(
                str(ds["paired_test_path"]) if ds.get("paired_test_path") else None
            ),
            seed=int(ds.get("seed", 0)),
            eval_subset_size=int(ds.get("eval_subset_size", 128)),
            expected_eval_size=(
                int(ds["expected_eval_size"])
                if ds.get("expected_eval_size") is not None
                else None
            ),
            timeout_seconds=self.timeout_seconds,
            correct_reward=float(ds.get("correct_reward", 1.0)),
            format_reward=float(ds.get("format_reward", 0.1)),
            prompt_file=(str(ds["prompt_file"]) if ds.get("prompt_file") else None),
            soft_token_budget=(int(ds["soft_token_budget"]) if ds.get("soft_token_budget") else None),
            overshoot_penalty=float(ds.get("overshoot_penalty", 0.0)),
            min_cf_rating=(
                int(ds["min_cf_rating"]) if ds.get("min_cf_rating") is not None else None
            ),
            max_cf_rating=(
                int(ds["max_cf_rating"]) if ds.get("max_cf_rating") is not None else None
            ),
            verifier=self.verifier,
            piston_url=self.piston_url,
            piston_python_version=self.piston_python_version,
        )
        train_path = Path(str(path)).expanduser().resolve()
        if ds.get("paired_test_path"):
            eval_path = Path(str(ds["paired_test_path"])).expanduser().resolve()
            eval_source_kind = "paired_test_path"
        elif ds.get("test_path"):
            eval_path = Path(str(ds["test_path"])).expanduser().resolve()
            eval_source_kind = "test_path"
        else:
            eval_path = train_path
            eval_source_kind = "derived_from_train"
        prompt_path = resolve_prompt_file(ds.get("prompt_file"), PROMPT_FILE).resolve()
        train_manifest = _manifest_for(train_path)
        eval_manifest = _manifest_for(eval_path)
        self._protocol_identity = {
            "grading_protocol": "codecontests_fresh_process_per_case_v3",
            "verifier": env.verifier,
            "piston_protocol": (
                piston.PROTOCOL_ID if env.verifier == "piston" else "none"
            ),
            "piston_python_version": (
                env.piston_python_version if env.verifier == "piston" else "none"
            ),
            "train_source_path": str(train_path),
            "eval_source_path": str(eval_path),
            "eval_source_kind": eval_source_kind,
            "train_content_sha256": _sha256_file(train_path),
            "eval_content_sha256": _sha256_file(eval_path),
            "train_cohort_sha256": _sha256_json(env.train_rows),
            "eval_cohort_sha256": _sha256_json(env.test_rows),
            "train_suites_sha256": _sha256_json(_suite_projection(env.train_rows)),
            "eval_suites_sha256": _sha256_json(_suite_projection(env.test_rows)),
            "train_manifest_path": (
                str(train_manifest.resolve()) if train_manifest else "none"
            ),
            "train_manifest_sha256": (
                _sha256_file(train_manifest) if train_manifest else "none"
            ),
            "eval_manifest_path": (
                str(eval_manifest.resolve()) if eval_manifest else "none"
            ),
            "eval_manifest_sha256": (
                _sha256_file(eval_manifest) if eval_manifest else "none"
            ),
            "loader_filter_protocol": "codecontests_structural_v1",
            "seed": str(int(ds.get("seed", 0))),
            "eval_subset_size": str(int(ds.get("eval_subset_size", 128))),
            "expected_eval_size": _optional(
                int(ds["expected_eval_size"])
                if ds.get("expected_eval_size") is not None
                else None
            ),
            "timeout_seconds": str(self.timeout_seconds),
            # Resolved source-env values make semantically equivalent config
            # spellings canonical while pinning every reward-affecting knob.
            "correct_reward": repr(float(env.correct_reward)),
            "format_reward": repr(float(env.format_reward)),
            "soft_token_budget": _optional(env.soft_token_budget),
            "overshoot_penalty": repr(float(env.overshoot_penalty)),
            "min_cf_rating": _optional(
                int(ds["min_cf_rating"])
                if ds.get("min_cf_rating") is not None
                else None
            ),
            "max_cf_rating": _optional(
                int(ds["max_cf_rating"])
                if ds.get("max_cf_rating") is not None
                else None
            ),
            "prompt_file": str(prompt_path),
            "prompt_sha256": _sha256_file(prompt_path),
        }
        return env

    def parse_answers(self, text: str) -> AnswerParse:
        return parse_code_answers(text)

    def grade(self, meta: dict[str, Any], solution: Any) -> Optional[bool]:
        """GROUND-TRUTH LABEL, never a reward. DebateEnv calls this to record
        whether the proposer was actually right; the label goes to the
        transcript and to info["solution_correct"], and the judge never sees it.

        Uses truth_tests, which the RLVR arm was NOT trained on. Returns None
        when a problem has no truth suite (~66% of them — the <=10 sample
        consumed every case). Deliberately no fallback to rlvr_tests: on an
        RLVR-trained policy that would be scoring the training signal and
        would read as inflated accuracy. Those problems are covered by the
        cco_eval suite instead (see the module docstring)."""
        if solution is None:
            return None
        inputs, outputs = meta.get("truth_inputs"), meta.get("truth_outputs")
        if not inputs or not outputs:
            return None
        if is_cpp_code(solution):
            return False
        verifier_kwargs: dict[str, Any] = {}
        if self.verifier == "piston":
            verifier_kwargs = {
                "verifier": self.verifier,
                "piston_url": self.piston_url,
                "piston_python_version": self.piston_python_version,
            }
        result = run_stdin_tests(
            solution,
            inputs,
            outputs,
            timeout=self.timeout_seconds,
            **verifier_kwargs,
        )
        return bool(result["passed"])

    def protocol_identity(self) -> dict[str, str]:
        return dict(self._protocol_identity)
