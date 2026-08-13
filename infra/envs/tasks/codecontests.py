"""CodeContests task family: competitive programming problems verified by
running the candidate program against stdin/stdout test cases.

Ported from the old repo (ai_debate/data_loader/codecontests_loader.py and
ai_debate/experiments/verifiers/codecontests_verifier.py), slimmed: rows are
held in memory (no lazy-JSONL dict). Candidate programs execute in a separate
process; production Linux additionally requires a bubblewrap + seccomp
sandbox and fails closed when that boundary cannot be established.

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

No suite is ever rendered into a prompt. Models see the problem
statement only. _task() puts test cases in Task.meta and the prompt renderer
binds only meta["question"]. The suites stay in memory for grading but the
environment marks all eight fields artifact-private, so raw, Planned, Docent,
and wandb records cannot persist them.

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

run_stdin_tests() keeps expected outputs and verdict state in the trusted
parent only. Each case launches a fresh candidate interpreter and connects
only that case's stdin plus bounded stdout/stderr files. Candidate code never
shares an interpreter, stack, globals, argv, or writable result file with the
verifier, so introspection and monkeypatching cannot forge a verdict.

On Linux the child is fail-closed behind bubblewrap: private user/PID/network/
mount namespaces, uid/gid 65534, no capabilities, a read-only Python runtime,
and no host working tree. A required libseccomp filter independently rejects
child processes, cross-process signals, network syscalls, and filesystem
mutation while permitting one same-process worker thread. macOS keeps the
separate-process, empty-environment, rlimit path for local tests but is not a
production security boundary.

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

import base64
import binascii
import concurrent.futures
import functools
import hashlib
import importlib.util
import json
import logging
import os
import random
import re
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from infra.envs.base import Env, SingleTurnEnv, Task
from infra.envs.task_prompts import load_generation_prompts, resolve_prompt_file
from infra.envs.tasks.base import TaskFamily, reject_unknown_keys
from infra.envs.tasks.codecontests_sandbox import OUTPUT_LIMIT_BYTES

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
# The external service executes four requests at once. Match that active count
# here so cross-solution queue time is spent outside each solution's deadline,
# just like the local verifier semaphore. The service's additional bounded
# queue remains a transport-burst backstop, not routine candidate time.
_MAX_CONCURRENT_REMOTE_VERIFIERS = min(
    4,
    max(1, int(os.environ.get("MAX_CONCURRENT_REMOTE_VERIFIERS", "4"))),
)
_remote_verifier_semaphore = threading.Semaphore(
    _MAX_CONCURRENT_REMOTE_VERIFIERS
)

# These signed UNKNOWN results mean the isolated controller did not produce a
# usable candidate verdict and therefore can never be scored. One fresh-sandbox
# retry tolerates a rare monitor/ptrace or host-controller race while retaining
# a hard fail-closed boundary if the same code/input cannot complete twice.
# This is deliberately not configurable at launch time: raising it would turn
# a broken executor into an unbounded retry loop paid for by training.
_REMOTE_FRESH_SANDBOX_RETRY_CATEGORIES = frozenset(
    {"LAUNCH_ATTESTATION_MISSING", "CONTROLLER_EXCEPTION"}
)
_REMOTE_FRESH_SANDBOX_RETRIES = 1

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


def normalize_output(value: str) -> str:
    """Apply the historical whitespace/case/float CodeContests comparison."""
    normalized = []
    for line in value.split("\n"):
        line = line.rstrip()

        def normalize_float(match: re.Match) -> str:
            try:
                return f"{float(match.group(0)):g}"
            except (ValueError, OverflowError):
                return match.group(0)

        line = re.sub(
            r"-?\d+\.\d+(?:[eE][+-]?\d+)?", normalize_float, line
        )
        normalized.append(line)
    while normalized and not normalized[-1]:
        normalized.pop()
    return "\n".join(normalized).lower()


# ------------------------------------------------------------------ verifier


class VerifierInfrastructureError(RuntimeError):
    """Trusted verifier setup/state failed; the rollout is scientifically invalid."""


class SandboxUnavailable(VerifierInfrastructureError):
    """The production Linux execution boundary could not be established."""


_SANDBOX_BOOTSTRAP_PATH = os.path.join(
    os.path.dirname(__file__), "codecontests_sandbox.py"
)
_sandbox_probe_lock = threading.Lock()
_verified_bubblewrap_path: Optional[str] = None
_remote_executor_lock = threading.Lock()
_verified_remote_executor_client: Any | None = None
_EXECUTOR_MODE_ENV = "CODECONTESTS_EXECUTOR_MODE"
_REMOTE_ADDRESS_SPACE_BYTES = 4 * 1024 * 1024 * 1024
_SANDBOX_READY_PREFIX = "CODECONTESTS_SANDBOX_READY:"
_CANDIDATE_ENV = {
    "HOME": "/nonexistent",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PATH": "/usr/bin:/bin",
    # Candidate Python uses -s -P rather than -I: this is a trusted, fixed
    # search root in the cleared environment and exposes only curated contest
    # dependencies. The bootstrap itself still starts under -I -S.
    "PYTHONPATH": "/packages",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONIOENCODING": "utf-8",
}
_SANDBOX_PROBE = r'''import os
import socket
import sys
import threading

import mpmath
import sympy

if os.geteuid() == 0:
    raise SystemExit("still privileged")

try:
    with open("/tmp/codecontests-write-probe", "w") as f:
        f.write("mutation escaped")
except OSError:
    pass
else:
    raise SystemExit("filesystem mutation was allowed")

try:
    child = os.fork()
except OSError:
    pass
else:
    if child == 0:
        os._exit(0)
    os.waitpid(child, 0)
    raise SystemExit("process creation was allowed")

# Exactly one same-process worker is the competitive-programming recursion
# idiom. A second concurrent worker must hit RLIMIT_NPROC=3 (the private PID
# namespace also has bubblewrap's init/reaper process).
release = threading.Event()
started = threading.Event()
worker = threading.Thread(target=lambda: (started.set(), release.wait()))
worker.start()
if not started.wait(2):
    raise SystemExit("single worker thread did not start")
try:
    extra = threading.Thread(target=lambda: None)
    extra.start()
except RuntimeError:
    pass
else:
    extra.join()
    raise SystemExit("second concurrent worker thread was allowed")
release.set()
worker.join()

try:
    socket.socket(socket.AF_INET, socket.SOCK_STREAM)
except OSError:
    pass
else:
    raise SystemExit("network socket creation was allowed")

try:
    os.kill(1, 0)
except OSError:
    pass
else:
    raise SystemExit("process signalling was allowed")

if os.path.exists("/root") or os.path.exists("/workspace"):
    raise SystemExit("host-private filesystem paths are visible")
if "/packages" not in sys.path:
    raise SystemExit("curated package root is absent from sys.path")
if not (os.statvfs("/packages").f_flag & os.ST_RDONLY):
    raise SystemExit("curated package root is writable")

symbol = sympy.Symbol("x")
if str(sympy.factor(symbol**2 - 1)) != "(x - 1)*(x + 1)":
    raise SystemExit("curated sympy dependency is not functional")
if mpmath.sqrt(4) != 2:
    raise SystemExit("curated mpmath dependency is not functional")

print("CODECONTESTS_SANDBOX_OK")
'''


def _python_runtime_mount() -> tuple[str, str]:
    """Return the base runtime tree and interpreter path inside the sandbox."""
    base_prefix = os.path.realpath(sys.base_prefix)
    base_executable = os.path.realpath(
        getattr(sys, "_base_executable", None) or sys.executable
    )
    try:
        if os.path.commonpath((base_prefix, base_executable)) != base_prefix:
            raise ValueError
        relative_executable = os.path.relpath(base_executable, base_prefix)
    except ValueError as exc:
        raise SandboxUnavailable(
            f"Python base executable {base_executable!r} is outside base prefix "
            f"{base_prefix!r}"
        ) from exc
    return base_prefix, "/runtime/" + relative_executable


@functools.lru_cache(maxsize=1)
def _curated_package_mounts() -> tuple[tuple[str, str], ...]:
    """Expose only historical contest dependencies, never the whole venv."""
    mounts: list[tuple[str, str]] = []
    for package in ("sympy", "mpmath"):
        spec = importlib.util.find_spec(package)
        locations = list(spec.submodule_search_locations or []) if spec else []
        if len(locations) != 1 or not os.path.isdir(locations[0]):
            raise SandboxUnavailable(
                f"curated CodeContests dependency {package!r} is missing from the training venv"
            )
        source = os.path.realpath(locations[0])
        mounts.append((source, f"/packages/{package}"))

        # Preserve importlib.metadata behavior without exposing unrelated
        # venv packages. Distribution metadata is read-only and tiny.
        parent = Path(source).parent
        for metadata in sorted(parent.glob(f"{package.replace('_', '-')}-*.dist-info")):
            mounts.append((str(metadata.resolve()), f"/packages/{metadata.name}"))
    return tuple(mounts)


def _bubblewrap_command(
    bubblewrap: str,
    solution_path: str,
    ready_token: str,
    bootstrap_path: str = _SANDBOX_BOOTSTRAP_PATH,
) -> list[str]:
    runtime_source, runtime_executable = _python_runtime_mount()
    command = [
        bubblewrap,
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--uid", "65534",
        "--gid", "65534",
        "--cap-drop", "ALL",
        "--clearenv",
    ]
    for key, value in _CANDIDATE_ENV.items():
        command.extend(("--setenv", key, value))

    # The OS runtime and UV-managed Python runtime are the only host trees
    # visible (plus the read-only dynamic-linker cache below). /root,
    # /workspace, the repo, datasets, credentials, and verifier tempdir never
    # enter the namespace.
    for source in ("/usr", "/bin", "/lib", "/lib64"):
        if not os.path.lexists(source):
            continue
        if os.path.islink(source):
            command.extend(("--symlink", os.readlink(source), source))
        else:
            command.extend(("--ro-bind", source, source))
    command.extend(("--dir", "/etc"))
    if os.path.isfile("/etc/ld.so.cache"):
        command.extend(
            ("--ro-bind", "/etc/ld.so.cache", "/etc/ld.so.cache")
        )
    command.extend(
        (
            "--ro-bind", runtime_source, "/runtime",
            # Keep venv-only dependencies out of the runtime tree. A separate
            # staging root lets bwrap create child bind targets without
            # writing into the read-only /runtime parent.
            "--tmpfs", "/packages",
        )
    )
    # Curated dependency binds must follow creation of their independent
    # /packages tmpfs. Bubblewrap creates each package mountpoint inside the
    # namespace and exposes only these two venv packages, then the whole
    # search root is remounted read-only.
    for source, destination in _curated_package_mounts():
        command.extend(("--ro-bind", source, destination))
    command.extend(
        (
            "--remount-ro", "/packages",
            "--ro-bind", bootstrap_path, "/sandbox_bootstrap.py",
            "--ro-bind", solution_path, "/solution.py",
            "--dev", "/dev",
            "--dir", "/tmp",
            "--chdir", "/tmp",
            "--",
            runtime_executable, "-I", "-S",
            "/sandbox_bootstrap.py", "/solution.py", ready_token,
        )
    )
    return command


def _verified_bubblewrap() -> str:
    """Return a proven bubblewrap binary or raise before candidate execution."""
    global _verified_bubblewrap_path
    if _verified_bubblewrap_path is not None:
        return _verified_bubblewrap_path
    # functools.lru_cache is thread-safe but may execute a first cache miss
    # more than once concurrently. The first rollout grades up to 32 samples
    # in parallel, so use a double-checked lock and run exactly one probe.
    with _sandbox_probe_lock:
        if _verified_bubblewrap_path is not None:
            return _verified_bubblewrap_path
        _verified_bubblewrap_path = _probe_bubblewrap()
        return _verified_bubblewrap_path


def _probe_bubblewrap() -> str:
    bubblewrap = shutil.which("bwrap")
    if not bubblewrap:
        raise SandboxUnavailable(
            "bwrap is required for CodeContests execution on Linux"
        )
    if not os.path.isfile(_SANDBOX_BOOTSTRAP_PATH):
        raise SandboxUnavailable(
            f"sandbox bootstrap is missing: {_SANDBOX_BOOTSTRAP_PATH}"
        )

    with tempfile.TemporaryDirectory(prefix="codecontests_sandbox_probe_") as tmpdir:
        probe_path = os.path.join(tmpdir, "probe.py")
        with open(probe_path, "w") as f:
            f.write(_SANDBOX_PROBE)
        ready_token = secrets.token_hex(16)
        command = _bubblewrap_command(bubblewrap, probe_path, ready_token)
        try:
            probe = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=_CANDIDATE_ENV,
                text=True,
                timeout=15,
                start_new_session=True,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SandboxUnavailable(f"bubblewrap feasibility probe failed: {exc}") from exc
    ready_marker = f"{_SANDBOX_READY_PREFIX}{ready_token}"
    if (
        probe.returncode != 0
        or probe.stdout.strip() != "CODECONTESTS_SANDBOX_OK"
        or ready_marker not in probe.stderr.splitlines()
    ):
        detail = (probe.stderr or probe.stdout or "no diagnostic").strip()[:1000]
        raise SandboxUnavailable(
            f"bubblewrap/libseccomp feasibility probe failed (rc={probe.returncode}): {detail}"
        )
    return bubblewrap


def _reset_sandbox_probe_for_tests() -> None:
    global _verified_bubblewrap_path, _verified_remote_executor_client
    with _sandbox_probe_lock:
        _verified_bubblewrap_path = None
        _curated_package_mounts.cache_clear()
    with _remote_executor_lock:
        _verified_remote_executor_client = None


def _executor_mode() -> str:
    """Resolve the execution boundary without permitting an implicit fallback."""
    mode = os.environ.get(_EXECUTOR_MODE_ENV, "").strip().lower()
    if mode in {"", "local"}:
        return "local"
    if mode == "remote":
        return mode
    raise VerifierInfrastructureError(
        f"invalid {_EXECUTOR_MODE_ENV}={mode!r}; expected local or remote"
    )


def _new_remote_executor_client() -> Any:
    # Imported lazily so ordinary local development retains the exact existing
    # bubblewrap/macOS dependency surface.
    from codecontests_executor.client import RemoteExecutorClient

    return RemoteExecutorClient.from_env(verifier_path=__file__)


def _verified_remote_executor() -> Any:
    """Return a client whose signed frozen identity was verified in this process."""
    global _verified_remote_executor_client
    if _verified_remote_executor_client is not None:
        return _verified_remote_executor_client
    with _remote_executor_lock:
        if _verified_remote_executor_client is not None:
            return _verified_remote_executor_client
        try:
            client = _new_remote_executor_client()
            client.verify_identity()
        except Exception as exc:
            raise VerifierInfrastructureError(
                "remote CodeContests executor identity/configuration preflight "
                f"failed: {type(exc).__name__}"
            ) from exc
        _verified_remote_executor_client = client
        return client


def verify_code_execution_sandbox() -> None:
    """Cheap launch preflight; neither boundary ever falls back to the other."""
    if _executor_mode() == "remote":
        _verified_remote_executor()
    elif sys.platform == "linux":
        _verified_bubblewrap()


def _candidate_command(solution_path: str) -> tuple[list[str], str]:
    ready_token = secrets.token_hex(16)
    if sys.platform == "linux":
        command = _bubblewrap_command(
            _verified_bubblewrap(), solution_path, ready_token
        )
        return command, ready_token
    # Local macOS test path: still a fresh interpreter with a blank environment
    # and irreversible rlimits, but without Linux namespace/seccomp claims.
    return (
        [
            sys.executable,
            "-I",
            "-S",
            _SANDBOX_BOOTSTRAP_PATH,
            solution_path,
            ready_token,
        ],
        ready_token,
    )


def _read_bounded_output(path: str) -> tuple[str, bool]:
    try:
        with open(path, "rb") as file_obj:
            raw = file_obj.read(OUTPUT_LIMIT_BYTES + 1)
    finally:
        # Captures are transport scratch, not run artifacts. Remove each one
        # as soon as its bounded contents have entered trusted memory rather
        # than retaining up to 4 MiB/case until the whole suite finishes.
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
    text = raw[:OUTPUT_LIMIT_BYTES].decode("utf-8", errors="replace")
    # RLIMIT_FSIZE stops the regular capture file at exactly this size, so
    # equality is the observable saturation signal.
    return text, len(raw) >= OUTPUT_LIMIT_BYTES


def _kill_process_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, OSError):
        try:
            proc.kill()
        except ProcessLookupError:
            pass


def _remote_failure_diagnostic(execution: Any) -> str:
    stderr = bytes(getattr(execution, "stderr", b"")).decode(
        "utf-8", errors="replace"
    )
    category = getattr(execution, "category", None) or "UNCLASSIFIED"
    diagnostic = f"remote candidate failure: {category}"
    if stderr:
        diagnostic += f". {stderr[:500]}"
    return diagnostic


def _remote_output_saturation(execution: Any) -> tuple[bool, bool]:
    """Read the already-validated, signed output saturation evidence."""
    payload = getattr(execution, "result_payload", None)
    evidence = payload.get("evidence") if isinstance(payload, dict) else None
    if not isinstance(evidence, dict):
        raise VerifierInfrastructureError(
            "remote candidate failure lacks signed output saturation evidence"
        )
    stdout_truncated = evidence.get("stdout_truncated")
    stderr_truncated = evidence.get("stderr_truncated")
    if not isinstance(stdout_truncated, bool) or not isinstance(
        stderr_truncated, bool
    ):
        raise VerifierInfrastructureError(
            "remote candidate failure has invalid signed output saturation evidence"
        )
    return stdout_truncated, stderr_truncated


def _remote_monitor_error(stderr: bytes) -> str | None:
    """Extract the bounded monitor exception class from signed stderr.

    The response envelope and its stderr bytes have already been authenticated
    and schema-validated by ``RemoteExecutorClient``.  This parser is only for
    operator diagnostics; failure to decode it never changes retry or reward
    semantics.
    """
    marker = b"PALAESTRA_EXECUTOR_STATUS:"
    offset = stderr.rfind(marker)
    if offset < 0:
        return None
    frame = stderr[offset + len(marker) :].split(b"\n", 1)[0]
    _nonce, separator, encoded = frame.partition(b":")
    if not separator or not encoded:
        return None
    try:
        status = json.loads(
            base64.b64decode(encoded, validate=True).decode(
                "ascii", errors="strict"
            )
        )
    except (binascii.Error, UnicodeError, ValueError, json.JSONDecodeError):
        return None
    if (
        not isinstance(status, dict)
        or status.get("candidate_ready_attested") is not False
    ):
        return None
    error = status.get("error")
    if (
        not isinstance(error, str)
        or not error.isascii()
        or re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", error) is None
    ):
        return None
    return error


def _remote_unknown_diagnostic(
    execution: Any,
    *,
    solution_code: str,
    test_input: str,
) -> str:
    """Return a bounded, replay-oriented fingerprint for a signed UNKNOWN."""
    stderr = bytes(getattr(execution, "stderr", b""))
    payload = getattr(execution, "result_payload", None)
    evidence = payload.get("evidence") if isinstance(payload, dict) else None
    timing = payload.get("timing") if isinstance(payload, dict) else None
    fields = [
        f"code_sha256={hashlib.sha256(solution_code.encode()).hexdigest()}",
        f"stdin_sha256={hashlib.sha256(test_input.encode()).hexdigest()}",
        f"stderr_sha256={hashlib.sha256(stderr).hexdigest()}",
    ]
    category = getattr(execution, "category", None)
    if isinstance(category, str):
        fields.append(f"category={category}")
    controller_error = getattr(execution, "error", None)
    if isinstance(controller_error, str):
        fields.append(f"controller_error={controller_error}")
    monitor_error = _remote_monitor_error(stderr)
    if monitor_error is not None:
        fields.append(f"monitor_error={monitor_error}")
    if isinstance(evidence, dict):
        for key in (
            "returncode",
            "host_cpu_usage_us",
            "host_memory_peak_bytes",
            "host_pids_peak",
        ):
            if key not in evidence:
                continue
            value = evidence.get(key)
            if value is None or (isinstance(value, int) and not isinstance(value, bool)):
                fields.append(f"{key}={value}")
    if isinstance(timing, dict):
        execution_ns = timing.get("execution_ns")
        if isinstance(execution_ns, int) and not isinstance(execution_ns, bool):
            fields.append(f"execution_ns={execution_ns}")
    return " ".join(fields)


def _run_stdin_tests_remote(
    solution_code: str,
    inputs: list[str],
    outputs: list[str],
    *,
    timeout: int,
    t0: float,
) -> dict[str, Any]:
    """Run one attested, fresh gVisor execution per input.

    Expected outputs remain in this trusted process. The remote boundary sees
    only candidate source, one stdin value, and resource limits.
    """
    _remote_verifier_semaphore.acquire()
    try:
        # This verifies the signed, frozen service identity before even a
        # syntactically invalid candidate can be treated as a measured result.
        client = _verified_remote_executor()
        try:
            compile(solution_code, "<remote-codecontests-solution>", "exec")
        except SyntaxError as exc:
            return _failure_result(
                "error",
                0,
                f"SyntaxError: {exc}",
                t0,
                candidate_error=True,
            )

        # Match the local verifier's per-solution budget: setup and identity
        # preflight are outside it, while every case spends from one deadline.
        deadline = time.perf_counter() + timeout
        tests_passed = 0
        first_failure = None
        candidate_error = False
        infrastructure_retries = 0
        for i, (test_input, expected_output) in enumerate(zip(inputs, outputs)):
            infrastructure_diagnostics: list[str] = []
            for infrastructure_attempt in range(_REMOTE_FRESH_SANDBOX_RETRIES + 1):
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    if infrastructure_diagnostics:
                        raise VerifierInfrastructureError(
                            "remote CodeContests executor fresh-sandbox retry "
                            f"exhausted the solution deadline on case {i}: "
                            + " | ".join(infrastructure_diagnostics)
                        )
                    return _failure_result(
                        "timeout",
                        len(inputs),
                        "Total execution timed out.",
                        t0,
                        timeout=True,
                    )

                # Preserve the one shared solution deadline across RPCs. The
                # v2 executor accepts nanoseconds, so conservatively floor this
                # clock sample instead of rounding a fractional remainder up.
                # A retry gets the newly remaining candidate budget, never a
                # fresh solution timeout. A signed unscorable infrastructure
                # failure may refund only its own RPC wall time below.
                remaining_ns = int(remaining * 1_000_000_000)
                if remaining_ns <= 0:
                    if infrastructure_diagnostics:
                        raise VerifierInfrastructureError(
                            "remote CodeContests executor fresh-sandbox retry "
                            f"exhausted the solution deadline on case {i}: "
                            + " | ".join(infrastructure_diagnostics)
                        )
                    return _failure_result(
                        "timeout",
                        len(inputs),
                        "Total execution timed out.",
                        t0,
                        timeout=True,
                    )
                remaining_seconds, remaining_nanos = divmod(
                    remaining_ns, 1_000_000_000
                )
                raw_limits = {
                    "time_limit": {
                        "seconds": remaining_seconds,
                        "nanos": remaining_nanos,
                    },
                    "memory_limit_bytes": _REMOTE_ADDRESS_SPACE_BYTES,
                }
                execution_started_at = time.perf_counter()
                try:
                    execution = client.execute(
                        code=solution_code,
                        stdin=test_input,
                        raw_limits=raw_limits,
                    )
                except Exception as exc:
                    raise VerifierInfrastructureError(
                        "remote CodeContests executor call failed on case "
                        f"{i}: {type(exc).__name__}"
                    ) from exc
                execution_elapsed = max(
                    0.0, time.perf_counter() - execution_started_at
                )

                outcome = getattr(execution, "outcome", None)
                category = getattr(execution, "category", None)
                if (
                    outcome == "unknown"
                    and category in _REMOTE_FRESH_SANDBOX_RETRY_CATEGORIES
                ):
                    diagnostic = _remote_unknown_diagnostic(
                        execution,
                        solution_code=solution_code,
                        test_input=test_input,
                    )
                    infrastructure_diagnostics.append(diagnostic)
                    if infrastructure_attempt < _REMOTE_FRESH_SANDBOX_RETRIES:
                        # This signed infrastructure failure did not yield a
                        # usable attested candidate result. Restore only the
                        # wall time spent inside this failed RPC; earlier cases
                        # and local retry overhead continue spending from the
                        # same solution deadline.
                        deadline += execution_elapsed
                        infrastructure_retries += 1
                        logger.warning(
                            "retrying signed remote executor infrastructure "
                            "failure with a fresh sandbox: case=%d attempt=%d/%d %s",
                            i,
                            infrastructure_attempt + 1,
                            _REMOTE_FRESH_SANDBOX_RETRIES + 1,
                            diagnostic,
                        )
                        continue
                if outcome == "unknown":
                    error = getattr(execution, "error", None)
                    detail = f" ({error})" if error else ""
                    diagnostics = (
                        " diagnostics=" + " | ".join(infrastructure_diagnostics)
                        if infrastructure_diagnostics
                        else ""
                    )
                    raise VerifierInfrastructureError(
                        "remote CodeContests executor returned UNKNOWN on case "
                        f"{i}: {category or 'UNCLASSIFIED'}{detail}{diagnostics}"
                    )
                break
            if outcome not in {"executed", "candidate_failure"}:
                raise VerifierInfrastructureError(
                    "remote CodeContests executor returned an invalid outcome "
                    f"on case {i}: {outcome!r}"
                )

            # A response completing after the shared deadline is still a
            # candidate timeout, even if its signed per-case outcome says it
            # exited normally. No later input is submitted. The remote service
            # policy must independently kill the case at this same deadline.
            if time.perf_counter() >= deadline:
                return _failure_result(
                    "timeout",
                    len(inputs),
                    "Total execution timed out.",
                    t0,
                    timeout=True,
                )

            if outcome == "candidate_failure":
                if category in {"CPU_LIMIT", "WALL_LIMIT"}:
                    return _failure_result(
                        "timeout",
                        len(inputs),
                        _remote_failure_diagnostic(execution),
                        t0,
                        timeout=True,
                    )
                candidate_error = True
                stdout_truncated, stderr_truncated = _remote_output_saturation(
                    execution
                )
                actual = bytes(getattr(execution, "stdout", b"")).decode(
                    "utf-8", errors="replace"
                )
                normalized_actual = normalize_output(actual)
                normalized_expected = normalize_output(expected_output)
                # Historical local semantics record the candidate error but
                # still award this case when its bounded stdout is correct.
                # This includes a later nonzero exit, file/process limit, or a
                # stderr-only output saturation. Saturated stdout never passes.
                if not stdout_truncated and normalized_actual == normalized_expected:
                    tests_passed += 1
                    continue
                if first_failure is None:
                    diagnostic = _remote_failure_diagnostic(execution)
                    if stdout_truncated:
                        diagnostic = "stdout exceeded the 2 MiB limit. " + diagnostic
                    if stderr_truncated:
                        diagnostic = diagnostic[:450] + " [stderr truncated]"
                    first_failure = {
                        "test_idx": i,
                        "expected": normalized_expected[:500],
                        "actual": normalized_actual[:500],
                        "stderr": diagnostic,
                    }
                continue

            actual = bytes(getattr(execution, "stdout", b"")).decode(
                "utf-8", errors="replace"
            )
            normalized_actual = normalize_output(actual)
            normalized_expected = normalize_output(expected_output)
            if normalized_actual == normalized_expected:
                tests_passed += 1
                continue
            if first_failure is None:
                stderr = bytes(getattr(execution, "stderr", b"")).decode(
                    "utf-8", errors="replace"
                )
                first_failure = {
                    "test_idx": i,
                    "expected": normalized_expected[:500],
                    "actual": normalized_actual[:500],
                    "stderr": stderr[:500],
                }

        passed = tests_passed == len(inputs)
        return {
            "status": "passed" if passed else "failed",
            "passed": passed,
            "tests_passed": tests_passed,
            "tests_total": len(inputs),
            "timeout": False,
            "candidate_error": candidate_error,
            "infrastructure_retries": infrastructure_retries,
            "first_failure": first_failure,
            "execution_time_seconds": time.perf_counter() - t0,
        }
    except VerifierInfrastructureError:
        raise
    except Exception as exc:
        raise VerifierInfrastructureError(
            f"trusted remote CodeContests verifier failed: {type(exc).__name__}"
        ) from exc
    finally:
        _remote_verifier_semaphore.release()


def run_stdin_tests(
    solution_code: str,
    inputs: list[str],
    outputs: list[str],
    timeout: int = 90,
) -> dict[str, Any]:
    """Run cases externally; expected outputs never cross the trust boundary."""
    t0 = time.perf_counter()
    if len(inputs) != len(outputs):
        raise VerifierInfrastructureError(
            "trusted CodeContests suite is ragged: "
            f"{len(inputs)} inputs != {len(outputs)} outputs"
        )
    if _executor_mode() == "remote":
        return _run_stdin_tests_remote(
            solution_code,
            inputs,
            outputs,
            timeout=timeout,
            t0=t0,
        )
    try:
        tmpdir = tempfile.mkdtemp(prefix="codecontests_test_")
    except OSError as exc:
        raise VerifierInfrastructureError(
            f"could not create verifier temporary directory: {exc}"
        ) from exc

    _verifier_semaphore.acquire()
    try:
        solution_path = os.path.join(tmpdir, "solution.py")
        try:
            compile(solution_code, solution_path, "exec")
        except SyntaxError as exc:
            return _failure_result(
                "error", 0, f"SyntaxError: {exc}", t0,
                candidate_error=True,
            )
        try:
            with open(solution_path, "w") as f:
                f.write(solution_code)
        except OSError as exc:
            raise VerifierInfrastructureError(
                f"could not stage candidate source: {exc}"
            ) from exc

        try:
            command, ready_token = _candidate_command(solution_path)
        except VerifierInfrastructureError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise VerifierInfrastructureError(
                f"could not construct candidate sandbox command: {exc}"
            ) from exc

        # The one-time namespace/seccomp feasibility probe is infrastructure
        # preflight, not candidate execution time. Start the historical total
        # per-solution timeout only after the sandbox command is ready.
        deadline = time.perf_counter() + timeout
        tests_passed = 0
        first_failure = None
        candidate_error = False
        for i, (test_input, expected_output) in enumerate(zip(inputs, outputs)):
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return _failure_result(
                    "timeout", len(inputs), "Total execution timed out.",
                    t0, timeout=True,
                )
            try:
                # Regular temporary files plus RLIMIT_FSIZE keep adversarial
                # stdout/stderr bounded. PIPE + communicate would buffer until
                # exhaustion before the parent got a chance to intervene. The
                # child gets write-only, append-only descriptors: it cannot
                # read or overwrite bootstrap's authenticated ready marker.
                stdout_path = os.path.join(tmpdir, f"stdout_{i}.capture")
                stderr_path = os.path.join(tmpdir, f"stderr_{i}.capture")
                stdout_fd = os.open(
                    stdout_path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND,
                    0o600,
                )
                try:
                    stderr_fd = os.open(
                        stderr_path,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND,
                        0o600,
                    )
                except Exception:
                    os.close(stdout_fd)
                    raise
                with (
                    os.fdopen(stdout_fd, "wb") as stdout_file,
                    os.fdopen(stderr_fd, "wb") as stderr_file,
                ):
                    proc = subprocess.Popen(
                        command,
                        stdin=subprocess.PIPE,
                        stdout=stdout_file,
                        stderr=stderr_file,
                        env=_CANDIDATE_ENV,
                        start_new_session=True,
                        close_fds=True,
                    )
                    try:
                        proc.communicate(
                            input=test_input.encode("utf-8"), timeout=remaining
                        )
                    except subprocess.TimeoutExpired:
                        _kill_process_group(proc)
                        proc.communicate()
                        return _failure_result(
                            "timeout", len(inputs), "Total execution timed out.",
                            t0, timeout=True,
                        )
                actual, stdout_truncated = _read_bounded_output(stdout_path)
                stderr, stderr_truncated = _read_bounded_output(stderr_path)
            except VerifierInfrastructureError:
                raise
            except Exception as exc:  # noqa: BLE001
                proc_obj = locals().get("proc")
                if isinstance(proc_obj, subprocess.Popen) and proc_obj.poll() is None:
                    _kill_process_group(proc_obj)
                    proc_obj.wait()
                raise VerifierInfrastructureError(
                    f"candidate transport failed on case {i}: {exc!r}"
                ) from exc

            # The readiness token is generated in the trusted parent, passed
            # only to bootstrap, and erased from argv by exec. Candidate text
            # therefore cannot masquerade as a sandbox failure or suppress a
            # real one by printing a familiar-looking traceback.
            ready_marker = f"{_SANDBOX_READY_PREFIX}{ready_token}\n"
            if ready_marker not in stderr:
                raise VerifierInfrastructureError(
                    "candidate sandbox/bootstrap did not emit its authenticated "
                    f"readiness marker on case {i}: {stderr[:1000]}"
                )
            stderr = stderr.replace(ready_marker, "", 1)
            candidate_error = candidate_error or proc.returncode != 0
            candidate_error = candidate_error or stdout_truncated or stderr_truncated

            normalized_actual = normalize_output(actual)
            normalized_expected = normalize_output(expected_output)
            if not stdout_truncated and normalized_actual == normalized_expected:
                tests_passed += 1
                continue

            if first_failure is None:
                diagnostic = stderr[:500]
                if stdout_truncated:
                    diagnostic = "stdout exceeded the 2 MiB limit. " + diagnostic
                if stderr_truncated:
                    diagnostic = diagnostic[:450] + " [stderr truncated]"
                first_failure = {
                    "test_idx": i,
                    "expected": normalized_expected[:500],
                    "actual": normalized_actual[:500],
                    "stderr": diagnostic,
                }

        passed = tests_passed == len(inputs)
        return {
            "status": "passed" if passed else "failed",
            "passed": passed,
            "tests_passed": tests_passed,
            "tests_total": len(inputs),
            "timeout": False,
            "candidate_error": candidate_error,
            "first_failure": first_failure,
            "execution_time_seconds": time.perf_counter() - t0,
        }
    except VerifierInfrastructureError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise VerifierInfrastructureError(
            f"trusted CodeContests verifier failed: {exc!r}"
        ) from exc
    finally:
        _verifier_semaphore.release()
        shutil.rmtree(tmpdir, ignore_errors=True)


def _failure_result(
    status: str,
    tests_total: int,
    stderr: str,
    t0: float,
    timeout: bool = False,
    candidate_error: bool = False,
) -> dict[str, Any]:
    return {
        "status": status,
        "passed": False,
        "tests_passed": 0,
        "tests_total": tests_total,
        "timeout": timeout,
        "candidate_error": candidate_error,
        "first_failure": {"test_idx": 0, "expected": "", "actual": "", "stderr": stderr},
        "execution_time_seconds": time.perf_counter() - t0,
    }


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
    # Suites must stay live on Task.meta for reward()/grade(), but may never
    # enter generic RLVR transcript, wandb, raw-fallback, or Docent records.
    artifact_private_meta_keys = SingleTurnEnv.artifact_private_meta_keys | frozenset(
        {
            "rlvr_inputs", "rlvr_outputs",
            "truth_inputs", "truth_outputs",
            "gdm_inputs", "gdm_outputs",
            "cco_inputs", "cco_outputs",
        }
    )

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
    ):
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
        # not public examples: nothing renders them, and the environment's
        # artifact-private key set redacts them from every transcript exporter.
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

    def reward(self, task: Task, text: str) -> tuple[float, dict[str, Any]]:
        """GDM-only RLVR reward plus paired held-out measurement.

        Training tasks use ``rlvr_inputs``. Paired held-out tasks use the
        explicitly named ``gdm_inputs`` and additionally run the SAME extracted
        program through CCO. CCO metrics never enter the returned reward.
        """
        code = extract_code(text, relaxed=True)
        # every key present in EVERY branch, so eval-time averages are means
        # over all samples rather than over the branch that happened to set it
        info: dict[str, Any] = {
            "correct": 0.0,
            "has_code": float(code is not None),
            "tests_passed_frac": 0.0,
            "gdm_correct": 0.0,
            "gdm_tests_passed_frac": 0.0,
            "cpp_code": 0.0,
            "exec_timeout": 0.0,
            "exec_error": 0.0,
            "exec_infrastructure_retries": 0.0,
        }
        has_cco = bool(task.meta.get("cco_inputs"))
        if has_cco:
            info["cco_correct"] = 0.0
            info["cco_tests_passed_frac"] = 0.0
            info["cco_exec_timeout"] = 0.0
            info["cco_exec_error"] = 0.0
            info["cco_exec_infrastructure_retries"] = 0.0
        if code is None:
            return 0.0, info
        if is_cpp_code(code):
            info["cpp_code"] = 1.0
            return self.format_reward, info

        if has_cco:
            cco = run_stdin_tests(
                code, task.meta["cco_inputs"], task.meta["cco_outputs"],
                timeout=self.timeout_seconds,
            )
            cco_total = cco.get("tests_total") or len(task.meta["cco_inputs"])
            info["cco_correct"] = float(bool(cco.get("passed")))
            info["cco_tests_passed_frac"] = (
                cco.get("tests_passed", 0) / cco_total if cco_total else 0.0
            )
            info["cco_exec_timeout"] = float(bool(cco.get("timeout")))
            info["cco_exec_error"] = float(bool(cco.get("candidate_error")))
            info["cco_exec_infrastructure_retries"] = float(
                cco.get("infrastructure_retries", 0)
            )

        gdm_inputs = task.meta.get("gdm_inputs") or task.meta["rlvr_inputs"]
        gdm_outputs = task.meta.get("gdm_outputs") or task.meta["rlvr_outputs"]
        result = run_stdin_tests(
            code, gdm_inputs, gdm_outputs,
            timeout=self.timeout_seconds,
        )
        total = result.get("tests_total") or len(gdm_inputs)
        gdm_correct = float(bool(result.get("passed")))
        gdm_passed_frac = result.get("tests_passed", 0) / total if total else 0.0
        info["gdm_correct"] = info["correct"] = gdm_correct
        info["gdm_tests_passed_frac"] = info["tests_passed_frac"] = gdm_passed_frac
        # Infrastructure failures raise and invalidate the rollout. These
        # metrics therefore describe only candidate behavior.
        info["exec_timeout"] = float(bool(result.get("timeout")))
        info["exec_error"] = float(bool(result.get("candidate_error")))
        info["exec_infrastructure_retries"] = float(
            result.get("infrastructure_retries", 0)
        )
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
        would read as inflated accuracy. Those problems are covered by the
        cco_eval suite instead (see the module docstring)."""
        if solution is None:
            return None
        inputs, outputs = meta.get("truth_inputs"), meta.get("truth_outputs")
        if not inputs or not outputs:
            return None
        if is_cpp_code(solution):
            return False
        return bool(run_stdin_tests(solution, inputs, outputs, timeout=self.timeout_seconds).get("passed"))

    def grade_batch(
        self, items: list[tuple[dict[str, Any], Any]]
    ) -> list[Optional[bool]]:
        """Grade concurrently while keeping infrastructure failures fatal.

        The generic family contract intentionally turns arbitrary per-item
        grader exceptions into ``None``. A missing/broken code-execution
        boundary is different: continuing would publish an unlabeled debate
        as if it were a valid evaluation, so this family propagates that
        specific exception and retains the generic behavior for other bugs.
        """
        self.last_grade_errors = 0
        if not items:
            return []

        def _one(
            item: tuple[dict[str, Any], Any]
        ) -> tuple[Optional[bool], bool]:
            try:
                return self.grade(*item), False
            except VerifierInfrastructureError:
                raise
            except Exception:  # noqa: BLE001
                return None, True

        if len(items) == 1 or self.grade_workers <= 1:
            scored = [_one(item) for item in items]
        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.grade_workers
            ) as pool:
                scored = list(pool.map(_one, items))
        self.last_grade_errors = sum(error for _, error in scored)
        return [grade for grade, _ in scored]

    def format_flags(self, text: str) -> dict[str, float]:
        # strict flag: a properly CLOSED fence, independent of the (possibly
        # relaxed) extractor used for position binding
        return {"code_fence": float(has_closed_fence(text))}
