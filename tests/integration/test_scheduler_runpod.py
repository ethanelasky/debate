"""Opt-in RunPod integration checks for the deterministic scheduler.

Ordinary pytest runs skip both tests before the scheduler package is imported
and before any provider call is made: the import is lazy, inside each enabled
test.  ``RUNPOD_INTEGRATION=1`` enables one authenticated, zero-mutation
preflight.  The cost-incurring lifecycle smoke is hard-disabled: no environment
variable enables it, and ``RUNPOD_PAID_INTEGRATION`` has no effect on either
test.  Its body remains only as an obsolete harness sketch: it lacks the
plan-required price and storage checks, a certified provider-enforced
``stopAfter`` attached to CREATE through the unresolved replacement seam, and
the actual CLI/socket/service/SQLite path.  The preflight requires exact
inventory identity and status equality before and after its check.

The expected production surface is deliberately narrow and neutral::

    harness = tools.scheduler.runpod.create_integration_harness(
        profile=..., artifact_root=..., command_runner=..., timeout_seconds=...,
    )
    harness.inventory()   -> sequence of mappings with id/name/status
    harness.preflight()   -> mapping; performs no mutation
    harness.run_owned_smoke(
        name=..., ttl_minutes=..., command=..., input_paths=...,
        output_paths=..., verify=..., timeout_seconds=...,
    ) -> mapping with worker_id / exit_code / artifacts

``artifacts`` maps each requested output path, plus ``stdout``, ``stderr`` and
``terminal``, to a durable file under the artifact root; ``terminal`` holds the
remote exit status as decimal bytes, so success is proven from a durable
artifact, not from the returned ``exit_code`` alone.  The harness owns
provider, transfer and remote composition: this module names no concrete
adapter class, no receipt schema, no private on-disk layout and no local
wrapper CLI (in particular it does not assume ``runpod-safe`` can stop a Pod).
``verify`` runs after collection and before STOP; raising must leave the worker
and its artifacts in place for operator recovery.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from uuid import uuid4

import pytest


_READ_ENABLED = os.environ.get("RUNPOD_INTEGRATION") == "1"
_PROFILE = os.environ.get("SCHEDULER_RUNPOD_PROFILE", "runpod:h200x1")
_ARTIFACT_ROOT_ENV = "SCHEDULER_RUNPOD_SMOKE_ARTIFACT_ROOT"
_SUBPROCESS_TIMEOUT_SECONDS = 120
_STOP_MARGIN_SECONDS = 300
_PROVIDER_MUTATION_TOKENS = frozenset(
    {
        "create",
        "cancel",
        "delete",
        "deploy",
        "destroy",
        "detach",
        "edit",
        "move",
        "patch",
        "reboot",
        "remove",
        "replace",
        "reset",
        "resize",
        "restart",
        "resume",
        "rm",
        "run",
        "scale",
        "set",
        "start",
        "stop",
        "terminate",
        "update",
    }
)
_DESTRUCTIVE_TOKENS = frozenset({"delete", "destroy", "remove", "rm", "terminate"})
_PAID_SMOKE_MUTATIONS = frozenset({"create", "stop"})


def _argv_words(value: str) -> tuple[str, ...]:
    """Split argv syntax, including camelCase and hyphenated option names."""

    separated = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    return tuple(word.lower() for word in re.split(r"[^A-Za-z0-9]+", separated) if word)


def _contains_destructive_verb(value: str) -> bool:
    return bool(_DESTRUCTIVE_TOKENS.intersection(_argv_words(value)))


def _mutation_verbs(value: str) -> frozenset[str]:
    return frozenset(_PROVIDER_MUTATION_TOKENS.intersection(_argv_words(value)))


def _looks_like_credential_syntax(value: str) -> bool:
    """Recognize credential-named options/assignments without reading secrets."""

    stripped = value.strip()
    has_credential_syntax = (
        stripped.startswith("-") or "=" in stripped or ":" in stripped
    )
    # Bare positional marker names (for example RUNPOD_API_KEY) are forbidden
    # too.  Restrict this path to identifier-like strings so ordinary opaque
    # positional values are not searched for credential-looking substrings.
    is_positional_marker = bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", stripped))
    if not (has_credential_syntax or is_positional_marker):
        return False
    candidate = stripped
    words = set(_argv_words(candidate))
    credential_words = {
        "apikey",
        "authorization",
        "credential",
        "password",
        "secret",
        "token",
    }
    if words.intersection(credential_words):
        return True
    return "api" in words and "key" in words


class _AuditingRunner:
    """Log argv within an explicit provider-mutation allowlist; shell-free.

    An *audit aid, not a security boundary*: the harness may use an SDK client
    or its own subprocesses, in which case nothing here is consulted and this
    record is incomplete by construction.  The load-bearing checks are the
    independent inventory comparisons and the byte-level artifact verification.
    """

    def __init__(
        self, audit_path: Path, *, allowed_mutations: frozenset[str]
    ) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.audit_path = audit_path
        self.allowed_mutations = allowed_mutations
        audit_path.parent.mkdir(parents=True, exist_ok=True)

    def mutation_calls(self) -> list[tuple[tuple[str, ...], frozenset[str]]]:
        calls: list[tuple[tuple[str, ...], frozenset[str]]] = []
        for call in self.calls:
            verbs = frozenset().union(*(_mutation_verbs(part) for part in call))
            if verbs:
                calls.append((call, verbs))
        return calls

    def destructive_calls(self) -> list[tuple[str, ...]]:
        return [
            c for c in self.calls if any(_contains_destructive_verb(p) for p in c)
        ]

    def __call__(self, argv: Sequence[object], **kwargs: object) -> object:
        if kwargs.pop("shell", False):
            raise AssertionError("shell execution is not permitted")
        normalized = tuple(str(part) for part in argv)
        for index, part in enumerate(normalized):
            forbidden = _mutation_verbs(part) - self.allowed_mutations
            if forbidden:
                raise AssertionError(
                    "refusing to run or audit provider mutation(s) "
                    f"{sorted(forbidden)} at index {index}"
                )
        for index, part in enumerate(normalized):
            # Fail closed before running *or* recording: rejecting the flag also
            # keeps its value out of the audit file.
            if _looks_like_credential_syntax(part):
                raise AssertionError(
                    f"refusing to run or audit credential-bearing argv at index {index}"
                )
        timeout = kwargs.pop("timeout", None)
        bounded = (
            _SUBPROCESS_TIMEOUT_SECONDS
            if timeout is None
            else min(float(timeout), _SUBPROCESS_TIMEOUT_SECONDS)
        )
        self.calls.append(normalized)
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"argv": list(normalized)}) + "\n")
        return subprocess.run(list(normalized), shell=False, timeout=bounded, **kwargs)


def _require_prerequisites(*, paid: bool) -> None:
    """Name the missing settings and the setup action, never a secret value."""

    missing: list[str] = []
    if "RUNPOD_API_KEY" not in os.environ and not (
        Path.home() / ".runpod" / "config.toml"
    ).is_file():
        missing.append(
            "RunPod authentication (`RUNPOD_API_KEY`, or `~/.runpod/config.toml` "
            "from `runpodctl config --apiKey ...`)"
        )
    if paid and not os.environ.get(_ARTIFACT_ROOT_ENV):
        missing.append(f"{_ARTIFACT_ROOT_ENV} (a durable, never-removed artifact root)")
    if missing:
        pytest.fail(
            "RunPod integration prerequisites are missing:\n- " + "\n- ".join(missing),
            pytrace=False,
        )


def _load_harness_factory():
    """Import the scheduler RunPod seam only after an opt-in flag is set."""

    try:
        module = importlib.import_module("tools.scheduler.runpod")
    except ModuleNotFoundError:
        pytest.fail("`tools.scheduler.runpod` is not implemented", pytrace=False)
    factory = getattr(module, "create_integration_harness", None)
    if not callable(factory):
        pytest.fail("`create_integration_harness` is missing", pytrace=False)
    return factory


def _capture_inventory(harness: object, path: Path) -> list[dict[str, str]]:
    records = harness.inventory()
    assert isinstance(records, Sequence) and not isinstance(records, (str, bytes)), (
        "inventory must be a sequence of pod mappings"
    )
    inventory: list[dict[str, str]] = []
    for record in records:
        assert isinstance(record, Mapping), "inventory entry must be a mapping"
        pod_id = record.get("id")
        assert isinstance(pod_id, str) and pod_id, "inventory entry needs an id"
        name, status = record.get("name") or "", record.get("status") or ""
        inventory.append({"id": pod_id, "name": str(name), "status": str(status)})
    inventory.sort(key=lambda pod: pod["id"])
    path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    return inventory


def _assert_preexisting_unchanged(
    before: list[dict[str, str]], after: list[dict[str, str]]
) -> None:
    after_by_id = {pod["id"]: pod for pod in after}
    for pod in before:
        survivor = after_by_id.get(pod["id"])
        assert survivor is not None and survivor["name"] == pod["name"], (
            "a pre-existing Pod disappeared or changed stable identity: " + pod["id"]
        )


def _assert_inventory_equal(
    before: list[dict[str, str]], after: list[dict[str, str]]
) -> None:
    assert after == before, (
        "read-only preflight changed provider inventory identity or status"
    )


def _assert_exact_owned_delta(
    before: list[dict[str, str]], current: list[dict[str, str]], worker_id: str
) -> None:
    before_ids = {pod["id"] for pod in before}
    current_ids = {pod["id"] for pod in current}
    assert current_ids - before_ids == {worker_id}, (
        "provider inventory delta was not exactly the one verified owned Pod"
    )


def _verified_bytes(root: Path, value: object, expected: bytes) -> str:
    """Assert a contained, non-symlink regular file holds exactly ``expected``."""

    path = Path(os.fspath(value))
    assert path.is_absolute(), f"artifact path must be absolute: {path}"
    assert path.resolve(strict=True).is_relative_to(root.resolve(strict=True)), (
        f"artifact escapes the artifact root: {path}"
    )
    mode = path.lstat().st_mode
    assert stat.S_ISREG(mode) and not stat.S_ISLNK(mode), (
        f"artifact must be a regular file, not a symlink: {path}"
    )
    actual = path.read_bytes()
    assert actual == expected, f"artifact bytes differ from expectation: {path}"
    return hashlib.sha256(actual).hexdigest()


@pytest.mark.skipif(
    not _READ_ENABLED,
    reason="set RUNPOD_INTEGRATION=1 for the zero-mutation RunPod preflight",
)
def test_runpod_authenticated_preflight_is_read_only(tmp_path: Path) -> None:
    _require_prerequisites(paid=False)
    create_integration_harness = _load_harness_factory()

    runner = _AuditingRunner(
        tmp_path / "preflight-invocations.jsonl", allowed_mutations=frozenset()
    )
    harness = create_integration_harness(
        profile=_PROFILE,
        artifact_root=tmp_path / "preflight",
        command_runner=runner,
        timeout_seconds=_SUBPROCESS_TIMEOUT_SECONDS,
    )

    before = _capture_inventory(harness, tmp_path / "inventory-before.json")
    try:
        report = harness.preflight()
        assert isinstance(report, Mapping) and report, (
            "preflight must return a non-empty mapping"
        )
    finally:
        after = _capture_inventory(harness, tmp_path / "inventory-after.json")

    _assert_inventory_equal(before, after)
    assert not runner.mutation_calls()
    assert not runner.destructive_calls()


@pytest.mark.skip(
    reason=(
        "paid RunPod smoke is hard-disabled: it lacks price/storage checks, "
        "a certified CREATE-time provider-enforced stopAfter path through the "
        "unresolved replacement seam, and the actual CLI/socket/service/SQLite "
        "path; no environment variable enables it"
    )
)
def test_paid_runpod_owned_worker_lifecycle_smoke() -> None:
    """Obsolete direct-harness sketch retained behind an unconditional skip."""

    _require_prerequisites(paid=True)
    create_integration_harness = _load_harness_factory()

    durable_root = Path(os.environ[_ARTIFACT_ROOT_ENV]).expanduser().resolve()
    durable_root.mkdir(parents=True, exist_ok=True)

    timeout_seconds = int(
        os.environ.get("SCHEDULER_RUNPOD_SMOKE_TIMEOUT_SECONDS", "1200")
    )
    ttl_minutes = int(os.environ.get("SCHEDULER_RUNPOD_TTL_MINUTES", "30"))
    assert timeout_seconds > 0
    assert 1 <= ttl_minutes <= 60
    assert ttl_minutes * 60 > timeout_seconds + _STOP_MARGIN_SECONDS, (
        "the provider TTL must exceed the smoke timeout plus the stop margin"
    )

    worker_name = f"scheduler-smoke-{uuid4().hex}"
    run_root = durable_root / worker_name
    run_root.mkdir(parents=False, exist_ok=False)
    source = run_root / "frozen-input.bin"
    source_bytes = b"deterministic scheduler RunPod smoke input\n"
    source.write_bytes(source_bytes)
    expected = {
        "outputs/result.bin": source_bytes + b"scheduler-smoke-ok\n",
        "stdout": b"scheduler-smoke-ok\n",
        "stderr": b"",
        "terminal": b"0\n",
    }

    runner = _AuditingRunner(
        run_root / "provider-invocations.jsonl",
        allowed_mutations=_PAID_SMOKE_MUTATIONS,
    )
    harness = create_integration_harness(
        profile=_PROFILE,
        artifact_root=run_root,
        command_runner=runner,
        timeout_seconds=timeout_seconds,
    )
    before = _capture_inventory(harness, run_root / "inventory-before.json")
    before_ids = {pod["id"] for pod in before}
    verified: dict[str, object] = {}
    candidate_observation: dict[str, str] = {}

    def verify(result: object) -> None:
        """Run before the harness stops the worker; raising preserves it."""

        assert isinstance(result, Mapping), "smoke result must be a mapping"
        worker_id = result.get("worker_id")
        assert isinstance(worker_id, str) and worker_id
        assert worker_id not in before_ids, "smoke reused a pre-existing Pod"

        mid = _capture_inventory(harness, run_root / "inventory-before-stop.json")
        _assert_preexisting_unchanged(before, mid)
        _assert_exact_owned_delta(before, mid, worker_id)
        owned = [pod for pod in mid if pod["id"] == worker_id]
        assert len(owned) == 1 and owned[0]["name"] == worker_name, (
            "the owned worker is not the uniquely named Pod this test created"
        )
        candidate_observation["worker_id"] = worker_id

        artifacts = result.get("artifacts")
        assert isinstance(artifacts, Mapping), "smoke must report artifact paths"
        missing = sorted(set(expected) - set(artifacts))
        assert not missing, f"smoke did not report artifacts: {missing}"

        # The durable terminal record proves success; exit_code is checked too.
        hashes = {
            name: _verified_bytes(run_root, artifacts[name], payload)
            for name, payload in expected.items()
        }
        assert int(result["exit_code"]) == 0, "remote command reported failure"

        (run_root / "independent-verification.json").write_text(
            json.dumps(hashes, indent=2, sort_keys=True) + "\n"
        )
        verified.update({"worker_id": worker_id, "hashes": hashes})

    try:
        result = harness.run_owned_smoke(
            name=worker_name,
            ttl_minutes=ttl_minutes,
            command=(
                "python3",
                "-c",
                "from pathlib import Path; "
                "src=Path('inputs/frozen-input.bin').read_bytes(); "
                "dst=Path('outputs/result.bin'); "
                "dst.parent.mkdir(parents=True, exist_ok=True); "
                "dst.write_bytes(src + b'scheduler-smoke-ok\\n'); "
                "print('scheduler-smoke-ok', flush=True)",
            ),
            input_paths={"inputs/frozen-input.bin": source},
            output_paths=("outputs/result.bin",),
            verify=verify,
            timeout_seconds=timeout_seconds,
        )
        assert verified, "the harness stopped or returned without verifying"
        assert result["worker_id"] == verified["worker_id"]
        assert result.get("stopped") is True, "the harness did not stop its worker"
    finally:
        after = _capture_inventory(harness, run_root / "inventory-after.json")
        _assert_preexisting_unchanged(before, after)
        audited_mutations = frozenset().union(
            *(verbs for _, verbs in runner.mutation_calls())
        )
        assert audited_mutations <= _PAID_SMOKE_MUTATIONS
        if not verified:
            assert "stop" not in audited_mutations, (
                "the harness issued STOP despite failed or incomplete verification"
            )
        assert not runner.destructive_calls(), (
            f"an audited provider call looked destructive; inspect {runner.audit_path}"
        )
        new_pods = [pod for pod in after if pod["id"] not in before_ids]
        assert len(new_pods) <= 1, (
            "the smoke created extra Pods, including differently named ones"
        )
        if new_pods:
            assert new_pods[0]["name"] == worker_name, (
                "the only new Pod is not the uniquely named worker owned by this smoke"
            )

        worker_id = verified.get("worker_id")
        if worker_id is not None:
            _assert_exact_owned_delta(before, after, worker_id)
            owned = [pod for pod in after if pod["id"] == worker_id]
            assert owned, "the owned Pod is gone; it must be stopped, not deleted"
            assert owned[0]["status"].lower() in {"stopped", "exited"}, (
                "the owned Pod did not reach a stopped state; artifacts and the "
                f"Pod are preserved under {run_root}"
            )
        else:
            candidate_id = candidate_observation.get("worker_id")
            if candidate_id is not None:
                assert len(new_pods) == 1 and new_pods[0]["id"] == candidate_id, (
                    "the worker observed before failed verification disappeared"
                )
        if not verified and new_pods:
            assert new_pods[0]["status"].lower() not in {
                "exited",
                "stopped",
                "stopping",
                "terminated",
            }, (
                "the harness stopped its worker despite failed verification"
            )
            print(
                f"RunPod smoke left owned Pod {new_pods[0]['id']} "
                f"({new_pods[0]['status']}) and artifacts under {run_root}"
            )
