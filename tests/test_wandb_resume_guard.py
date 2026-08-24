"""W&B resume state guard and same-host process exclusion.

These tests use fake W&B objects and real kernel ``flock`` calls. They never
contact W&B or any other network service.
"""

from __future__ import annotations

import fcntl
import gc
import hashlib
import inspect
import os
import pwd
import stat
import subprocess
import sys
import threading
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import infra.train as train_mod
from infra.run_common import (
    acquire_wandb_resume_cli_lease,
    apply_wandb_resume_cli_authorization,
    release_wandb_resume_cli_lease,
)
from infra.train import Config, validate_resume_args
from wandb.sdk.wandb_settings import Settings as RealWandbSettings


IDENTITY = {"dataset_type": "math", "grading_protocol": "numeric-v1"}
REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeConfig(dict):
    def __init__(self, *args, fail_update: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.fail_update = fail_update
        self.updates = []

    def update(self, values, *, allow_val_change=False):
        if self.fail_update:
            raise RuntimeError("config update failed")
        self.updates.append((dict(values), allow_val_change))
        super().update(values)


class FakeRun:
    def __init__(self, config, *, fail_update=False, fail_finish=False):
        self.config = FakeConfig(config, fail_update=fail_update)
        self.fail_finish = fail_finish
        self.finish_calls = 0

    def finish(self):
        self.finish_calls += 1
        if self.fail_finish:
            raise RuntimeError("finish failed")


class PublicRun:
    def __init__(self, state, config, *, state_error=None, missing_state=False):
        self._state = state
        self.config = config
        self.state_error = state_error
        self.missing_state = missing_state

    @property
    def state(self):
        if self.state_error is not None:
            raise self.state_error
        if self.missing_state:
            raise AttributeError("state")
        return self._state


class FakeWandb:
    __version__ = "0.28.1"
    Settings = RealWandbSettings

    def __init__(
        self,
        *,
        state="finished",
        stored_config=None,
        state_error=None,
        missing_state=False,
        fail_init=False,
        fail_update=False,
        fail_finish=False,
    ):
        self.stored_config = stored_config or {
            "protocol_identity": IDENTITY,
            "launch_namespaces": ["attempt-original"],
        }
        self.public_run = PublicRun(
            state,
            self.stored_config,
            state_error=state_error,
            missing_state=missing_state,
        )
        self.run = FakeRun(
            self.stored_config,
            fail_update=fail_update,
            fail_finish=fail_finish,
        )
        self.fail_init = fail_init
        self.api_run_calls = []
        self.init_calls = []

    def Api(self):
        owner = self

        class Api:
            def run(self, path):
                owner.api_run_calls.append(path)
                return owner.public_run

        return Api()

    def init(self, **kwargs):
        if kwargs.get("resume") == "must":
            assert kwargs.get("mode") == "online"
        self.init_calls.append(kwargs)
        if self.fail_init:
            raise RuntimeError("init failed")
        return self.run


@pytest.fixture
def lock_root(tmp_path, monkeypatch):
    root = tmp_path / "wandb-resume-locks"
    monkeypatch.setattr(train_mod, "_wandb_resume_lock_root", lambda: str(root))
    return root


def _install(monkeypatch, wandb):
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    monkeypatch.setattr(train_mod, "_env_identity", lambda: {"env/git_dirty": "no"})
    return wandb


def _source_line(function, needle: str) -> int:
    lines, first = inspect.getsourcelines(function)
    matches = [
        first + offset
        for offset, line in enumerate(lines)
        if needle in line
    ]
    assert len(matches) == 1, (function.__qualname__, needle, matches)
    return matches[0]


def _call_with_line_interrupt(function, line: int, injected, callback):
    fired = False

    def trace(frame, event, arg):
        nonlocal fired
        if (
            not fired
            and event == "line"
            and frame.f_code is function.__code__
            and frame.f_lineno == line
        ):
            fired = True
            raise injected
        return trace

    sys.settrace(trace)
    try:
        return callback()
    finally:
        sys.settrace(None)
        assert fired, f"trace never reached {function.__qualname__}:{line}"


def _assert_raw_flock_reacquirable(lock_root: Path, run_id: str) -> None:
    lock_path = lock_root / _lock_name(run_id)
    fd = os.open(lock_path, os.O_RDWR | os.O_CLOEXEC)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _cfg(*, running_override=False, **overrides):
    values = dict(
        wandb_project="project",
        wandb_run_id="run-id",
        launch_namespace="attempt-new",
        protocol_identity=IDENTITY,
        log_transcripts=False,
        steps=0,
        eval_every=0,
        save_every=0,
    )
    values.update(overrides)
    cfg = Config(**values)
    if running_override:
        args = SimpleNamespace(
            wandb_resume=cfg.wandb_run_id,
            wandb_resume_running_override=True,
            no_wandb=False,
            load="/checkpoints/step-00000",
            start_step=None,
        )
        handoff = acquire_wandb_resume_cli_lease(args)
        apply_wandb_resume_cli_authorization(
            cfg,
            args,
            resume_lease_handoff=handoff,
        )
    return cfg


@pytest.mark.parametrize("state", ["finished", "crashed", "failed", "killed"])
def test_terminal_state_allows_resume_with_exactly_one_public_fetch(
    monkeypatch, lock_root, state
):
    monkeypatch.setenv("WANDB_MODE", "offline")
    wandb = _install(monkeypatch, FakeWandb(state=state))
    logger = train_mod._make_logger(_cfg())
    try:
        assert wandb.api_run_calls == ["project/run-id"]
        assert len(wandb.init_calls) == 1
        assert wandb.init_calls[0]["mode"] == "online"
        update = wandb.run.config.updates[0][0]
        assert update["launch_namespaces"] == [
            "attempt-original",
            "attempt-new",
        ]
        assert update["wandb_resume_running_overrides"] == []
        assert "wandb_resume_running_override" not in update
        assert "_wandb_resume_running_capability" not in update
        assert "_wandb_resume_lease_handoff" not in update
    finally:
        logger.close()


def test_running_state_refuses_without_override(monkeypatch, lock_root):
    wandb = _install(monkeypatch, FakeWandb(state="running"))
    with pytest.raises(RuntimeError, match="state is 'running'"):
        train_mod._make_logger(_cfg())
    assert wandb.api_run_calls == ["project/run-id"]
    assert wandb.init_calls == []


def test_running_override_is_recorded_by_namespace_not_raw_boolean(
    monkeypatch, lock_root, capsys
):
    wandb = _install(
        monkeypatch,
        FakeWandb(
            state="running",
            stored_config={
                "protocol_identity": IDENTITY,
                "launch_namespaces": [
                    "attempt-original",
                    "attempt-stale-before",
                ],
                "wandb_resume_running_overrides": ["attempt-stale-before"],
            },
        ),
    )
    logger = train_mod._make_logger(_cfg(running_override=True))
    try:
        update = wandb.run.config.updates[0][0]
        assert update["wandb_resume_running_overrides"] == [
            "attempt-stale-before",
            "attempt-new",
        ]
        assert "wandb_resume_running_override" not in update
        assert "_wandb_resume_running_capability" not in update
        assert "_wandb_resume_lease_handoff" not in update
        assert wandb.api_run_calls == ["project/run-id"]
        assert wandb.init_calls[0]["mode"] == "online"
        notice = capsys.readouterr().err
        assert "explicit running-state resume override accepted" in notice
        assert "run-id" in notice
        assert "attempt-new" in notice
    finally:
        logger.close()


@pytest.mark.parametrize("state", ["finished", "crashed", "failed", "killed"])
def test_running_override_refuses_when_terminal_and_unnecessary(
    monkeypatch, lock_root, state
):
    wandb = _install(monkeypatch, FakeWandb(state=state))
    with pytest.raises(ValueError, match="override is unnecessary"):
        train_mod._make_logger(_cfg(running_override=True))
    assert wandb.init_calls == []


@pytest.mark.parametrize("state", [None, 3, "pending", "RUNNING", " finished "])
@pytest.mark.parametrize("override", [False, True])
def test_unknown_or_non_string_state_always_refuses(
    monkeypatch, lock_root, state, override
):
    wandb = _install(monkeypatch, FakeWandb(state=state))
    with pytest.raises(RuntimeError, match="unknown or missing remote state"):
        train_mod._make_logger(
            _cfg(running_override=override)
        )
    assert wandb.init_calls == []


@pytest.mark.parametrize(
    "wandb",
    [
        pytest.param(FakeWandb(missing_state=True), id="missing"),
        pytest.param(
            FakeWandb(state_error=RuntimeError("property unavailable")),
            id="property-error",
        ),
    ],
)
def test_unreadable_state_always_refuses_even_with_override(
    monkeypatch, lock_root, wandb
):
    _install(monkeypatch, wandb)
    with pytest.raises(RuntimeError, match="state could not be read"):
        train_mod._make_logger(_cfg(running_override=True))
    assert wandb.init_calls == []


@pytest.mark.parametrize("control_flow", [KeyboardInterrupt, SystemExit])
def test_remote_state_control_flow_propagates_and_releases_lock(
    monkeypatch, lock_root, control_flow
):
    injected = control_flow("remote state read interrupted")
    wandb = _install(monkeypatch, FakeWandb(state_error=injected))
    with pytest.raises(control_flow) as caught:
        train_mod._make_logger(_cfg())
    assert caught.value is injected
    assert wandb.api_run_calls == ["project/run-id"]
    assert wandb.init_calls == []
    _assert_raw_flock_reacquirable(lock_root, "run-id")


def test_pinned_wandb_sdk_state_and_config_update_contract_without_network():
    import inspect

    import wandb
    from wandb.sdk import wandb_init
    from wandb.sdk.wandb_config import Config as WandbConfig

    assert wandb.__version__ == "0.28.1"
    public_run = object.__new__(wandb.apis.public.Run)
    public_run._state = "running"
    assert public_run.state == "running"
    public_run._state = "finished"
    assert public_run.state == "finished"

    # Pinned SDK source applies explicit wandb.init settings after inherited
    # global/env settings. Its Settings merge confirms online wins over both
    # offline and disabled without opening a run or touching the network.
    make_settings_source = inspect.getsource(
        wandb_init._WandbInit.make_run_settings
    )
    inherited = "settings = self._wl.settings.model_copy()"
    explicit = "settings.update_from_settings(init_settings)"
    assert make_settings_source.index(inherited) < make_settings_source.index(explicit)
    for inherited_mode in ("offline", "disabled"):
        settings = wandb.Settings(mode=inherited_mode)
        settings.update_from_settings(wandb.Settings(mode="online"))
        assert settings.mode == "online"
        assert settings._offline is False
        assert settings._noop is False

    callbacks = []
    config = WandbConfig()
    config._set_callback(lambda **kwargs: callbacks.append(kwargs))
    config.update(
        {"launch_namespaces": ["attempt-original"]},
        allow_val_change=True,
    )
    config.update(
        {"launch_namespaces": ["attempt-original", "attempt-new"]},
        allow_val_change=True,
    )
    assert config["launch_namespaces"] == ["attempt-original", "attempt-new"]
    assert callbacks == [
        {"data": {"launch_namespaces": ["attempt-original"]}},
        {
            "data": {
                "launch_namespaces": ["attempt-original", "attempt-new"]
            }
        },
    ]


def test_wandb_version_is_exactly_pinned_in_all_install_surfaces():
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    assert "wandb==0.28.1" in project["project"]["dependencies"]
    assert project["project"]["optional-dependencies"]["logging"] == [
        "wandb==0.28.1"
    ]
    for relative in ("scripts/provision_sm90.sh", "scripts/provision_blackwell.sh"):
        provision = (REPO_ROOT / relative).read_text()
        assert '"wandb==0.28.1"' in provision
        assert '"wandb>=0.18"' not in provision
    lock = (REPO_ROOT / "uv.lock").read_text()
    assert '{ name = "wandb", specifier = "==0.28.1" }' in lock
    assert (
        '{ name = "wandb", marker = "extra == \'logging\'", '
        'specifier = "==0.28.1" }'
    ) in lock


@pytest.mark.parametrize(
    ("raw", "validated"),
    [
        ("abc123", "abc123"),
        ("a b", "a b"),
        ("雪", "雪"),
        ("back\\slash", "back\\slash"),
        ("a\nb", "a\nb"),
        (b"bytes-id", "bytes-id"),
    ],
)
def test_installed_sdk_is_the_no_network_run_id_oracle(
    monkeypatch, raw, validated
):
    import wandb

    def forbidden_api(*args, **kwargs):
        raise AssertionError("run-ID validation must not construct a W&B API")

    monkeypatch.setattr(wandb, "Api", forbidden_api)
    assert RealWandbSettings(run_id=raw).run_id == validated
    assert train_mod._validate_wandb_resume_sdk_contract(wandb, raw) == validated


def test_resume_uses_sdk_normalized_run_id_for_lock_api_and_provenance(
    monkeypatch, lock_root
):
    wandb = _install(monkeypatch, FakeWandb())
    cfg = _cfg(wandb_run_id=b"bytes-id")
    logger = train_mod._make_logger(cfg)
    try:
        assert cfg.wandb_run_id == "bytes-id"
        assert wandb.api_run_calls == ["project/bytes-id"]
        assert wandb.init_calls[0]["id"] == "bytes-id"
        assert wandb.run.config.updates[0][0]["wandb_run_id"] == "bytes-id"
    finally:
        logger.close()


@pytest.mark.parametrize(
    "run_id",
    ["", " run", "run ", " ", ":", ";", ",", "#", "?", "/", "'", 3],
)
def test_invalid_sdk_run_id_refuses_before_lock_or_wandb_call(
    monkeypatch, lock_root, run_id
):
    wandb = _install(monkeypatch, FakeWandb())

    def forbidden_lock(*args, **kwargs):
        raise AssertionError("invalid run ID reached local lock")

    monkeypatch.setattr(train_mod, "_acquire_wandb_resume_lock", forbidden_lock)
    with pytest.raises(ValueError, match="invalid W&B resume run ID"):
        train_mod._make_logger(_cfg(wandb_run_id=run_id))
    assert wandb.api_run_calls == []
    assert wandb.init_calls == []


@pytest.mark.parametrize("drift", [None, "0.28.0", "0.28.2"])
def test_loaded_sdk_version_drift_refuses_before_lock_or_api(
    monkeypatch, lock_root, drift
):
    wandb = _install(monkeypatch, FakeWandb())
    wandb.__version__ = drift

    def forbidden_lock(*args, **kwargs):
        raise AssertionError("version drift reached local lock")

    monkeypatch.setattr(train_mod, "_acquire_wandb_resume_lock", forbidden_lock)
    with pytest.raises(RuntimeError, match="requires exact SDK version 0.28.1"):
        train_mod._make_logger(_cfg())
    assert wandb.api_run_calls == []
    assert wandb.init_calls == []


def test_installed_distribution_version_drift_refuses_before_lock_or_api(
    monkeypatch, lock_root
):
    wandb = _install(monkeypatch, FakeWandb())
    monkeypatch.setattr(train_mod.importlib.metadata, "version", lambda name: "0.28.2")

    def forbidden_lock(*args, **kwargs):
        raise AssertionError("distribution drift reached local lock")

    monkeypatch.setattr(train_mod, "_acquire_wandb_resume_lock", forbidden_lock)
    with pytest.raises(RuntimeError, match="requires exact SDK version 0.28.1"):
        train_mod._make_logger(_cfg())
    assert wandb.api_run_calls == []
    assert wandb.init_calls == []


@pytest.mark.parametrize(
    "history",
    ["attempt", ["valid", 3], ["../escape"], ["duplicate", "duplicate"]],
)
def test_malformed_override_history_refuses_before_init(
    monkeypatch, lock_root, history
):
    wandb = _install(
        monkeypatch,
        FakeWandb(
            stored_config={
                "protocol_identity": IDENTITY,
                "launch_namespaces": ["attempt-original"],
                "wandb_resume_running_overrides": history,
            }
        ),
    )
    with pytest.raises(ValueError, match="malformed wandb_resume_running_overrides"):
        train_mod._make_logger(_cfg())
    assert wandb.init_calls == []


@pytest.mark.parametrize(
    ("launches", "overrides"),
    [
        (["attempt-original"], ["attempt-orphan"]),
        (
            ["attempt-original", "attempt-second"],
            ["attempt-second", "attempt-original"],
        ),
    ],
)
def test_override_history_must_be_ordered_subset_of_launch_history(
    monkeypatch, lock_root, launches, overrides
):
    wandb = _install(
        monkeypatch,
        FakeWandb(
            stored_config={
                "protocol_identity": IDENTITY,
                "launch_namespaces": launches,
                "wandb_resume_running_overrides": overrides,
            }
        ),
    )
    with pytest.raises(ValueError, match="order-consistent subset"):
        train_mod._make_logger(_cfg())
    assert wandb.init_calls == []


def test_local_contention_refuses_before_any_wandb_api_call(
    monkeypatch, lock_root
):
    held = train_mod._acquire_wandb_resume_lock(
        "run-id", state_root=str(lock_root)
    )
    try:
        wandb = _install(monkeypatch, FakeWandb())
        with pytest.raises(RuntimeError, match="another local process"):
            train_mod._make_logger(_cfg())
        assert wandb.api_run_calls == []
        assert wandb.init_calls == []
    finally:
        held.release()


def test_local_contention_cannot_be_bypassed_by_running_override(
    monkeypatch, lock_root
):
    held = train_mod._acquire_wandb_resume_lock(
        "run-id", state_root=str(lock_root)
    )
    try:
        wandb = _install(monkeypatch, FakeWandb(state="running"))
        with pytest.raises(RuntimeError, match="another local process"):
            train_mod._make_logger(
                _cfg(running_override=True)
            )
        assert wandb.api_run_calls == []
        assert wandb.init_calls == []
    finally:
        held.release()


def test_lock_lives_until_close_then_releases(monkeypatch, lock_root):
    wandb = _install(monkeypatch, FakeWandb())
    logger = train_mod._make_logger(_cfg())
    with pytest.raises(RuntimeError, match="another local process"):
        train_mod._acquire_wandb_resume_lock("run-id", state_root=str(lock_root))
    logger.close()
    replacement = train_mod._acquire_wandb_resume_lock(
        "run-id", state_root=str(lock_root)
    )
    replacement.release()
    assert wandb.run.finish_calls == 1


def test_runner_handoff_is_consumed_once_without_release_reacquire_gap(
    monkeypatch, lock_root
):
    wandb = _install(monkeypatch, FakeWandb())
    args = SimpleNamespace(
        wandb_resume="run-id",
        wandb_resume_running_override=False,
        no_wandb=False,
        load="/checkpoints/step-00000",
        start_step=None,
    )
    handoff = acquire_wandb_resume_cli_lease(args)
    cfg = _cfg()
    apply_wandb_resume_cli_authorization(
        cfg, args, resume_lease_handoff=handoff
    )
    logger = train_mod._make_logger(cfg)
    assert cfg._wandb_resume_lease_handoff is None
    assert "_wandb_resume_lease_handoff" not in wandb.run.config.updates[0][0]
    # The runner's finally path sees a consumed handoff. It must not release
    # the descriptor now owned by the logger.
    release_wandb_resume_cli_lease(handoff)
    with pytest.raises(RuntimeError, match="another local process"):
        train_mod._acquire_wandb_resume_lock(
            "run-id", state_root=str(lock_root)
        )
    logger.close()
    replacement = train_mod._acquire_wandb_resume_lock(
        "run-id", state_root=str(lock_root)
    )
    replacement.release()
    with pytest.raises(ValueError, match="stale, reused, or mismatched"):
        handoff.take(cfg, "run-id")


@pytest.mark.parametrize("control_flow", [KeyboardInterrupt, SystemExit])
def test_post_transfer_baseexception_releases_raw_flock_and_preserves_signal(
    monkeypatch, lock_root, control_flow
):
    wandb = _install(monkeypatch, FakeWandb())
    monkeypatch.setattr(
        train_mod, "_env_identity", lambda: {"env/git_dirty": "yes"}
    )
    args = SimpleNamespace(
        wandb_resume="run-id",
        wandb_resume_running_override=False,
        no_wandb=False,
        load="/checkpoints/step-00000",
        start_step=None,
    )
    handoff = acquire_wandb_resume_cli_lease(args)
    cfg = _cfg()
    apply_wandb_resume_cli_authorization(
        cfg, args, resume_lease_handoff=handoff
    )
    injected = control_flow("post-transfer provenance interruption")

    def interrupt_provenance(_run):
        raise injected

    monkeypatch.setattr(train_mod, "_save_dirty_patch", interrupt_provenance)
    with pytest.raises(control_flow) as caught:
        train_mod._make_logger(cfg)
    assert caught.value is injected
    assert wandb.run.finish_calls == 1
    assert cfg._wandb_resume_lease_handoff is None

    # The runner's outer cleanup remains a no-op after logger adoption, and a
    # raw kernel lock proves the descriptor was released in this process.
    release_wandb_resume_cli_lease(handoff)
    lock_path = lock_root / _lock_name("run-id")
    fd = os.open(lock_path, os.O_RDWR | os.O_CLOEXEC)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


@pytest.mark.parametrize("failure", ["init", "update"])
def test_prelogger_failures_release_lock(monkeypatch, lock_root, failure):
    wandb = _install(
        monkeypatch,
        FakeWandb(
            fail_init=failure == "init",
            fail_update=failure == "update",
        ),
    )
    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        train_mod._make_logger(_cfg())
    replacement = train_mod._acquire_wandb_resume_lock(
        "run-id", state_root=str(lock_root)
    )
    replacement.release()
    if failure == "update":
        assert wandb.run.finish_calls == 1


@pytest.mark.parametrize("control_flow", [KeyboardInterrupt, SystemExit])
def test_resume_update_control_flow_survives_finish_cleanup_control_flow(
    monkeypatch, lock_root, control_flow
):
    original = control_flow("resume config update interrupted")
    cleanup = BaseException("finish cleanup interrupted")
    wandb = _install(monkeypatch, FakeWandb())

    def interrupt_update(_values, *, allow_val_change=False):
        raise original

    def interrupt_finish():
        wandb.run.finish_calls += 1
        raise cleanup

    wandb.run.config.update = interrupt_update
    wandb.run.finish = interrupt_finish
    with pytest.raises(control_flow) as caught:
        train_mod._make_logger(_cfg())
    assert caught.value is original
    assert wandb.run.finish_calls == 1
    _assert_raw_flock_reacquirable(lock_root, "run-id")


@pytest.mark.parametrize("control_flow", [KeyboardInterrupt, SystemExit])
def test_resume_control_flow_survives_release_cleanup_control_flow(
    monkeypatch, lock_root, control_flow
):
    original = control_flow("remote state interrupted")
    cleanup = BaseException("lease release interrupted")
    wandb = _install(monkeypatch, FakeWandb(state_error=original))
    real_release = train_mod._WandbResumeLock.release

    def release_then_interrupt(lease):
        real_release(lease)
        raise cleanup

    monkeypatch.setattr(
        train_mod._WandbResumeLock, "release", release_then_interrupt
    )
    with pytest.raises(control_flow) as caught:
        train_mod._make_logger(_cfg())
    assert caught.value is original
    assert wandb.api_run_calls == ["project/run-id"]
    assert wandb.init_calls == []
    _assert_raw_flock_reacquirable(lock_root, "run-id")


@pytest.mark.parametrize("control_flow", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize(
    "window",
    [
        "logger = _RunLogger(run, resume_lock=resume_lock)",
        "resume_lock = None",
    ],
    ids=["before-logger-ownership", "before-lock-local-clear"],
)
def test_post_init_control_flow_finishes_run_and_releases_lock(
    monkeypatch, lock_root, control_flow, window
):
    original = control_flow("post-init ownership interrupted")
    wandb = _install(monkeypatch, FakeWandb())
    target_line = _source_line(train_mod._make_logger, window)
    with pytest.raises(control_flow) as caught:
        _call_with_line_interrupt(
            train_mod._make_logger,
            target_line,
            original,
            lambda: train_mod._make_logger(_cfg()),
        )
    assert caught.value is original
    assert len(wandb.init_calls) == 1
    assert wandb.run.finish_calls == 1
    assert wandb.run.config.updates == []
    _assert_raw_flock_reacquirable(lock_root, "run-id")


@pytest.mark.parametrize(
    "invalid_run",
    [None, object(), SimpleNamespace(finish=None, config={})],
    ids=["none", "missing-contract", "noncallable-finish"],
)
def test_resume_init_invalid_run_object_fails_closed_and_releases_lock(
    monkeypatch, lock_root, invalid_run
):
    wandb = _install(monkeypatch, FakeWandb())
    wandb.run = invalid_run
    with pytest.raises(RuntimeError, match="returned an invalid run object"):
        train_mod._make_logger(_cfg())
    assert len(wandb.init_calls) == 1
    _assert_raw_flock_reacquirable(lock_root, "run-id")


def test_finish_failure_still_releases_lock(monkeypatch, lock_root, capsys):
    wandb = _install(monkeypatch, FakeWandb(fail_finish=True))
    logger = train_mod._make_logger(_cfg())
    logger.close()
    replacement = train_mod._acquire_wandb_resume_lock(
        "run-id", state_root=str(lock_root)
    )
    replacement.release()
    assert "finish failed: RuntimeError" in capsys.readouterr().err


class FailingBackend:
    tokenizer = None

    def save(self, name):
        raise RuntimeError("training failed")


def test_training_failure_closes_logger_and_releases_lock(
    monkeypatch, lock_root
):
    wandb = _install(monkeypatch, FakeWandb())
    with pytest.raises(RuntimeError, match="training failed"):
        train_mod.train(object(), FailingBackend(), _cfg())
    replacement = train_mod._acquire_wandb_resume_lock(
        "run-id", state_root=str(lock_root)
    )
    replacement.release()
    assert wandb.run.finish_calls == 1


def test_operational_assignment_baseexception_still_closes_logger(monkeypatch):
    cfg = Config(
        launch_namespace="attempt-new",
        run_name="experiment",
        log_transcripts=True,
        steps=0,
        eval_every=0,
        save_every=0,
    )
    closed = []
    armed = set()

    class Logger:
        def close(self):
            closed.append(True)

    real_setattr = Config.__setattr__

    def injected_setattr(self, name, value):
        if id(self) in armed and name == "_transcript_run_name":
            raise BaseException("injected operational assignment failure")
        return real_setattr(self, name, value)

    def make_logger(active_cfg):
        armed.add(id(active_cfg))
        return Logger()

    monkeypatch.setattr(Config, "__setattr__", injected_setattr)
    monkeypatch.setattr(train_mod, "claim_directory", lambda path: path)
    monkeypatch.setattr(train_mod, "_make_logger", make_logger)

    with pytest.raises(BaseException, match="operational assignment failure"):
        train_mod.train(object(), FailingBackend(), cfg)
    assert closed == [True]


def test_direct_config_override_keyword_is_not_public():
    with pytest.raises(TypeError, match="wandb_resume_running_override"):
        Config(
            wandb_project="project",
            wandb_resume_running_override=True,
            launch_namespace="attempt-new",
        )

    with pytest.raises(TypeError, match="_wandb_resume_running_capability"):
        Config(_wandb_resume_running_capability=object())
    with pytest.raises(TypeError, match="_wandb_resume_lease_handoff"):
        Config(_wandb_resume_lease_handoff=object())


def test_direct_public_attr_boolean_tamper_refuses_before_wandb(
    monkeypatch, lock_root
):
    wandb = _install(monkeypatch, FakeWandb())
    cfg = _cfg()
    cfg.wandb_resume_running_override = True
    with pytest.raises(ValueError, match="not a public Config field"):
        train_mod._make_logger(cfg)
    assert wandb.api_run_calls == []
    assert wandb.init_calls == []


def test_direct_public_false_attr_also_refuses_instead_of_masking_tamper(
    monkeypatch, lock_root
):
    wandb = _install(monkeypatch, FakeWandb())
    cfg = _cfg()
    cfg.wandb_resume_running_override = False
    with pytest.raises(ValueError, match="not a public Config field"):
        train_mod._make_logger(cfg)
    assert wandb.api_run_calls == []


@pytest.mark.parametrize(
    "tamper", [False, True, object()], ids=["false-bool", "true-bool", "object"]
)
def test_direct_private_capability_tamper_refuses_before_wandb(
    monkeypatch, lock_root, tamper
):
    wandb = _install(monkeypatch, FakeWandb())
    cfg = _cfg()
    cfg._wandb_resume_running_capability = tamper
    with pytest.raises(ValueError, match="invalid or mismatched"):
        train_mod._make_logger(cfg)
    assert wandb.api_run_calls == []


def test_direct_private_lease_handoff_tamper_refuses_before_lock_or_api(
    monkeypatch, lock_root
):
    wandb = _install(monkeypatch, FakeWandb())
    cfg = _cfg()
    cfg._wandb_resume_lease_handoff = object()

    def forbidden_lock(*args, **kwargs):
        raise AssertionError("tampered handoff reached lock acquisition")

    monkeypatch.setattr(train_mod, "_acquire_wandb_resume_lock", forbidden_lock)
    with pytest.raises(ValueError, match="invalid W&B resume lease handoff"):
        train_mod._make_logger(cfg)
    assert wandb.api_run_calls == []


def test_runner_capability_is_bound_to_exact_run_id(monkeypatch, lock_root):
    wandb = _install(monkeypatch, FakeWandb())
    authorized = _cfg(running_override=True)
    try:
        cfg = _cfg(wandb_run_id="other-run-id")
        cfg._wandb_resume_running_capability = (
            authorized._wandb_resume_running_capability
        )
        with pytest.raises(ValueError, match="invalid or mismatched"):
            train_mod._make_logger(cfg)
        assert wandb.api_run_calls == []
    finally:
        release_wandb_resume_cli_lease(
            authorized._wandb_resume_lease_handoff
        )


def test_direct_config_resume_without_project_refuses_instead_of_becoming_fresh(
    monkeypatch, lock_root
):
    wandb = _install(monkeypatch, FakeWandb())
    with pytest.raises(ValueError, match="wandb_run_id requires W&B logging"):
        train_mod._make_logger(
            Config(
                wandb_run_id="run-id",
                launch_namespace="attempt-new",
            )
        )
    assert wandb.api_run_calls == []
    assert wandb.init_calls == []


def test_runner_capability_still_requires_enabled_project(monkeypatch, lock_root):
    wandb = _install(monkeypatch, FakeWandb())
    cfg = Config(
        wandb_run_id="run-id",
        launch_namespace="attempt-new",
    )
    args = SimpleNamespace(
        wandb_resume="run-id",
        wandb_resume_running_override=True,
        no_wandb=False,
        load="/checkpoints/step-00000",
        start_step=None,
    )
    handoff = acquire_wandb_resume_cli_lease(args)
    try:
        apply_wandb_resume_cli_authorization(
            cfg, args, resume_lease_handoff=handoff
        )
        with pytest.raises(ValueError, match="wandb_run_id requires W&B logging"):
            train_mod._make_logger(cfg)
        assert wandb.api_run_calls == []
    finally:
        release_wandb_resume_cli_lease(handoff)


def test_cli_override_requires_resume_flag():
    with pytest.raises(ValueError, match="requires --wandb-resume RUN_ID"):
        validate_resume_args(
            SimpleNamespace(
                wandb_resume=None,
                wandb_resume_running_override=True,
                no_wandb=False,
                load="/checkpoints/step-00010",
                start_step=None,
            )
        )


def test_cli_override_preserves_no_wandb_and_load_guards():
    with pytest.raises(ValueError, match="cannot be combined with --no-wandb"):
        validate_resume_args(
            SimpleNamespace(
                wandb_resume="run-id",
                wandb_resume_running_override=True,
                no_wandb=True,
                load="/checkpoints/step-00010",
                start_step=None,
            )
        )
    with pytest.raises(ValueError, match="requires --load"):
        validate_resume_args(
            SimpleNamespace(
                wandb_resume="run-id",
                wandb_resume_running_override=True,
                no_wandb=False,
                load=None,
                start_step=None,
            )
        )


@pytest.mark.parametrize("module_name", ["infra.run_debate", "infra.run_rlvr"])
def test_runtime_override_is_rejected_from_training_yaml(module_name):
    module = __import__(module_name, fromlist=["validate_experiment"])
    experiment = yaml.safe_load(
        """
        training:
          wandb_resume_running_override: true
        """
    )
    with pytest.raises(
        ValueError, match=r"unknown training key.*wandb_resume_running_override"
    ):
        module.validate_experiment(experiment)


def test_environment_variable_cannot_activate_cli_only_override(
    monkeypatch, tmp_path
):
    from infra.run_common import runner_parser, training_config_kwargs

    monkeypatch.setenv("WANDB_RESUME_RUNNING_OVERRIDE", "1")
    monkeypatch.setattr(
        train_mod, "_wandb_resume_lock_root", lambda: str(tmp_path / "locks")
    )
    args = runner_parser(None).parse_args(
        [
            "--experiment-file",
            "experiment.yaml",
            "--experiment",
            "arm",
            "--load",
            "/checkpoints/step-00010",
            "--wandb-resume",
            "run-id",
        ]
    )
    assert args.wandb_resume_running_override is False
    kwargs = training_config_kwargs({}, args)
    assert "wandb_resume_running_override" not in kwargs
    cfg = Config(**kwargs)
    handoff = acquire_wandb_resume_cli_lease(args)
    try:
        apply_wandb_resume_cli_authorization(
            cfg, args, resume_lease_handoff=handoff
        )
        assert cfg._wandb_resume_running_capability is None
    finally:
        release_wandb_resume_cli_lease(handoff)


@pytest.mark.parametrize("module_name", ["infra.run_debate", "infra.run_rlvr"])
def test_production_runner_frontier_refuses_contention_before_construction_api(
    monkeypatch, lock_root, module_name
):
    from infra.run_common import runner_parser

    module = __import__(module_name, fromlist=["_main"])
    wandb = _install(monkeypatch, FakeWandb())
    reached = []
    bound_configs = []

    class SetupFailure(BaseException):
        pass

    def args_for(run_id):
        return runner_parser(None).parse_args(
            [
                "--experiment-file",
                "experiment.yaml",
                "--experiment",
                "arm",
                "--load",
                "/checkpoints/step-00000",
                "--wandb-resume",
                run_id,
            ]
        )

    parsed_args = args_for("run-id")
    monkeypatch.setattr(
        module,
        "runner_parser",
        lambda description: SimpleNamespace(parse_args=lambda: parsed_args),
    )
    monkeypatch.setattr(
        module, "resolve_launch_namespace", lambda: "attempt-runner"
    )

    def backend_construction_sentinel(
        cleanups, args, launch_namespace, resume_lease_handoff
    ):
        reached.append("backend-construction")
        cfg = Config(
            wandb_project="project",
            wandb_run_id=args.wandb_resume,
            launch_namespace=launch_namespace,
        )
        apply_wandb_resume_cli_authorization(
            cfg, args, resume_lease_handoff=resume_lease_handoff
        )
        bound_configs.append(cfg)
        raise SetupFailure("injected setup failure")

    monkeypatch.setattr(
        module, "_main_after_resume_frontier", backend_construction_sentinel
    )

    held = train_mod._acquire_wandb_resume_lock(
        "run-id", state_root=str(lock_root)
    )
    try:
        with pytest.raises(RuntimeError, match="another local process"):
            module._main([])
        assert reached == []
        assert wandb.api_run_calls == []
        assert wandb.init_calls == []
    finally:
        held.release()

    # A different exact run ID does not contend and reaches construction. The
    # outer runner finally must release its early lease when setup fails.
    parsed_args = args_for("different-run-id")
    with pytest.raises(SetupFailure, match="injected setup failure"):
        module._main([])
    assert reached == ["backend-construction"]
    assert bound_configs[0]._wandb_resume_lease_handoff is None
    replacement = train_mod._acquire_wandb_resume_lock(
        "different-run-id", state_root=str(lock_root)
    )
    replacement.release()
    assert wandb.api_run_calls == []
    assert wandb.init_calls == []


def test_fresh_run_payload_never_contains_override_control_bit(
    monkeypatch, lock_root
):
    wandb = _install(monkeypatch, FakeWandb(stored_config={}))
    logger = train_mod._make_logger(
        Config(
            wandb_project="project",
            launch_namespace="attempt-new",
            protocol_identity=IDENTITY,
        )
    )
    try:
        assert "wandb_resume_running_override" not in wandb.init_calls[0]["config"]
        assert "_wandb_resume_running_capability" not in wandb.init_calls[0]["config"]
        assert "_wandb_resume_lease_handoff" not in wandb.init_calls[0]["config"]
    finally:
        logger.close()


def _lock_name(run_id: str) -> str:
    return hashlib.sha256(run_id.encode("utf-8")).hexdigest() + ".lock"


@pytest.mark.parametrize("control_flow", [KeyboardInterrupt, SystemExit])
def test_acquire_interrupt_after_fd_transfer_releases_raw_flock(
    lock_root, control_flow
):
    injected = control_flow("lock return interrupted")
    return_line = _source_line(
        train_mod._acquire_wandb_resume_lock, "return lease"
    )
    with pytest.raises(control_flow) as caught:
        _call_with_line_interrupt(
            train_mod._acquire_wandb_resume_lock,
            return_line,
            injected,
            lambda: train_mod._acquire_wandb_resume_lock(
                "run-id", state_root=str(lock_root)
            ),
        )
    assert caught.value is injected
    _assert_raw_flock_reacquirable(lock_root, "run-id")


@pytest.mark.parametrize("control_flow", [KeyboardInterrupt, SystemExit])
def test_handoff_take_interrupt_after_move_releases_raw_flock(
    monkeypatch, lock_root, control_flow
):
    wandb = _install(monkeypatch, FakeWandb())
    args = SimpleNamespace(
        wandb_resume="run-id",
        wandb_resume_running_override=False,
        no_wandb=False,
        load="/checkpoints/step-00000",
        start_step=None,
    )
    handoff = acquire_wandb_resume_cli_lease(args)
    cfg = _cfg()
    apply_wandb_resume_cli_authorization(
        cfg, args, resume_lease_handoff=handoff
    )
    injected = control_flow("handoff return interrupted")
    return_line = _source_line(type(handoff).take, "return lease")
    with pytest.raises(control_flow) as caught:
        _call_with_line_interrupt(
            type(handoff).take,
            return_line,
            injected,
            lambda: train_mod._make_logger(cfg),
        )
    assert caught.value is injected
    assert cfg._wandb_resume_lease_handoff is None
    assert wandb.api_run_calls == []
    release_wandb_resume_cli_lease(handoff)
    _assert_raw_flock_reacquirable(lock_root, "run-id")


@pytest.mark.parametrize("control_flow", [KeyboardInterrupt, SystemExit])
def test_handoff_factory_return_interrupt_releases_raw_flock(
    monkeypatch, lock_root, control_flow
):
    _install(monkeypatch, FakeWandb())
    injected = control_flow("handoff factory return interrupted")
    return_line = _source_line(
        train_mod._acquire_wandb_resume_lease_handoff, "return handoff"
    )
    with pytest.raises(control_flow) as caught:
        _call_with_line_interrupt(
            train_mod._acquire_wandb_resume_lease_handoff,
            return_line,
            injected,
            lambda: train_mod._acquire_wandb_resume_lease_handoff("run-id"),
        )
    assert caught.value is injected
    _assert_raw_flock_reacquirable(lock_root, "run-id")


def test_unstored_lock_and_handoff_temporaries_finalize_under_cpython(
    monkeypatch, lock_root
):
    assert sys.implementation.name == "cpython"

    train_mod._acquire_wandb_resume_lock(
        "bare-lock", state_root=str(lock_root)
    )
    gc.collect()
    _assert_raw_flock_reacquirable(lock_root, "bare-lock")

    _install(monkeypatch, FakeWandb())
    train_mod._acquire_wandb_resume_lease_handoff("run-id")
    gc.collect()
    _assert_raw_flock_reacquirable(lock_root, "run-id")


def test_default_lock_root_uses_account_database_not_home_environment(monkeypatch):
    monkeypatch.setenv("HOME", "/attacker-controlled-home")
    expected_home = pwd.getpwuid(os.geteuid()).pw_dir
    assert train_mod._wandb_resume_lock_root() == os.path.join(
        expected_home, ".local", "state", "debate", "wandb-resume-locks"
    )


def test_existing_lock_file_is_never_truncated(lock_root):
    lock_root.mkdir(mode=0o700)
    path = lock_root / _lock_name("run-id")
    path.write_bytes(b"persistent evidence")
    path.chmod(0o600)
    lease = train_mod._acquire_wandb_resume_lock(
        "run-id", state_root=str(lock_root)
    )
    lease.release()
    assert path.read_bytes() == b"persistent evidence"


@pytest.mark.parametrize("run_id", ["../escape", "a/b", " spaces ", "雪/☃", "a\0b"])
def test_run_id_is_hashed_and_never_used_as_a_path(lock_root, run_id):
    lease = train_mod._acquire_wandb_resume_lock(
        run_id, state_root=str(lock_root)
    )
    try:
        assert os.path.basename(lease.path) == _lock_name(run_id)
        assert os.path.dirname(lease.path) == str(lock_root)
    finally:
        lease.release()


def test_exact_run_ids_use_independent_locks(lock_root):
    first = train_mod._acquire_wandb_resume_lock("run", state_root=str(lock_root))
    second = train_mod._acquire_wandb_resume_lock("run ", state_root=str(lock_root))
    second.release()
    first.release()


def test_synchronized_independent_hosts_pin_residual_remote_race(
    tmp_path, monkeypatch
):
    roots = {
        "host-a": tmp_path / "host-a",
        "host-b": tmp_path / "host-b",
    }
    events = []
    event_lock = threading.Lock()
    state_barrier = threading.Barrier(2)
    init_barrier = threading.Barrier(2)
    runs = []

    def record(event):
        with event_lock:
            events.append(event)

    class BarrierPublicRun:
        config = {
            "protocol_identity": IDENTITY,
            "launch_namespaces": ["attempt-original"],
        }

        @property
        def state(self):
            host = threading.current_thread().name
            record(("state-enter", host))
            state_barrier.wait(timeout=5)
            record(("state-complete", host))
            # Neither property may return until BOTH completed observations
            # are in the trace. Production cannot reach init before return.
            state_barrier.wait(timeout=5)
            return "finished"

    class BarrierWandb:
        __version__ = "0.28.1"
        Settings = RealWandbSettings

        def Api(self):
            class Api:
                def run(self, path):
                    record(("api-run", threading.current_thread().name, path))
                    return BarrierPublicRun()

            return Api()

        def init(self, **kwargs):
            assert kwargs["mode"] == "online"
            host = threading.current_thread().name
            run = FakeRun(BarrierPublicRun.config)
            with event_lock:
                events.append(("init", host))
                runs.append(run)
            # Give both production paths a distinct initialized run before
            # either worker can close its logger.
            init_barrier.wait(timeout=5)
            return run

    monkeypatch.setattr(
        train_mod,
        "_wandb_resume_lock_root",
        lambda: str(roots[threading.current_thread().name]),
    )
    _install(monkeypatch, BarrierWandb())
    errors = []

    def worker(namespace):
        try:
            logger = train_mod._make_logger(_cfg(launch_namespace=namespace))
            logger.close()
        except BaseException as exc:
            with event_lock:
                errors.append(exc)

    workers = [
        threading.Thread(target=worker, name="host-a", args=("attempt-host-a",)),
        threading.Thread(target=worker, name="host-b", args=("attempt-host-b",)),
    ]
    for worker_thread in workers:
        worker_thread.start()
    for worker_thread in workers:
        worker_thread.join(timeout=10)

    assert all(not worker_thread.is_alive() for worker_thread in workers)
    assert errors == []
    first_init = next(i for i, event in enumerate(events) if event[0] == "init")
    completed = [
        i for i, event in enumerate(events) if event[0] == "state-complete"
    ]
    assert len(completed) == 2
    assert max(completed) < first_init
    assert sum(event[0] == "init" for event in events) == 2
    assert sum(event[0] == "api-run" for event in events) == 2
    assert len(runs) == 2 and runs[0] is not runs[1]
    assert [run.finish_calls for run in runs] == [1, 1]


def test_unsafe_lock_root_symlink_refuses(tmp_path):
    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    root = tmp_path / "locks"
    root.symlink_to(target, target_is_directory=True)
    with pytest.raises(RuntimeError, match="unsafe W&B resume lock directory"):
        train_mod._acquire_wandb_resume_lock("run-id", state_root=str(root))


def test_unsafe_lock_root_mode_refuses(tmp_path):
    root = tmp_path / "locks"
    root.mkdir(mode=0o755)
    root.chmod(0o755)
    with pytest.raises(RuntimeError, match="mode 0700"):
        train_mod._acquire_wandb_resume_lock("run-id", state_root=str(root))


@pytest.mark.parametrize("kind", ["symlink", "fifo", "hardlink", "mode"])
def test_unsafe_lock_leaf_refuses_without_blocking(tmp_path, kind):
    root = tmp_path / "locks"
    root.mkdir(mode=0o700)
    path = root / _lock_name("run-id")
    if kind == "symlink":
        target = tmp_path / "target"
        target.write_text("target")
        path.symlink_to(target)
    elif kind == "fifo":
        os.mkfifo(path, 0o600)
    elif kind == "hardlink":
        target = tmp_path / "target"
        target.write_text("target")
        target.chmod(0o600)
        os.link(target, path)
    else:
        path.write_text("lock")
        path.chmod(0o644)
    with pytest.raises(RuntimeError, match="W&B resume lock|unsafe W&B resume lock"):
        train_mod._acquire_wandb_resume_lock("run-id", state_root=str(root))


_HOLDER = r"""
import fcntl
import hashlib
import os
import sys
name = hashlib.sha256(sys.argv[2].encode("utf-8")).hexdigest() + ".lock"
fd = os.open(os.path.join(sys.argv[1], name), os.O_RDWR | os.O_CLOEXEC)
fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
print("READY", flush=True)
sys.stdin.readline()
os.close(fd)
"""


def _start_holder(root, run_id="run-id"):
    # Let the production helper safely create the persistent role, then use an
    # independent raw-flock process as the lock oracle.
    prepared = train_mod._acquire_wandb_resume_lock(
        run_id, state_root=str(root)
    )
    prepared.release()
    proc = subprocess.Popen(
        [sys.executable, "-c", _HOLDER, str(root), run_id],
        cwd=os.getcwd(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout.readline().strip() == "READY"
    return proc


def _release_holder(proc):
    proc.stdin.write("\n")
    proc.stdin.flush()
    stdout, stderr = proc.communicate(timeout=10)
    assert proc.returncode == 0, (stdout, stderr)


def test_real_subprocess_same_id_contends_and_different_id_does_not(lock_root):
    holder = _start_holder(lock_root)
    try:
        with pytest.raises(RuntimeError, match="another local process"):
            train_mod._acquire_wandb_resume_lock(
                "run-id", state_root=str(lock_root)
            )
        independent = train_mod._acquire_wandb_resume_lock(
            "other-run-id", state_root=str(lock_root)
        )
        independent.release()
    finally:
        _release_holder(holder)


def test_real_subprocess_normal_release_and_crash_release(lock_root):
    holder = _start_holder(lock_root)
    _release_holder(holder)
    lease = train_mod._acquire_wandb_resume_lock(
        "run-id", state_root=str(lock_root)
    )
    lease.release()

    crashed = _start_holder(lock_root)
    crashed.kill()
    crashed.wait(timeout=10)
    lease = train_mod._acquire_wandb_resume_lock(
        "run-id", state_root=str(lock_root)
    )
    lease.release()


_FORK_HOLDER = r"""
import os
import sys
import time
from infra.train import _acquire_wandb_resume_lock
lease = _acquire_wandb_resume_lock(sys.argv[2], state_root=sys.argv[1])
pid = os.fork()
if pid == 0:
    print("CHILD", flush=True)
    time.sleep(5)
    os._exit(0)
print("PARENT", flush=True)
sys.stdin.readline()
lease.release()
print("RELEASED", flush=True)
os.waitpid(pid, 0)
"""


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_fork_child_does_not_inherit_lock_lifetime(lock_root):
    proc = subprocess.Popen(
        [sys.executable, "-c", _FORK_HOLDER, str(lock_root), "run-id"],
        cwd=os.getcwd(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert {proc.stdout.readline().strip(), proc.stdout.readline().strip()} == {
            "CHILD",
            "PARENT",
        }
        proc.stdin.write("\n")
        proc.stdin.flush()
        assert proc.stdout.readline().strip() == "RELEASED"
        # The child is still sleeping. Acquisition succeeds only if the
        # after-fork hook closed its inherited descriptor.
        lease = train_mod._acquire_wandb_resume_lock(
            "run-id", state_root=str(lock_root)
        )
        lease.release()
    finally:
        proc.wait(timeout=10)


def test_created_lock_roles_have_private_modes_and_single_links(lock_root):
    lease = train_mod._acquire_wandb_resume_lock(
        "run-id", state_root=str(lock_root)
    )
    lease.release()
    root_info = lock_root.stat()
    leaf_info = (lock_root / _lock_name("run-id")).stat()
    assert stat.S_IMODE(root_info.st_mode) == 0o700
    assert stat.S_IMODE(leaf_info.st_mode) == 0o600
    assert leaf_info.st_nlink == 1
    assert root_info.st_uid == leaf_info.st_uid == os.geteuid()
