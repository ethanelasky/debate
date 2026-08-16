"""Required transcript evidence across real filesystem and training boundaries.

Coverage includes SingleTurnEnv record retention, exclusive local JSONL writes,
claimed-directory replacement defenses, public/private train-loop propagation,
final-eval persistence, and the best-effort external W&B artifact boundary.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infra.backend.base import SamplingParams
from infra.envs.base import Policy, SingleTurnEnv, Task, Trajectory
from infra.transcript_log import (
    SAMPLE_CHARS,
    flatten_messages,
    log_rollout_transcripts,
    singleturn_sample_rows,
)
from infra.launch_namespace import claim_directory, open_claimed_text_file
from test_single_turn_env import ScriptedBackend


class MetaEnv(SingleTurnEnv):
    """Tasks carry MB-shaped meta including the big 'bindings' payload."""

    def tasks(self, n, split="train"):
        return [
            Task(
                messages=[{"role": "user", "content": f"q{i}"}],
                meta={
                    "task_id": f"row_{i}",
                    "label": "attack",
                    "bindings": {"PROBLEM": "SECRET-TRAJECTORY" * 1000},
                },
            )
            for i in range(n)
        ]

    def reward(self, task, text):
        return 1.0, {
            "correct_strict": 1.0,
            "correct_relaxed": 1.0,
            "answer_format_valid": 1.0,
        }


def _policy(script):
    return Policy(ScriptedBackend(script), SamplingParams(max_tokens=32))


def test_rollout_retains_records_without_bindings():
    env = MetaEnv()
    env.rollout(env.tasks(2), _policy(["yes", "no"]), group_size=1)
    recs = env.last_rollout_records
    assert [r["completion"] for r in recs] == ["yes", "no"]
    assert [r["meta"]["task_id"] for r in recs] == ["row_0", "row_1"]
    # bindings duplicate the rendered prompt (and can be a 60k-token
    # trajectory); the retained meta must not carry them.
    assert all("bindings" not in r["meta"] for r in recs)
    # reward() 's own keys survive into the record. Subset rather than equality:
    # rollout() also stamps the length metrics (tokens, over_budget) that every
    # run reports, and pinning the exact dict here would make adding a metric
    # look like a regression in a test about bindings.
    assert all(
        r["reward"] == 1.0 and r["info"]["correct_strict"] == 1.0
        for r in recs
    )
    # each rollout overwrites, never appends
    env.rollout(env.tasks(1), _policy(["third"]), group_size=1)
    assert [r["completion"] for r in env.last_rollout_records] == ["third"]


def test_dropped_samples_leave_no_record():
    env = MetaEnv()
    env.rollout(env.tasks(2), _policy(["ok", ""]), group_size=1)  # "" = fidelity drop
    assert [r["completion"] for r in env.last_rollout_records] == ["ok"]


def test_singleturn_sample_rows_shape_and_truncation():
    records = [
        {
            "task_index": 0,
            "meta": {"task_id": "row_0"},
            "messages": [{"role": "user", "content": "x" * (2 * SAMPLE_CHARS)}],
            "completion": "ans",
            "stop_reason": "stop",
            "reward": 1.1,
            "info": {
                "correct_strict": 1.0,
                "correct_relaxed": 1.0,
                "answer_format_valid": 1.0,
            },
        }
    ]
    (row,) = singleturn_sample_rows(records, step=7)
    step, task, reward, info, transcript = row
    assert (step, task, reward) == (7, "row_0", 1.1)
    assert "truncated" in transcript and len(transcript) < 2 * SAMPLE_CHARS
    # cap rows at SAMPLE_ROWS
    assert len(singleturn_sample_rows(records * 10, step=0)) == 4


def test_flatten_messages():
    assert flatten_messages(
        [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    ) == "[system]\ns\n\n[user]\nu"


def test_logger_is_noop_without_wandb_run(tmp_path, monkeypatch):
    # No wandb.init in tests: wandb.run is None and the logger must return
    # before touching the filesystem.
    monkeypatch.chdir(tmp_path)
    env = MetaEnv()
    env.rollout(env.tasks(1), _policy(["a"]), group_size=1)
    log_rollout_transcripts(0, env, "train")
    assert not (tmp_path / "transcripts").exists()


def test_transcript_every_gates_upload_never_local_write(monkeypatch):
    import infra.train as train_mod
    import infra.transcript_log as tl

    calls = []
    monkeypatch.setattr(
        tl, "log_rollout_transcripts",
        lambda step, env, split, run_name=None, launch_namespace=None,
        output_dir=None, upload=True: calls.append(
            (step, split, launch_namespace, output_dir, upload)
        ),
    )
    cfg = train_mod.Config(
        wandb_project="p",
        transcript_every=10,
        run_name="r",
        launch_namespace="attempt",
        transcript_dir="transcripts/r/attempt",
    )
    cfg._transcript_run_name = "r"
    for s in (1, 10, 20):
        train_mod._log_transcripts(cfg, s, env=None, split="train")
    train_mod._log_transcripts(cfg, 7, env=None, split="eval")
    # every round reaches the writer (local persistence); upload is the only
    # thing the cadence gates
    assert calls == [
        (1, "train", "attempt", "transcripts/r/attempt", False),
        (10, "train", "attempt", "transcripts/r/attempt", True),
        (20, "train", "attempt", "transcripts/r/attempt", True),
        (7, "eval", "attempt", "transcripts/r/attempt", True),
    ]

    calls.clear()
    nowandb = train_mod.Config(
        wandb_project=None,
        transcript_every=10,
        run_name="r",
        launch_namespace="attempt",
        transcript_dir="transcripts/r/attempt",
    )
    nowandb._transcript_run_name = "r"
    train_mod._log_transcripts(nowandb, 3, env=None, split="train")
    assert calls == [
        (3, "train", "attempt", "transcripts/r/attempt", False)
    ]  # no wandb: local write still happens


def test_local_transcript_path_is_namespaced_and_refuses_overwrite(
    tmp_path, monkeypatch
):
    import wandb

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(wandb, "run", None)
    env = MetaEnv()
    env.rollout(env.tasks(1), _policy(["answer"]), group_size=1)
    output_dir = str(claim_directory("transcripts/run/attempt-7"))

    log_rollout_transcripts(
        3,
        env,
        "train",
        run_name="run",
        launch_namespace="attempt-7",
        output_dir=output_dir,
        upload=False,
    )
    path = tmp_path / "transcripts" / "run" / "attempt-7" / "train-step-00003.jsonl"
    assert path.is_file()
    original = path.read_bytes()

    with pytest.raises(FileExistsError):
        log_rollout_transcripts(
            3,
            env,
            "train",
            run_name="run",
            launch_namespace="attempt-7",
            output_dir=output_dir,
            upload=False,
        )
    assert path.read_bytes() == original


def test_transcript_writer_refuses_unclaimed_existing_directory(
    tmp_path, monkeypatch
):
    import wandb

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(wandb, "run", None)
    existing = tmp_path / "transcripts" / "run" / "attempt-unclaimed"
    existing.mkdir(parents=True)
    sentinel = existing / "sentinel"
    sentinel.write_bytes(b"existing evidence")
    env = MetaEnv()
    env.rollout(env.tasks(1), _policy(["answer"]), group_size=1)

    with pytest.raises(ValueError, match="not claimed by this process"):
        log_rollout_transcripts(
            3,
            env,
            "train",
            run_name="run",
            launch_namespace="attempt-unclaimed",
            output_dir="transcripts/run/attempt-unclaimed",
            upload=False,
        )

    assert sentinel.read_bytes() == b"existing evidence"
    assert sorted(path.name for path in existing.iterdir()) == ["sentinel"]


def test_transcript_writer_does_not_follow_replaced_claimed_ancestor(
    tmp_path, monkeypatch
):
    import wandb

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(wandb, "run", None)
    output_dir = "transcripts/run/attempt-replaced"
    claim_directory(output_dir)
    transcripts = tmp_path / "transcripts"
    retained = tmp_path / "transcripts-retained"
    transcripts.rename(retained)
    outside = tmp_path / "outside"
    outside.mkdir()
    transcripts.symlink_to(outside, target_is_directory=True)
    env = MetaEnv()
    env.rollout(env.tasks(1), _policy(["answer"]), group_size=1)

    with pytest.raises(ValueError, match="no longer safely reachable"):
        log_rollout_transcripts(
            4,
            env,
            "train",
            run_name="run",
            launch_namespace="attempt-replaced",
            output_dir=output_dir,
            upload=False,
        )

    assert list(outside.iterdir()) == []
    assert list((retained / "run" / "attempt-replaced").iterdir()) == []


def test_wandb_transcript_artifact_carries_scalar_namespace_and_internal_path(
    tmp_path, monkeypatch
):
    import wandb

    monkeypatch.chdir(tmp_path)
    artifacts = []

    class Artifact:
        def __init__(self, name, *, type, metadata):
            self.name = name
            self.type = type
            self.metadata = metadata
            self.files = []
            artifacts.append(self)

        def add_file(self, path, *, name):
            self.files.append((path, name))

    run = SimpleNamespace(
        name="display-name",
        id="wandb-id",
        log_artifact=lambda artifact: None,
        log=lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(wandb, "run", run)
    monkeypatch.setattr(wandb, "Artifact", Artifact)
    monkeypatch.setattr(
        wandb, "Table", lambda *, columns, data: SimpleNamespace(columns=columns, data=data)
    )
    env = MetaEnv()
    env.rollout(env.tasks(1), _policy(["answer"]), group_size=1)
    output_dir = str(claim_directory("transcripts/run/attempt-8"))

    log_rollout_transcripts(
        4,
        env,
        "eval",
        run_name="run",
        launch_namespace="attempt-8",
        output_dir=output_dir,
        upload=True,
    )

    assert artifacts[0].metadata["launch_namespace"] == "attempt-8"
    assert artifacts[0].metadata["run"] == "run"
    assert artifacts[0].files[0][1] == "run/attempt-8/eval-step-00004.jsonl"


def test_wandb_upload_failure_is_best_effort_after_local_success(
    tmp_path, monkeypatch, capsys
):
    import wandb

    monkeypatch.chdir(tmp_path)

    class Artifact:
        def __init__(self, name, *, type, metadata):
            pass

        def add_file(self, path, *, name):
            raise RuntimeError("simulated external failure")

    monkeypatch.setattr(
        wandb,
        "run",
        SimpleNamespace(name="display", id="wandb-id"),
    )
    monkeypatch.setattr(wandb, "Artifact", Artifact)
    env = MetaEnv()
    env.rollout(env.tasks(1), _policy(["answer"]), group_size=1)
    output_dir = str(claim_directory("transcripts/run/attempt-upload-failure"))

    log_rollout_transcripts(
        5,
        env,
        "train",
        run_name="run",
        launch_namespace="attempt-upload-failure",
        output_dir=output_dir,
        upload=True,
    )

    path = (
        tmp_path
        / "transcripts"
        / "run"
        / "attempt-upload-failure"
        / "train-step-00005.jsonl"
    )
    assert path.is_file() and path.read_bytes()
    assert "W&B upload failed: RuntimeError" in capsys.readouterr().err


def test_real_train_loop_aborts_on_required_local_transcript_failure(
    tmp_path, monkeypatch
):
    import infra.train as train_mod

    monkeypatch.chdir(tmp_path)
    source = MetaEnv()
    source.rollout(source.tasks(1), _policy(["answer"]), group_size=1)

    class LoopEnv:
        last_rollout_records = source.last_rollout_records

        def tasks(self, n, split="train"):
            return [object()]

        def rollout(self, tasks, policy, group_size):
            return [[Trajectory(datums=[], reward=1.0)]]

    class LoopBackend:
        tokenizer = object()

        def __init__(self):
            self.saved = []

        def sync_sampler(self):
            pass

        def save(self, name):
            self.saved.append(name)

    transcript_dir = str(
        claim_directory("transcripts/run/attempt-local-failure")
    )
    with open_claimed_text_file(
        transcript_dir, "train-step-00003.jsonl"
    ) as handle:
        handle.write("existing transcript evidence\n")
    existing = Path(transcript_dir) / "train-step-00003.jsonl"
    original = existing.read_bytes()
    cfg = train_mod.Config(
        start_step=3,
        steps=4,
        batch_size=1,
        group_size=1,
        eval_every=0,
        save_every=0,
        run_name="run",
        launch_namespace="attempt-local-failure",
        transcript_dir=transcript_dir,
    )
    cfg._transcript_run_name = "run"
    backend = LoopBackend()
    logged = []

    with pytest.raises(FileExistsError, match="refusing existing launch output"):
        train_mod._train_with_logger(
            LoopEnv(), backend, cfg, None, lambda *args: logged.append(args)
        )

    assert existing.read_bytes() == original
    assert logged == []
    assert backend.saved == []


class _FinalEvalBackend(ScriptedBackend):
    def __init__(self, script):
        super().__init__(script)
        self.saved = []

    def save(self, name):
        self.saved.append(name)
        return name

    def forward_backward(self, data, loss):
        return {}

    def optim_step(self, params):
        return {}


@pytest.mark.parametrize(
    ("eval_split", "periodic_prefix"),
    [("dev", "dev"), ("test", "eval")],
)
def test_final_periodic_and_final_test_evals_persist_distinct_transcripts(
    tmp_path, monkeypatch, eval_split, periodic_prefix
):
    import infra.train as train_mod

    monkeypatch.chdir(tmp_path)
    eval_env = MetaEnv()
    training_env = SimpleNamespace()
    backend = _FinalEvalBackend(["periodic answer", "test answer"])
    namespace = f"final-evals-{eval_split}"
    transcript_dir = str(claim_directory(f"transcripts/run/{namespace}"))
    cfg = train_mod.Config(
        steps=0,
        eval_every=1,
        eval_n=1,
        eval_split=eval_split,
        final_test_eval=True,
        save_every=0,
        run_name="run",
        launch_namespace=namespace,
        transcript_dir=transcript_dir,
    )
    cfg._transcript_run_name = "run"
    logged = []

    train_mod._train_with_logger(
        training_env, backend, cfg, eval_env, lambda *args: logged.append(args)
    )

    periodic = Path(transcript_dir) / f"{periodic_prefix}-step-00000.jsonl"
    test = Path(transcript_dir) / "test-step-00000.jsonl"
    assert periodic.is_file() and test.is_file()
    assert b"periodic answer" in periodic.read_bytes()
    assert b"test answer" in test.read_bytes()
    assert [step for step, _ in logged] == [0, 0]
    assert f"{periodic_prefix}/n" in logged[0][1]
    assert "test/n" in logged[1][1]
    assert backend.saved == ["final"]


@pytest.mark.parametrize(
    ("eval_every", "final_test_eval", "eval_split", "prefix"),
    [(1, False, "dev", "dev"), (0, True, "test", "test")],
)
def test_final_eval_transcript_failure_prevents_metric_log_and_final_save(
    tmp_path,
    monkeypatch,
    eval_every,
    final_test_eval,
    eval_split,
    prefix,
):
    import infra.train as train_mod

    monkeypatch.chdir(tmp_path)
    eval_env = MetaEnv()
    training_env = SimpleNamespace()
    backend = _FinalEvalBackend(["answer"])
    namespace = f"final-failure-{prefix}"
    transcript_dir = str(claim_directory(f"transcripts/run/{namespace}"))
    filename = f"{prefix}-step-00000.jsonl"
    with open_claimed_text_file(transcript_dir, filename) as handle:
        handle.write("existing evidence\n")
    existing = Path(transcript_dir) / filename
    original = existing.read_bytes()
    cfg = train_mod.Config(
        steps=0,
        eval_every=eval_every,
        eval_n=1,
        eval_split=eval_split,
        final_test_eval=final_test_eval,
        save_every=0,
        run_name="run",
        launch_namespace=namespace,
        transcript_dir=transcript_dir,
    )
    cfg._transcript_run_name = "run"
    logged = []

    with pytest.raises(FileExistsError, match="refusing existing launch output"):
        train_mod._train_with_logger(
            training_env,
            backend,
            cfg,
            eval_env,
            lambda *args: logged.append(args),
        )

    assert existing.read_bytes() == original
    assert logged == []
    assert backend.saved == []


def test_real_train_loop_fails_closed_on_empty_retained_rollout(
    tmp_path, monkeypatch
):
    import infra.train as train_mod

    monkeypatch.chdir(tmp_path)

    class EmptyRetainedEnv:
        last_rollout_records = []

        def tasks(self, n, split="train"):
            return [object()]

        def rollout(self, tasks, policy, group_size):
            self.last_rollout_records = []
            return [[Trajectory(datums=[], reward=1.0)]]

    backend = _ZeroStepBackend()
    backend.tokenizer = object()
    backend.sync_sampler = lambda: None
    transcript_dir = str(claim_directory("transcripts/run/empty-retained"))
    cfg = train_mod.Config(
        steps=1,
        batch_size=1,
        group_size=1,
        eval_every=0,
        save_every=0,
        run_name="run",
        launch_namespace="empty-retained",
        transcript_dir=transcript_dir,
    )
    cfg._transcript_run_name = "run"

    with pytest.raises(RuntimeError, match="retained no transcript records"):
        train_mod._train_with_logger(
            EmptyRetainedEnv(), backend, cfg, None, lambda *args: None
        )

    assert list(Path(transcript_dir).iterdir()) == []
    assert backend.saved == []


def test_fresh_wandb_init_failure_preserves_local_training_and_evidence(
    tmp_path, monkeypatch, capsys
):
    import infra.train as train_mod

    monkeypatch.chdir(tmp_path)

    def fail_init(**kwargs):
        raise RuntimeError("simulated W&B outage")

    fake_wandb = SimpleNamespace(run=None, init=fail_init)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setattr(train_mod, "_env_identity", lambda: {"env/git_dirty": "no"})
    env = MetaEnv()
    backend = _FinalEvalBackend(["answer"])
    cfg = train_mod.Config(
        steps=1,
        batch_size=1,
        group_size=1,
        eval_every=0,
        save_every=0,
        run_name="run",
        launch_namespace="wandb-init-outage",
        wandb_project="project",
    )

    train_mod.train(env, backend, cfg)

    transcript = (
        tmp_path
        / "transcripts"
        / "run"
        / "wandb-init-outage"
        / "train-step-00000.jsonl"
    )
    assert transcript.is_file() and b"answer" in transcript.read_bytes()
    assert backend.saved == ["final"]
    assert "fresh init failed: RuntimeError" in capsys.readouterr().err


class _ZeroStepBackend:
    tokenizer = None

    def __init__(self):
        self.saved = []

    def save(self, name):
        self.saved.append(name)


def test_config_without_run_name_claims_once_and_reuses_for_later_steps(
    tmp_path, monkeypatch
):
    import wandb
    import infra.train as train_mod

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(wandb, "run", None)
    env = MetaEnv()
    env.rollout(env.tasks(1), _policy(["answer"]), group_size=1)
    backend = _ZeroStepBackend()
    cfg = train_mod.Config(
        steps=0,
        eval_every=0,
        save_every=0,
        run_name=None,
        launch_namespace="direct-cli-attempt",
    )

    train_mod.train(env, backend, cfg)
    assert cfg.transcript_dir == "transcripts/run/direct-cli-attempt"
    claimed = cfg.transcript_dir
    train_mod._log_transcripts(cfg, 1, env, "train")
    train_mod._log_transcripts(cfg, 2, env, "train")

    assert claimed == "transcripts/run/direct-cli-attempt"
    assert cfg.transcript_dir == claimed
    assert (
        tmp_path
        / "transcripts"
        / "run"
        / "direct-cli-attempt"
        / "train-step-00001.jsonl"
    ).is_file()
    assert (
        tmp_path
        / "transcripts"
        / "run"
        / "direct-cli-attempt"
        / "train-step-00002.jsonl"
    ).is_file()
    assert backend.saved == ["final"]
    with pytest.raises(FileExistsError, match="already consumed"):
        train_mod.train(env, _ZeroStepBackend(), cfg)


def test_config_cannot_adopt_preexisting_transcript_directory(tmp_path, monkeypatch):
    import wandb
    import infra.train as train_mod

    monkeypatch.chdir(tmp_path)
    init_calls = []
    monkeypatch.setattr(wandb, "init", lambda **kwargs: init_calls.append(kwargs))
    existing = tmp_path / "transcripts" / "run" / "attempt-existing"
    existing.mkdir(parents=True)
    cfg = train_mod.Config(
        steps=0,
        eval_every=0,
        save_every=0,
        run_name=None,
        launch_namespace="attempt-existing",
        transcript_dir="transcripts/run/attempt-existing",
        wandb_project="project",
    )

    with pytest.raises(ValueError, match="not claimed by this process"):
        train_mod.train(MetaEnv(), _ZeroStepBackend(), cfg)
    assert init_calls == []


def test_direct_wandb_run_preclaims_fallback_and_reuses_it_for_two_steps(
    tmp_path, monkeypatch
):
    import wandb
    import infra.train as train_mod

    monkeypatch.chdir(tmp_path)
    env = MetaEnv()
    env.rollout(env.tasks(1), _policy(["answer"]), group_size=1)
    artifacts = []
    init_calls = []

    class Artifact:
        def __init__(self, name, *, type, metadata):
            self.metadata = metadata
            self.files = []
            artifacts.append(self)

        def add_file(self, path, *, name):
            self.files.append((path, name))

    class Run:
        name = "wandb-assigned-display-name"
        id = "wandb-id"
        dir = str(tmp_path / "wandb-run")

        def __init__(self, config):
            self.config = dict(config)
            self.finish_calls = 0

        def log(self, *args, **kwargs):
            pass

        def log_artifact(self, artifact):
            pass

        def finish(self):
            self.finish_calls += 1

    run = None

    def init(**kwargs):
        nonlocal run
        # Reservation is a precondition of opening external W&B state.
        assert (tmp_path / "transcripts" / "run" / "wandb-attempt").is_dir()
        init_calls.append(kwargs)
        run = Run(kwargs["config"])
        monkeypatch.setattr(wandb, "run", run)
        return run

    def two_steps(env_, backend_, cfg_, eval_env_, logger_):
        train_mod._log_transcripts(cfg_, 1, env_, "train")
        first_dir = cfg_.transcript_dir
        train_mod._log_transcripts(cfg_, 2, env_, "train")
        assert cfg_.transcript_dir == first_dir

    monkeypatch.setattr(wandb, "run", None)
    monkeypatch.setattr(wandb, "init", init)
    monkeypatch.setattr(wandb, "Artifact", Artifact)
    monkeypatch.setattr(
        wandb, "Table", lambda *, columns, data: SimpleNamespace(columns=columns, data=data)
    )
    monkeypatch.setattr(train_mod, "_env_identity", lambda: {"env/git_dirty": "no"})
    monkeypatch.setattr(train_mod, "_train_with_logger", two_steps)
    cfg = train_mod.Config(
        run_name=None,
        wandb_project="project",
        transcript_every=1,
        launch_namespace="wandb-attempt",
    )

    train_mod.train(env, _ZeroStepBackend(), cfg)

    assert cfg.run_name is None
    assert cfg.transcript_dir == "transcripts/run/wandb-attempt"
    assert init_calls[0]["name"] is None
    assert "_transcript_run_name" not in init_calls[0]["config"]
    assert "_transcript_sink_used" not in init_calls[0]["config"]
    assert [artifact.metadata["launch_namespace"] for artifact in artifacts] == [
        "wandb-attempt",
        "wandb-attempt",
    ]
    assert [artifact.files[0][1] for artifact in artifacts] == [
        "run/wandb-attempt/train-step-00001.jsonl",
        "run/wandb-attempt/train-step-00002.jsonl",
    ]
    assert run is not None and run.finish_calls == 1
