"""Immutable task/grader protocol identity in W&B run config."""

import copy
import sys
from types import SimpleNamespace

import pytest

import infra.train as train_mod
from infra.train import Config, resolve_protocol_identity, validate_resume_args


class IdentityFamily:
    def protocol_identity(self):
        return {"grading_protocol": "test-grader-v1"}


def _runner_args(**overrides):
    values = {
        "batch_size": None,
        "group_size": None,
        "lr": None,
        "steps": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _rlvr_exp():
    return {
        "model": "org/model",
        "enable_thinking": False,
        "max_completion_tokens": 512,
        "temperature": 0.8,
        "top_p": 0.95,
        "min_completion_tokens": 8,
        "dataset": {"type": "math"},
        "training": {
            "backend": "verl",
            "batch_size": 4,
            "group_size": 8,
            "dynamic_sampling_retries": 2,
            "eval_every": 5,
            "eval_n": 32,
            "eval_split": "dev",
            "eval_max_tokens": 600,
            "final_test_eval": True,
            "rl_seed": 7,
        },
    }


def _debate_exp(prompt_file):
    return {
        "protocol": {
            "turns": [
                {
                    "alice": [
                        {
                            "name": "proposal",
                            "kind": "solution",
                            "max_total_tokens": 256,
                        }
                    ]
                },
                {
                    "judge": [
                        {
                            "name": "verdict",
                            "kind": "decision",
                            "max_total_tokens": 128,
                        }
                    ]
                },
            ]
        },
        "prompt_config": {"file_path": str(prompt_file), "entry": "debate"},
        "agents": {
            "alice": {
                "trained": True,
                "model_settings": {
                    "model_type": "local",
                    "model_file_path": "org/policy",
                    "alias": "Alice",
                    "enable_thinking": False,
                    "sampling": {"train": {"temperature": 1.0, "top_p": 1.0}},
                },
            },
            "judge": {
                "model_settings": {
                    "model_type": "openrouter",
                    "model_file_path": "org/judge",
                    "alias": "Judge",
                    "sampling": {"train": {"temperature": 0.6, "top_p": 0.9}},
                    "provider_order": ["provider-a"],
                }
            },
        },
        "judge_config": {"schema_name": "competitive", "retries": 4},
        "scoring": {
            "scoring": "continuous",
            "confidence_source": "json",
            "shaping": [
                {
                    "kind": "format_reward",
                    "flag": "answer_format_valid",
                    "coeff": 0.1,
                }
            ],
        },
        "dataset": {"type": "math", "relaxed_extraction": True},
        "fresh_positions": False,
        "flip": False,
        "first_speech_non_debate_aware": False,
        "training": {
            "backend": "verl",
            "batch_size": 4,
            "group_size": 4,
            "eval_every": 5,
            "eval_n": 16,
            "eval_split": "dev",
            "eval_max_tokens": 512,
            "final_test_eval": True,
            "rl_seed": 3,
        },
    }


def _debate_identity(exp, *, args=None, topology=None):
    import infra.run_debate as run_debate

    trained, frozen = run_debate.split_agents(exp)
    return run_debate.debate_protocol_identity(
        exp, "math", IdentityFamily(), trained, frozen, args=args, topology=topology
    )


class FakeConfig(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.updates = []

    def update(self, values, *, allow_val_change=False):
        self.updates.append((dict(values), allow_val_change))
        super().update(values)


class FakeRun:
    def __init__(self, config):
        self.config = FakeConfig(config)
        self.logged = []
        self.finish_calls = 0

    def log(self, metrics, step):
        self.logged.append((step, metrics))

    def finish(self):
        self.finish_calls += 1


class FakeWandb:
    def __init__(self, run, stored_config):
        self.run = run
        self.stored_config = stored_config
        self.init_calls = []
        self.api_run_calls = []

    def Api(self):
        owner = self

        class Api:
            def run(self, path):
                owner.api_run_calls.append(path)
                return SimpleNamespace(config=owner.stored_config)

        return Api()

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        return self.run


def _install_fake_wandb(monkeypatch, stored_config):
    run = FakeRun(stored_config)
    wandb = FakeWandb(run, stored_config)
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    monkeypatch.setattr(train_mod, "_env_identity", lambda: {"env/git_dirty": "no"})
    return wandb, run


def test_config_protocol_identity_defaults_are_independent():
    left = Config()
    right = Config()
    left.protocol_identity["dataset_type"] = "math"
    assert right.protocol_identity == {}


def test_resolve_protocol_identity_keeps_runner_registry_type():
    family = SimpleNamespace(protocol_identity=lambda: {"grading_protocol": "symbolic-v1"})
    assert resolve_protocol_identity("math_symbolic", family) == {
        "dataset_type": "math_symbolic",
        "grading_protocol": "symbolic-v1",
    }


def test_resolve_protocol_identity_allows_empty_family_identity():
    family = SimpleNamespace(protocol_identity=lambda: {})
    assert resolve_protocol_identity("math", family) == {"dataset_type": "math"}


def test_resolve_protocol_identity_rejects_family_dataset_type_collision():
    family = SimpleNamespace(protocol_identity=lambda: {"dataset_type": "math"})
    with pytest.raises(ValueError, match="reserved key 'dataset_type'"):
        resolve_protocol_identity("math_symbolic", family)


@pytest.mark.parametrize("dataset_type", [None, "", "   ", 3])
def test_resolve_protocol_identity_requires_nonempty_string_dataset_type(dataset_type):
    family = SimpleNamespace(protocol_identity=lambda: {})
    with pytest.raises(ValueError, match="nonempty string"):
        resolve_protocol_identity(dataset_type, family)


@pytest.mark.parametrize(
    "identity",
    [
        [],
        {1: "value"},
        {"key": 1},
    ],
)
def test_resolve_protocol_identity_requires_string_dict(identity):
    family = SimpleNamespace(protocol_identity=lambda: identity)
    with pytest.raises(ValueError, match=r"dict\[str, str\]"):
        resolve_protocol_identity("math", family)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("model",), "org/other-model"),
        (("enable_thinking",), True),
        (("max_completion_tokens",), 513),
        (("temperature",), 0.7),
        (("top_p",), 0.9),
        (("min_completion_tokens",), 9),
        (("plan_tokens",), 128),
        (("training", "batch_size"), 5),
        (("training", "group_size"), 9),
        (("training", "dynamic_sampling_retries"), 3),
        (("training", "eval_n"), 33),
        (("training", "eval_split"), "test"),
        (("training", "eval_max_tokens"), 601),
        (("training", "final_test_eval"), False),
        (("training", "rl_seed"), 8),
    ],
)
def test_rlvr_runner_identity_tracks_rollout_and_eval_protocol(path, value):
    from infra.run_rlvr import rlvr_protocol_identity

    exp = _rlvr_exp()
    baseline = rlvr_protocol_identity(exp, "math", IdentityFamily())
    changed = copy.deepcopy(exp)
    target = changed
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    assert rlvr_protocol_identity(changed, "math", IdentityFamily()) != baseline
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in baseline.items())


def test_rlvr_runner_identity_tracks_cli_grouping_but_not_continuation_knobs():
    from infra.run_rlvr import rlvr_protocol_identity

    exp = _rlvr_exp()
    baseline = rlvr_protocol_identity(exp, "math", IdentityFamily(), args=_runner_args())
    assert (
        rlvr_protocol_identity(
            exp, "math", IdentityFamily(), args=_runner_args(group_size=9)
        )
        != baseline
    )

    mutable = copy.deepcopy(exp)
    mutable["training"].update(
        {
            "steps": 999,
            "lr": 4e-6,
            "save_every": 1,
            "eval_every": 1,
            "log_transcripts": False,
            "wandb_project": "renamed",
            "wandb_entity": "different-team",
            "verl": {"checkpoint_dir": "/different/output"},
        }
    )
    assert (
        rlvr_protocol_identity(
            mutable,
            "math",
            IdentityFamily(),
            args=_runner_args(lr=9e-7, steps=1000),
        )
        == baseline
    )


def test_rlvr_runner_identity_rejects_secret_bearing_model_url():
    from infra.run_rlvr import rlvr_protocol_identity

    exp = _rlvr_exp()
    exp["model"] = "https://user:secret@example.test/model?token=secret"
    with pytest.raises(ValueError, match="may not contain URL credentials"):
        rlvr_protocol_identity(exp, "math", IdentityFamily())


@pytest.mark.parametrize(
    "mutate",
    [
        lambda exp: exp["protocol"]["turns"][0]["alice"][0].update(max_total_tokens=300),
        lambda exp: exp["prompt_config"].update(entry="other-entry"),
        lambda exp: exp["dataset"].update(relaxed_extraction=False),
        lambda exp: exp["judge_config"].update(retries=5),
        lambda exp: exp["scoring"].update(scoring="binary"),
        lambda exp: exp["scoring"]["shaping"][0].update(coeff=0.2),
        lambda exp: exp.update(fresh_positions=True),
        lambda exp: exp.update(flip=True),
        lambda exp: exp.update(first_speech_non_debate_aware=True),
        lambda exp: exp.update(plan_tokens=100),
        lambda exp: exp["agents"]["alice"]["model_settings"].update(
            model_file_path="org/other-policy"
        ),
        lambda exp: exp["agents"]["judge"]["model_settings"]["sampling"]["train"].update(
            temperature=0.7
        ),
        lambda exp: exp["training"].update(group_size=5),
        lambda exp: exp["training"].update(eval_split="test"),
        lambda exp: exp["training"].update(eval_max_tokens=600),
    ],
)
def test_debate_runner_identity_tracks_scientific_protocol(tmp_path, mutate):
    prompt = tmp_path / "prompts.yaml"
    prompt.write_text("debate:\n  system: prompt one\n")
    exp = _debate_exp(prompt)
    baseline = _debate_identity(exp)
    changed = copy.deepcopy(exp)
    mutate(changed)

    assert _debate_identity(changed) != baseline
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in baseline.items())


def test_debate_runner_identity_tracks_prompt_bytes(tmp_path):
    prompt = tmp_path / "prompts.yaml"
    prompt.write_text("debate:\n  system: prompt one\n")
    exp = _debate_exp(prompt)
    baseline = _debate_identity(exp)

    prompt.write_text("debate:\n  system: prompt two\n")
    changed = _debate_identity(exp)
    assert changed["runner_prompt_sha256"] != baseline["runner_prompt_sha256"]
    assert changed["runner_protocol_sha256"] != baseline["runner_protocol_sha256"]


def test_debate_runner_identity_ignores_continuation_and_output_knobs(tmp_path):
    prompt = tmp_path / "prompts.yaml"
    prompt.write_text("debate:\n  system: prompt\n")
    exp = _debate_exp(prompt)
    baseline = _debate_identity(exp, args=_runner_args())

    mutable = copy.deepcopy(exp)
    mutable["training"].update(
        {
            "steps": 1000,
            "lr": 3e-6,
            "save_every": 1,
            "eval_every": 1,
            "log_transcripts": False,
            "wandb_project": "renamed",
            "wandb_entity": "different-team",
            "verl": {"checkpoint_dir": "/different/output"},
        }
    )
    assert _debate_identity(
        mutable, args=_runner_args(lr=2e-6, steps=2000)
    ) == baseline


@pytest.mark.parametrize(
    "mutate",
    [
        lambda tr: tr.update(kl_coef=0.05),
        lambda tr: tr.update(ppo_epochs=2),
        lambda tr: tr.update(norm_adv_by_std=False),
        lambda tr: tr.update(loss={"kind": "importance_sampling"}),
        lambda tr: tr.update(loss={"kind": "ppo", "clip_high": 1.3}),
        lambda tr: tr.update(backend="tinker"),
        lambda tr: tr.setdefault("verl", {}).update(strategy="megatron"),
        lambda tr: tr.update(micro_batch=8),
        lambda tr: tr.update(warmup_steps=10),
    ],
)
def test_both_runner_identities_track_learning_algorithm(tmp_path, mutate):
    """A resume may not cross an optimizer/backend protocol boundary."""
    from infra.run_rlvr import rlvr_protocol_identity

    prompt = tmp_path / "prompts.yaml"
    prompt.write_text("debate:\n  system: prompt\n")
    rlvr = _rlvr_exp()
    debate = _debate_exp(prompt)
    rlvr_baseline = rlvr_protocol_identity(rlvr, "math", IdentityFamily())
    debate_baseline = _debate_identity(debate)

    mutate(rlvr["training"])
    mutate(debate["training"])

    assert rlvr_protocol_identity(rlvr, "math", IdentityFamily()) != rlvr_baseline
    assert _debate_identity(debate) != debate_baseline


def test_runner_identity_redacts_secret_and_absolute_output_overrides(tmp_path):
    """Sensitive/operational values must not even influence the stored hash."""
    from infra.run_rlvr import rlvr_protocol_identity

    prompt = tmp_path / "prompts.yaml"
    prompt.write_text("debate:\n  system: prompt\n")
    left_rlvr = _rlvr_exp()
    left_debate = _debate_exp(prompt)
    overrides = [
        "++service.api_key=first-secret",
        "++trainer.output_dir=/private/first-output",
    ]
    left_rlvr["training"]["verl"] = {"extra_overrides": overrides}
    left_debate["training"]["verl"] = {"extra_overrides": overrides}

    right_rlvr = copy.deepcopy(left_rlvr)
    right_debate = copy.deepcopy(left_debate)
    right_rlvr["training"]["verl"]["extra_overrides"] = [
        "++service.api_key=second-secret",
        "++trainer.output_dir=/private/second-output",
    ]
    right_debate["training"]["verl"]["extra_overrides"] = [
        "++service.api_key=second-secret",
        "++trainer.output_dir=/private/second-output",
    ]

    assert rlvr_protocol_identity(left_rlvr, "math", IdentityFamily()) == (
        rlvr_protocol_identity(right_rlvr, "math", IdentityFamily())
    )
    assert _debate_identity(left_debate) == _debate_identity(right_debate)


@pytest.mark.parametrize(
    "topology",
    [
        {"n_gpus": 2, "rollout_tp": 2},
        {
            "extra_overrides": [
                "++actor_rollout_ref.actor.policy_loss.loss_mode=gspo"
            ]
        },
    ],
)
def test_both_runner_identities_track_topology_derived_immutable_fields(
    tmp_path, topology
):
    from infra.run_rlvr import rlvr_protocol_identity

    prompt = tmp_path / "prompts.yaml"
    prompt.write_text("debate:\n  system: prompt\n")
    rlvr = _rlvr_exp()
    debate = _debate_exp(prompt)

    assert rlvr_protocol_identity(
        rlvr, "math", IdentityFamily(), topology=topology
    ) != rlvr_protocol_identity(rlvr, "math", IdentityFamily(), topology={})
    assert _debate_identity(debate, topology=topology) != _debate_identity(
        debate, topology={}
    )


def test_topology_capacity_and_output_fields_remain_operational(tmp_path):
    from infra.run_rlvr import rlvr_protocol_identity

    prompt = tmp_path / "prompts.yaml"
    prompt.write_text("debate:\n  system: prompt\n")
    rlvr = _rlvr_exp()
    debate = _debate_exp(prompt)
    left = {"gpu_memory_utilization": 0.4, "checkpoint_dir": "/first"}
    right = {"gpu_memory_utilization": 0.8, "checkpoint_dir": "/second"}

    assert rlvr_protocol_identity(
        rlvr, "math", IdentityFamily(), topology=left
    ) == rlvr_protocol_identity(rlvr, "math", IdentityFamily(), topology=right)
    assert _debate_identity(debate, topology=left) == _debate_identity(
        debate, topology=right
    )


def test_arm_verl_values_override_topology_in_both_runner_identities(tmp_path):
    from infra.run_rlvr import rlvr_protocol_identity

    prompt = tmp_path / "prompts.yaml"
    prompt.write_text("debate:\n  system: prompt\n")
    rlvr = _rlvr_exp()
    debate = _debate_exp(prompt)
    for exp in (rlvr, debate):
        exp["training"]["verl"] = {"n_gpus": 4, "rollout_tp": 2}
    left = {"n_gpus": 1, "rollout_tp": 1}
    right = {"n_gpus": 8, "rollout_tp": 8}

    assert rlvr_protocol_identity(
        rlvr, "math", IdentityFamily(), topology=left
    ) == rlvr_protocol_identity(rlvr, "math", IdentityFamily(), topology=right)
    assert _debate_identity(debate, topology=left) == _debate_identity(
        debate, topology=right
    )


def _main_args():
    return SimpleNamespace(
        experiment_file="unused.yaml",
        experiment="experiment",
        levels=None,
        wandb_resume=None,
        no_wandb=True,
        load=None,
        lr=None,
        group_size=None,
        batch_size=None,
        steps=None,
        start_step=None,
        wandb_entity=None,
        wandb_project=None,
    )


def test_rlvr_resolves_topology_once_and_reuses_same_value(monkeypatch):
    import infra.run_rlvr as run_rlvr

    marker = {"n_gpus": 2, "rollout_tp": 2}
    seen = []
    family = SimpleNamespace(
        source=lambda ds: SimpleNamespace(),
        close=lambda: None,
    )
    backend = SimpleNamespace(
        config=SimpleNamespace(checkpoint_dir=None), tokenizer=object()
    )
    exp = {
        "model": "org/model",
        "max_completion_tokens": 32,
        "dataset": {"type": "math"},
        "training": {},
    }

    monkeypatch.setattr(
        run_rlvr, "runner_parser", lambda doc: SimpleNamespace(parse_args=_main_args)
    )
    monkeypatch.setattr(run_rlvr, "load_experiment", lambda *args: exp)
    monkeypatch.setattr(run_rlvr, "get_family", lambda kind: family)
    monkeypatch.setattr(
        run_rlvr,
        "resolve_topology",
        lambda: seen.append(("resolve", marker)) or marker,
    )
    monkeypatch.setattr(
        run_rlvr,
        "rlvr_protocol_identity",
        lambda *args, topology=None, **kwargs: seen.append(("identity", topology)) or {},
    )
    monkeypatch.setattr(
        run_rlvr,
        "build_backend",
        lambda *args, topology=None, **kwargs: seen.append(("backend", topology)) or backend,
    )
    monkeypatch.setattr(run_rlvr, "train", lambda *args, **kwargs: None)

    run_rlvr._main([])

    assert seen == [("resolve", marker), ("identity", marker), ("backend", marker)]
    assert seen[1][1] is marker and seen[2][1] is marker


def test_debate_resolves_topology_once_and_reuses_same_value(monkeypatch):
    import infra.run_debate as run_debate
    from infra.models.base import ModelSettings

    marker = {"n_gpus": 2, "rollout_tp": 2}
    seen = []
    family = SimpleNamespace(close=lambda: None)
    trained = {
        "alice": ModelSettings(
            model_type="local", model_file_path="org/model", alias="Alice"
        )
    }
    env = SimpleNamespace(
        family=family,
        protocol=object(),
        task_source=object(),
    )
    backend = SimpleNamespace(config=SimpleNamespace(checkpoint_dir=None))
    exp = {"dataset": {"type": "math"}, "training": {}}

    monkeypatch.setattr(
        run_debate, "runner_parser", lambda doc: SimpleNamespace(parse_args=_main_args)
    )
    monkeypatch.setattr(run_debate, "load_experiment", lambda *args: exp)
    monkeypatch.setattr(run_debate, "validate_experiment", lambda exp: None)
    monkeypatch.setattr(run_debate, "split_agents", lambda exp: (trained, {}))
    monkeypatch.setattr(run_debate, "validate_trained_seats", lambda *args: None)
    monkeypatch.setattr(run_debate, "build_env", lambda *args: env)
    monkeypatch.setattr(run_debate, "debate_gen_budgets", lambda *args: {})
    monkeypatch.setattr(
        run_debate,
        "resolve_topology",
        lambda: seen.append(("resolve", marker)) or marker,
    )
    monkeypatch.setattr(
        run_debate,
        "debate_protocol_identity",
        lambda *args, topology=None, **kwargs: seen.append(("identity", topology)) or {},
    )
    monkeypatch.setattr(
        run_debate,
        "build_backend",
        lambda *args, topology=None, **kwargs: seen.append(("backend", topology)) or backend,
    )
    monkeypatch.setattr(run_debate, "train", lambda *args, **kwargs: None)

    run_debate._main([])

    assert seen == [("resolve", marker), ("identity", marker), ("backend", marker)]
    assert seen[1][1] is marker and seen[2][1] is marker


def test_debate_runner_identity_rejects_secret_bearing_base_url(tmp_path):
    prompt = tmp_path / "prompts.yaml"
    prompt.write_text("debate:\n  system: prompt\n")
    exp = _debate_exp(prompt)
    exp["agents"]["judge"]["model_settings"]["base_url"] = (
        "https://example.test/v1?api_key=secret"
    )
    with pytest.raises(ValueError, match="base_url may not contain"):
        _debate_identity(exp)


def test_new_wandb_run_records_protocol_identity(monkeypatch):
    wandb, _ = _install_fake_wandb(monkeypatch, {})
    identity = {"dataset_type": "math", "grading_protocol": "numeric-v1"}

    logger = train_mod._make_logger(
        Config(wandb_project="project", run_name="fresh", protocol_identity=identity)
    )

    assert wandb.init_calls[0]["config"]["protocol_identity"] == identity
    logger.close()


def test_resume_checks_identity_before_mutable_config_update(monkeypatch):
    identity = {"dataset_type": "math_symbolic", "grading_protocol": "symbolic-v1"}
    wandb, run = _install_fake_wandb(monkeypatch, {"protocol_identity": identity, "steps": 10})

    logger = train_mod._make_logger(
        Config(
            wandb_project="project",
            wandb_run_id="run-id",
            steps=20,
            protocol_identity=identity,
        )
    )

    assert wandb.init_calls == [
        {"project": "project", "entity": None, "id": "run-id", "resume": "must"}
    ]
    assert wandb.api_run_calls == ["project/run-id"]
    assert len(run.config.updates) == 1
    mutable, allow_val_change = run.config.updates[0]
    assert allow_val_change is True
    assert mutable["steps"] == 20
    assert "protocol_identity" not in mutable
    assert run.config["protocol_identity"] == identity
    logger.close()


def test_resume_api_path_includes_explicit_entity(monkeypatch):
    identity = {"dataset_type": "math"}
    wandb, _ = _install_fake_wandb(monkeypatch, {"protocol_identity": identity})
    logger = train_mod._make_logger(
        Config(
            wandb_project="project",
            wandb_entity="team",
            wandb_run_id="run-id",
            protocol_identity=identity,
        )
    )
    assert wandb.api_run_calls == ["team/project/run-id"]
    logger.close()


@pytest.mark.parametrize("stored", [None, {"dataset_type": "math"}])
def test_resume_rejects_missing_or_different_identity_before_update(monkeypatch, stored):
    stored_config = {} if stored is None else {"protocol_identity": stored}
    wandb, run = _install_fake_wandb(monkeypatch, stored_config)
    cfg = Config(
        wandb_project="project",
        wandb_run_id="run-id",
        protocol_identity={"dataset_type": "math_symbolic"},
    )

    with pytest.raises(ValueError, match="different or missing protocol_identity"):
        train_mod._make_logger(cfg)

    assert wandb.init_calls == []
    assert wandb.api_run_calls == ["project/run-id"]
    assert run.config.updates == []
    assert run.finish_calls == 0


class LifecycleBackend:
    tokenizer = None

    def __init__(self, *, fail_save=False):
        self.fail_save = fail_save

    def save(self, name):
        if self.fail_save:
            raise RuntimeError("save failed")


def _zero_step_config(**kwargs):
    return Config(steps=0, eval_every=0, save_every=0, **kwargs)


def test_train_finishes_wandb_run_once_on_normal_completion(monkeypatch):
    _, run = _install_fake_wandb(monkeypatch, {})
    train_mod.train(object(), LifecycleBackend(), _zero_step_config(wandb_project="project"))
    assert run.finish_calls == 1


def test_train_finishes_wandb_run_once_on_exception(monkeypatch):
    _, run = _install_fake_wandb(monkeypatch, {})
    with pytest.raises(RuntimeError, match="save failed"):
        train_mod.train(
            object(),
            LifecycleBackend(fail_save=True),
            _zero_step_config(wandb_project="project"),
        )
    assert run.finish_calls == 1


def test_logger_finishes_run_when_resume_config_update_fails(monkeypatch):
    identity = {"dataset_type": "math"}
    wandb, run = _install_fake_wandb(monkeypatch, {"protocol_identity": identity})

    def fail_update(values, *, allow_val_change=False):
        raise RuntimeError("config update failed")

    run.config.update = fail_update
    with pytest.raises(RuntimeError, match="config update failed"):
        train_mod._make_logger(
            _zero_step_config(
                wandb_project="project",
                wandb_run_id="run-id",
                protocol_identity=identity,
            )
        )

    assert len(wandb.init_calls) == 1
    assert run.finish_calls == 1


def test_train_without_wandb_does_not_open_or_finish_run(monkeypatch):
    wandb, run = _install_fake_wandb(monkeypatch, {})
    train_mod.train(object(), LifecycleBackend(), _zero_step_config())
    assert wandb.init_calls == []
    assert run.finish_calls == 0


def test_logger_close_is_idempotent(monkeypatch):
    _, run = _install_fake_wandb(monkeypatch, {})
    logger = train_mod._make_logger(_zero_step_config(wandb_project="project"))
    logger.close()
    logger.close()
    assert run.finish_calls == 1


def test_two_train_calls_create_and_finish_distinct_wandb_runs(monkeypatch):
    class MultiRunWandb:
        def __init__(self):
            self.runs = []

        def init(self, **kwargs):
            run = FakeRun(kwargs.get("config", {}))
            self.runs.append(run)
            return run

    wandb = MultiRunWandb()
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    monkeypatch.setattr(train_mod, "_env_identity", lambda: {"env/git_dirty": "no"})
    cfg = _zero_step_config(wandb_project="project")

    train_mod.train(object(), LifecycleBackend(), cfg)
    train_mod.train(object(), LifecycleBackend(), cfg)

    assert len(wandb.runs) == 2
    assert wandb.runs[0] is not wandb.runs[1]
    assert [run.finish_calls for run in wandb.runs] == [1, 1]


def test_wandb_resume_requires_checkpoint_load():
    args = SimpleNamespace(wandb_resume="run-id", no_wandb=False, load=None)
    with pytest.raises(ValueError, match="requires --load"):
        validate_resume_args(args)


def test_wandb_resume_rejects_no_wandb():
    args = SimpleNamespace(
        wandb_resume="run-id", no_wandb=True, load="step-00010", start_step=None
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        validate_resume_args(args)


def test_resume_args_allow_valid_continuation_and_fresh_run():
    validate_resume_args(
        SimpleNamespace(
            wandb_resume="run-id",
            no_wandb=False,
            load="/checkpoints/step-00010",
            start_step=None,
        )
    )
    validate_resume_args(
        SimpleNamespace(
            wandb_resume=None, no_wandb=True, load=None, start_step=None
        )
    )


@pytest.mark.parametrize("load", ["/checkpoints/final", "tinker://opaque-id"])
def test_resume_requires_explicit_step_for_final_or_opaque_checkpoint(load):
    with pytest.raises(ValueError, match="requires an explicit --start-step"):
        validate_resume_args(
            SimpleNamespace(
                wandb_resume="run-id",
                no_wandb=False,
                load=load,
                start_step=None,
            )
        )


@pytest.mark.parametrize("load", ["/checkpoints/final", "tinker://opaque-id"])
def test_resume_accepts_explicit_step_for_final_or_opaque_checkpoint(load):
    validate_resume_args(
        SimpleNamespace(
            wandb_resume="run-id",
            no_wandb=False,
            load=load,
            start_step=100,
        )
    )


def test_rlvr_closes_family_when_source_construction_fails(monkeypatch):
    import infra.run_rlvr as run_rlvr

    closed = []

    class Family:
        def source(self, ds):
            raise RuntimeError("source failed")

        def close(self):
            closed.append(True)

    args = SimpleNamespace(
        experiment_file="unused.yaml",
        experiment="unused",
        levels=None,
        wandb_resume=None,
        no_wandb=False,
        load=None,
    )
    monkeypatch.setattr(
        run_rlvr, "runner_parser", lambda description: SimpleNamespace(parse_args=lambda: args)
    )
    monkeypatch.setattr(run_rlvr, "load_experiment", lambda *args: {})
    monkeypatch.setattr(run_rlvr, "validate_experiment", lambda exp: None)
    monkeypatch.setattr(run_rlvr, "get_family", lambda dataset_type: Family())

    with pytest.raises(RuntimeError, match="source failed"):
        run_rlvr.main()

    assert closed == [True]


def test_debate_build_env_closes_family_when_source_fails(monkeypatch):
    import infra.run_debate as run_debate

    closed = []

    class Family:
        def source(self, ds):
            raise RuntimeError("source failed")

        def close(self):
            closed.append(True)

    protocol = SimpleNamespace(decision_slot=None, compile=lambda: [])
    monkeypatch.setattr(run_debate.Protocol, "parse", lambda spec: protocol)
    monkeypatch.setattr(run_debate, "get_family", lambda dataset_type: Family())
    exp = {
        "protocol": {},
        "prompt_config": {"file_path": "unused", "entry": "unused"},
        "dataset": {"type": "math"},
    }

    with pytest.raises(RuntimeError, match="source failed"):
        run_debate.build_env(exp, {}, {})

    assert closed == [True]


def test_debate_build_env_closes_family_when_env_construction_fails(monkeypatch):
    import infra.run_debate as run_debate

    closed = []

    class Family:
        def source(self, ds):
            return object()

        def close(self):
            closed.append(True)

    protocol = SimpleNamespace(decision_slot=None, compile=lambda: [])
    monkeypatch.setattr(run_debate.Protocol, "parse", lambda spec: protocol)
    monkeypatch.setattr(run_debate, "get_family", lambda dataset_type: Family())

    def fail_env(*args, **kwargs):
        raise RuntimeError("env failed")

    monkeypatch.setattr(run_debate, "DebateEnv", fail_env)
    exp = {
        "protocol": {},
        "prompt_config": {"file_path": "unused", "entry": "unused"},
        "dataset": {"type": "math"},
    }

    with pytest.raises(RuntimeError, match="env failed"):
        run_debate.build_env(exp, {}, {})

    assert closed == [True]


def test_supported_train_entrypoint_records_identity_and_closes_family(monkeypatch):
    import infra.envs.tasks as tasks_mod

    closed = []
    trained = []

    class Family:
        def source(self, ds):
            return object()

        def protocol_identity(self):
            return {}

        def close(self):
            closed.append(True)

    class Backend:
        def __init__(self, *args, **kwargs):
            pass

    args = SimpleNamespace(
        env="math",
        base_model="model",
        lora_rank=4,
        steps=0,
        batch_size=1,
        group_size=1,
        lr=1e-5,
        max_tokens=16,
        loss="ppo",
        eval_every=0,
        eval_n=1,
        wandb_project=None,
        wandb_entity=None,
        load=None,
    )

    class Parser:
        def add_argument(self, *args, **kwargs):
            pass

        def parse_args(self):
            return args

    monkeypatch.setattr(train_mod.argparse, "ArgumentParser", lambda **kwargs: Parser())
    monkeypatch.setitem(sys.modules, "infra.backend.tinker", SimpleNamespace(TinkerBackend=Backend))
    monkeypatch.setattr(tasks_mod, "get_family", lambda dataset_type: Family())
    monkeypatch.setattr(
        train_mod,
        "train",
        lambda env, backend, cfg: trained.append(dict(cfg.protocol_identity)),
    )

    train_mod.main()

    assert trained == [{"dataset_type": "math"}]
    assert closed == [True]
