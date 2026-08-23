"""The debate arm's dev metric must be the RLVR arm's instrument.

A dev number is comparable across arms only when both arms read the same pool,
render the same prompt and generate under the same caps. The debate arm read
its own MATH-L5 dev carve at a bare token cap while the RLVR arm read AMC
under a budget-forced think phase, so their step-0 points (0.531 against
0.506, on 177 problems against 83) measured different quantities from the same
base weights. These tests pin the pieces of that instrument: the shipped pair
agree, and the runner seam actually applies what the config declares.
"""

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

import infra.envs.tasks.amc as amc_mod
from infra.config import load_experiment
from infra.envs.base import SingleTurnEnv, SlotLimits, Task
from infra.run_common import build_eval_source, eval_slot_limits, prepare_eval_env
from infra.run_debate import validate_experiment as validate_debate

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
DEBATE_ARM = "mathl5_qwen35_pc_debate_cispo_verl"
RLVR_ARM = "mathl5_qwen35_cispo"

FAKE_AMC = [
    {"problem": "What is 2+2?", "answer": "4"},
    {"problem": "What is 10*10?", "answer": "100"},
]


@pytest.fixture
def fake_amc(monkeypatch):
    """The AMC pool without the HuggingFace download; row content is
    irrelevant here, only that the declared family is the one built."""
    monkeypatch.setattr(amc_mod, "_load_amc", lambda dataset_id=None: FAKE_AMC)


class StubSource(SingleTurnEnv):
    """A task source that is NOT the declared eval pool. Real eval envs are
    SingleTurnEnv subclasses, and slot_limits is read off that class."""

    def tasks(self, n: int, split: str = "train") -> list[Task]:
        return [Task(messages=[{"role": "user", "content": "training pool"}])]

    def reward(self, task: Task, text: str):
        return 0.0, {}


def _debate() -> dict:
    return load_experiment(CONFIG_DIR / "math_pc_debate.yaml", DEBATE_ARM)


def _rlvr() -> dict:
    return load_experiment(CONFIG_DIR / "math_qwen35.yaml", RLVR_ARM)


def test_shipped_arms_read_the_same_eval_pool():
    debate, rlvr = _debate(), _rlvr()
    assert debate["eval_dataset"] == rlvr["eval_dataset"]
    assert debate["eval_dataset"]["type"] == "amc"


def test_shipped_arms_render_the_same_rollout_prompt():
    """The proposal slot renders the task source's ANSWER_GEN_USER, so
    dataset.prompt_file is the rollout prompt on both arms."""
    debate, rlvr = _debate(), _rlvr()
    assert debate["dataset"]["prompt_file"] == rlvr["dataset"]["prompt_file"]
    assert debate["eval_dataset"]["prompt_file"] == rlvr["dataset"]["prompt_file"]


def test_debate_eval_caps_match_the_rlvr_think_budget():
    """run_rlvr derives its eval caps from think_tokens + max_completion_tokens;
    debate slot budgets never reach an eval env, so the debate arm states the
    same numbers directly."""
    limits = eval_slot_limits(_debate())
    rlvr = _rlvr()
    assert limits.max_think_tokens == rlvr["think_tokens"]
    assert limits.max_total_tokens == rlvr["think_tokens"] + rlvr["max_completion_tokens"]


def test_shipped_arms_share_the_eval_loop_settings():
    debate, rlvr = _debate()["training"], _rlvr()["training"]
    for key in ("eval_split", "eval_n", "eval_every", "eval_max_tokens"):
        assert debate[key] == rlvr[key], key


def test_declared_pool_replaces_the_task_source(fake_amc):
    source, close = build_eval_source(_debate())
    try:
        assert type(source).__name__ == "AmcEnv"
        assert len(source.dev_rows) == len(FAKE_AMC)
        env = prepare_eval_env(source, limits=eval_slot_limits(_debate()))
        assert env is source
        assert env.slot_limits == SlotLimits(
            max_think_tokens=4000, max_total_tokens=5024
        )
    finally:
        close()


def test_declared_pool_renders_the_declared_prompt(fake_amc):
    source, close = build_eval_source(_debate())
    try:
        content = source.tasks(1, split="dev")[0].messages[-1]["content"]
    finally:
        close()
    assert "Give your answer directly, as EXACTLY one" in content
    assert "rigorous justification" not in content  # that is math.yaml's wording


def test_no_eval_dataset_leaves_the_caller_its_own_env():
    assert build_eval_source({}) == (None, None)
    assert build_eval_source({"eval_dataset": {}}) == (None, None)
    assert prepare_eval_env(None, limits=SlotLimits(max_think_tokens=8)) is None


def test_caps_reach_a_fallback_task_source():
    """An arm with no eval_dataset still evaluates under its declared caps."""
    source = StubSource()
    env = prepare_eval_env(source, limits=eval_slot_limits({"eval_slot_limits": {"max_think_tokens": 64}}))
    assert env is source
    assert env.slot_limits == SlotLimits(max_think_tokens=64)


def test_no_caps_declared_leaves_the_env_untouched():
    source = StubSource()
    assert eval_slot_limits({}) is None
    assert prepare_eval_env(source, limits=None).slot_limits is None


def test_plan_tokens_wraps_the_eval_pool(fake_amc):
    source, close = build_eval_source(_debate())
    try:
        env = prepare_eval_env(source, plan_tokens=128)
        assert type(env).__name__ == "PlannedEnv"
        assert env.inner is source
    finally:
        close()


def test_eval_slot_limit_typo_fails_at_launch():
    exp = copy.deepcopy(_debate())
    exp["eval_slot_limits"]["max_think_token"] = exp["eval_slot_limits"].pop(
        "max_think_tokens"
    )
    with pytest.raises(ValueError, match="max_think_token"):
        validate_debate(exp)


def test_eval_dataset_typo_fails_when_the_pool_is_built():
    with pytest.raises(ValueError, match="amc"):
        build_eval_source({"eval_dataset": {"type": "amc", "levels": 5}})


def _run_debate_main(monkeypatch, tmp_path, exp) -> dict:
    """Drive run_debate._main on a real experiment dict, stubbing only the
    seat/backend layers, and return what train() was handed. The eval wiring,
    validate_experiment, the pool build and the caps all run for real."""
    import infra.run_debate as run_debate
    from infra.models.base import ModelSettings

    monkeypatch.chdir(tmp_path)
    captured: dict = {}
    task_source = StubSource()
    runner_env = SimpleNamespace(
        family=SimpleNamespace(close=lambda: None),
        protocol=object(),
        task_source=task_source,
    )
    args = SimpleNamespace(
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
    trained = {
        "alice": ModelSettings(
            model_type="local", model_file_path="org/model", alias="Alice"
        )
    }
    monkeypatch.setattr(
        run_debate, "runner_parser", lambda doc: SimpleNamespace(parse_args=lambda: args)
    )
    monkeypatch.setattr(run_debate, "load_experiment", lambda *a: exp)
    monkeypatch.setattr(run_debate, "split_agents", lambda e: (trained, {}))
    monkeypatch.setattr(run_debate, "validate_trained_seats", lambda *a: None)
    monkeypatch.setattr(run_debate, "build_env", lambda *a: runner_env)
    monkeypatch.setattr(run_debate, "debate_gen_budgets", lambda *a: {})
    monkeypatch.setattr(run_debate, "resolve_launch_namespace", lambda: "eval-pool-test")
    monkeypatch.setattr(run_debate, "resolve_topology", lambda: {})
    monkeypatch.setattr(run_debate, "debate_protocol_identity", lambda *a, **k: {})
    monkeypatch.setattr(
        run_debate,
        "build_backend",
        lambda *a, **k: SimpleNamespace(config=SimpleNamespace(checkpoint_dir=None)),
    )
    monkeypatch.setattr(
        run_debate,
        "train",
        lambda env, backend, cfg, eval_env=None: captured.update(
            eval_env=eval_env, task_source=env.task_source
        ),
    )

    run_debate._main([])
    return captured


def test_runner_evaluates_the_declared_pool_under_the_declared_caps(
    monkeypatch, tmp_path, fake_amc
):
    captured = _run_debate_main(monkeypatch, tmp_path, _debate())
    eval_env = captured["eval_env"]
    assert type(eval_env).__name__ == "AmcEnv"
    assert eval_env is not captured["task_source"]
    assert eval_env.slot_limits == SlotLimits(
        max_think_tokens=4000, max_total_tokens=5024
    )


def test_runner_falls_back_to_the_task_source_without_a_declared_pool(
    monkeypatch, tmp_path
):
    exp = copy.deepcopy(_debate())
    exp.pop("eval_dataset")
    captured = _run_debate_main(monkeypatch, tmp_path, exp)
    assert captured["eval_env"] is captured["task_source"]
    assert captured["eval_env"].slot_limits == SlotLimits(
        max_think_tokens=4000, max_total_tokens=5024
    )
