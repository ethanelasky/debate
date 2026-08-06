"""FrozenSeat sampling plumbing: the seat's resolved train profile rides each
predict call as kwargs (temperature/top_p/presence/frequency), so local/API
seats sample at the YAML's values instead of server defaults. Absent profile =
the old behavior (no sampling kwargs at all)."""

import pytest
import yaml

from infra.envs.base import SlotLimits
from infra.envs.debate.env import DebateEnv, DebateEnvConfig
from infra.envs.debate.round import FrozenSeat, GenRequest
from infra.envs.tasks.monitoringbench import MonitoringBenchFamily
from infra.models.base import ModelSettings, SamplingProfile, resolved_sampling_profile

from test_debate_env import ScriptedModel
from test_env_extensions import GOOD_VERDICT, MBTaskSource, PROMPTS_YAML, PROTOCOL

SAMPLING_KEYS = {"temperature", "top_p", "presence_penalty", "frequency_penalty"}


class KwargRecordingModel(ScriptedModel):
    """ScriptedModel that keeps every predict call's kwargs, in call order."""

    def __init__(self, alias, script):
        super().__init__(alias, script)
        self.calls: list[dict] = []

    def predict(self, inputs, **kw):
        self.calls.append(dict(kw))
        return super().predict(inputs, **kw)


def req(**kw):
    return GenRequest(
        messages=[{"role": "user", "content": "hi"}], limits=SlotLimits(max_total_tokens=64), **kw
    )


# ------------------------------------------------------------- seat level


def test_profile_fields_ride_as_predict_kwargs():
    model = KwargRecordingModel("judge", ["ok"])
    seat = FrozenSeat(
        model,
        sampling=SamplingProfile(
            temperature=0.9, top_p=0.95, presence_penalty=0.1, frequency_penalty=0.2
        ),
    )
    seat.generate([req()])
    (kw,) = model.calls
    assert kw["temperature"] == 0.9
    assert kw["top_p"] == 0.95
    assert kw["presence_penalty"] == 0.1
    assert kw["frequency_penalty"] == 0.2


def test_absent_profile_sends_no_sampling_kwargs():
    model = KwargRecordingModel("judge", ["ok"])
    FrozenSeat(model).generate([req()])
    (kw,) = model.calls
    assert not (SAMPLING_KEYS & set(kw))


def test_unset_profile_fields_are_not_sent():
    model = KwargRecordingModel("judge", ["ok"])
    FrozenSeat(model, sampling=SamplingProfile(temperature=1.0)).generate([req()])
    (kw,) = model.calls
    assert kw["temperature"] == 1.0
    assert not ((SAMPLING_KEYS - {"temperature"}) & set(kw))


def test_yaml_train_profile_resolves_and_reaches_predict():
    # The exact composition run_debate.build_env performs per frozen seat.
    settings = ModelSettings(
        alias="judge",
        model_type="local",
        model_file_path="qwen3-8b",
        sampling={"train": {"temperature": 1.0, "top_p": 1.0}},
    )
    model = KwargRecordingModel("judge", ["ok"])
    FrozenSeat(model, sampling=resolved_sampling_profile(settings, "train")).generate([req()])
    (kw,) = model.calls
    assert kw["temperature"] == 1.0 and kw["top_p"] == 1.0


# -------------------------------------------------------------- env level


@pytest.fixture(scope="module")
def prompt_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("prompts") / "mb_test_prompts.yaml"
    path.write_text(PROMPTS_YAML)
    assert "mb_test" in yaml.safe_load(PROMPTS_YAML)
    return str(path)


def test_env_plumbs_frozen_sampling_per_speaker(prompt_file):
    alice = KwargRecordingModel("alice", ["a"] * 4)
    bob = KwargRecordingModel("bob", ["b"] * 4)
    judge = KwargRecordingModel("judge", ["deliberating", GOOD_VERDICT])
    env = DebateEnv(
        DebateEnvConfig(
            protocol=PROTOCOL,
            prompt_file=prompt_file,
            prompt_entry="mb_test",
            trained_speakers=[],
            frozen_models={"alice": alice, "bob": bob, "judge": judge},
            frozen_sampling={
                "judge": SamplingProfile(temperature=1.0, top_p=1.0),
                "alice": SamplingProfile(temperature=0.7),
            },
            fresh_positions=False,
        ),
        MBTaskSource(),
        MonitoringBenchFamily(),
    )
    env.rollout(MBTaskSource(1).tasks(1), policy=None, group_size=1)
    assert env.last_states[0].failed is None
    assert alice.calls and all(c["temperature"] == 0.7 and "top_p" not in c for c in alice.calls)
    assert judge.calls and all(c["temperature"] == 1.0 and c["top_p"] == 1.0 for c in judge.calls)
    # bob has no profile: server defaults, exactly as before
    assert bob.calls and all(not (SAMPLING_KEYS & set(c)) for c in bob.calls)
