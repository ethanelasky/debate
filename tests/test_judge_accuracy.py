"""Judge accuracy derived from transcripts (AGENTS.md: never from aggregated
result files). Covers the decision table in judge_was_right plus one real
round trip through the exporter, so the metadata contract the derivation reads
is pinned to the metadata the exporter actually writes."""

import json

import pytest

from infra.envs.debate.judge_accuracy import from_jsonl, judge_was_right, tally

A, B = "Debater_A", "Debater_B"


def meta(winner=A, grades=None, failed=None, ok=True, seats=(A, B)):
    return {
        "failed": failed,
        "bindings": {"alice": A, "bob": B, "judge": A},  # judge dupes A, as exported
        "grades": {"alice": True} if grades is None else grades,
        "verdict": {"ok": ok, "winner": winner, "seats": {s: "correct" for s in seats}},
    }


# 1 ------------------------------------------------------- proposer-critic


def test_proposal_right_and_judge_sided_with_proposer():
    assert judge_was_right(meta(winner=A, grades={"alice": True})) == (True, "decidable")


def test_proposal_right_but_judge_sided_with_critic():
    assert judge_was_right(meta(winner=B, grades={"alice": True})) == (False, "decidable")


def test_proposal_wrong_and_judge_sided_with_critic():
    """The critic wins by refuting: a wrong proposal means B is the right call."""
    assert judge_was_right(meta(winner=B, grades={"alice": False})) == (True, "decidable")


def test_proposal_wrong_but_judge_sided_with_proposer():
    assert judge_was_right(meta(winner=A, grades={"alice": False})) == (False, "decidable")


def test_critic_seat_can_be_the_graded_one():
    """Nothing assumes the proposer is 'alice' — the label's own speaker binds."""
    assert judge_was_right(meta(winner=B, grades={"bob": True})) == (True, "decidable")


# 2 ------------------------------------------------------- dual proposer


def test_dual_proposer_judge_picks_the_correct_answer():
    m = meta(winner=B, grades={"alice": False, "bob": True})
    assert judge_was_right(m) == (True, "decidable")


def test_dual_proposer_both_right_is_undecidable():
    m = meta(winner=A, grades={"alice": True, "bob": True})
    assert judge_was_right(m) == (None, "ambiguous")


def test_dual_proposer_both_wrong_is_undecidable():
    m = meta(winner=A, grades={"alice": False, "bob": False})
    assert judge_was_right(m) == (None, "ambiguous")


# 3 ------------------------------------------------------------ skips


@pytest.mark.parametrize(
    "m, reason",
    [
        (meta(failed="alice/proposal: fidelity"), "failed"),
        (meta(ok=False), "no_verdict"),
        (meta(winner=None), "no_winner"),
        (meta(winner="Tie"), "tie"),
        (meta(grades={}), "no_labels"),
        (meta(grades={"alice": None}), "label_missing"),      # graded, ungradeable
        (meta(grades={"nobody": True}), "unmapped"),          # speaker not a seat
    ],
)
def test_undecidable_cases(m, reason):
    assert judge_was_right(m) == (None, reason)


def test_missing_verdict_block_entirely():
    assert judge_was_right({"failed": None}) == (None, "no_verdict")


# 4 ------------------------------------------------------------ tally


def test_tally_counts_denominator_separately_from_rate():
    acc = tally(
        [
            meta(winner=A, grades={"alice": True}),    # correct
            meta(winner=A, grades={"alice": False}),   # wrong
            meta(winner=A, grades={"alice": True}),    # correct
            meta(winner="Tie"),                        # skipped
            meta(failed="verdict_unparseable"),        # skipped
        ]
    )
    assert (acc.total, acc.decidable, acc.correct) == (5, 3, 2)
    assert acc.accuracy == pytest.approx(2 / 3)
    assert acc.skipped == {"tie": 1, "failed": 1}


def test_accuracy_is_none_rather_than_zero_when_nothing_is_decidable():
    acc = tally([meta(winner="Tie"), meta(failed="x")])
    assert acc.accuracy is None
    assert "n/a" in str(acc)


# 5 ------------------------------------------- round trip through the exporter


def test_end_to_end_from_exported_jsonl(tmp_path):
    """The real contract test: labels attached during rollout must survive
    export and be readable by the derivation with no re-grading."""
    from tests.test_debate_env import GOOD_VERDICT, ScriptedBackend, TaskSource, make_env
    from infra.backend.base import SamplingParams
    from infra.envs.base import Policy
    from infra.envs.debate.docent_export import agent_runs, export_jsonl

    backend = ScriptedBackend(["\\boxed{2}", "Defense."])
    policy = Policy(backend, SamplingParams(max_tokens=128))
    env = make_env(["alice"], [GOOD_VERDICT])
    # TaskSource question 1 is "What is 1+1?" with gt 2.0; the proposal is
    # \boxed{2}, so alice is genuinely correct and the judge picked Debater_A.
    env.rollout(TaskSource().tasks(1), policy, group_size=1)

    path = export_jsonl(agent_runs(env), str(tmp_path / "debates.jsonl"))
    line = json.loads(open(path).read().splitlines()[0])
    assert line["metadata"]["grades"] == {"alice": True}

    acc = from_jsonl(path)
    assert (acc.total, acc.decidable, acc.correct) == (1, 1, 1)
    assert acc.accuracy == 1.0


def test_label_survives_export_where_the_ground_truth_cannot(tmp_path):
    """The codecontests case. Its ground truth is the test cases, which are
    list-valued and so are stripped by the exporter's scalar filter (they must
    never leave the verifier). Without the persisted label there would be
    nothing left to derive from — math only escapes this because its `gt` is a
    float that survives the same filter."""
    from tests.test_debate_env import GOOD_VERDICT, ScriptedBackend, TaskSource, make_env
    from infra.backend.base import SamplingParams
    from infra.envs.base import Policy, Task
    from infra.envs.debate.docent_export import agent_runs, export_jsonl

    class TestCaseSource(TaskSource):
        def tasks(self, n, split="train"):
            return [
                Task(
                    messages=[{"role": "user", "content": "solve it"}],
                    meta={
                        "question": "solve it",
                        "name": "problem-1",
                        "inputs": ["1\n"],      # list-valued: stripped by the
                        "outputs": ["2\n"],     # exporter, like codecontests
                    },
                )
                for _ in range(n)
            ]

    class TestCaseFamily:
        def extractor(self, relaxed):
            return lambda text: text.strip() or None

        def grade(self, meta, solution):
            # Reads exactly the keys the export drops.
            if not meta.get("inputs") or not meta.get("outputs"):
                return None
            return "print" in str(solution)

        def format_flags(self, text):
            return {}

    backend = ScriptedBackend(["print(2)", "Defense."])
    policy = Policy(backend, SamplingParams(max_tokens=128))
    env = make_env(["alice"], [GOOD_VERDICT], task_source=TestCaseSource(), family=TestCaseFamily())
    env.rollout(TestCaseSource().tasks(1), policy, group_size=1)

    path = export_jsonl(agent_runs(env), str(tmp_path / "cc.jsonl"))
    line = json.loads(open(path).read().splitlines()[0])

    # ground truth is gone from the export...
    assert "inputs" not in line["metadata"]["task"]
    assert "outputs" not in line["metadata"]["task"]
    # ...but the label it produced is not, so accuracy is still derivable
    assert line["metadata"]["grades"] == {"alice": True}
    acc = from_jsonl(path)
    assert (acc.decidable, acc.correct) == (1, 1)
