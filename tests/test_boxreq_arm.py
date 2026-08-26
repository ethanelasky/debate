"""The judge-enforced-format arm changes only what the judge reads.

Under the base prompt a proposal with no \\boxed{...} still binds a position
from the relaxed last-number fallback and can win the debate, so the only
pressure toward a box is the 0.1 format shaping. This arm moves that rule into
the verdict. The debaters' prompts, the protocol and the whole training block
must stay byte-identical, or the pair stops being a one-variable comparison.
"""

import hashlib
import warnings
from pathlib import Path

import pytest

from infra.config import load_experiment
from infra.envs.debate.prompts import load_prompt_library
from infra.envs.debate.protocol import Protocol

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "math_pc_debate.yaml"
BASE_ARM = "mathl5_qwen35_pc_debate_cispo_verl"
BOXREQ_ARM = "mathl5_qwen35_pc_debate_cispo_verl_boxreq"

# runner_prompt_sha256 hashes the WHOLE prompt file, not the entry, so an edit
# here moves the identity of every arm reading it and locks in-flight runs out
# of --wandb-resume. Published by runs 0s44ib3q and 7dlmbk7i.
PUBLISHED_MATH_PROMPT_SHA = (
    "0a4369a104abbb841ec818887c8fed5be132db2f881ce7491130a382032c8323"
)


def _arm(name: str) -> dict:
    return load_experiment(CONFIG, name)


def _library(exp: dict):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_prompt_library(
            exp["prompt_config"]["file_path"],
            exp["prompt_config"]["entry"],
            Protocol.parse(exp["protocol"]),
        )


def _judge_text(lib) -> str:
    """Every rendered string the judge reads that this pack authors."""
    return "\n".join(
        [lib.system["judge"], *lib.preamble["judge"], lib.slots["deliberation"], lib.slots["verdict"]]
    )


def test_the_two_arms_differ_only_in_prompt_config():
    base, boxreq = _arm(BASE_ARM), _arm(BOXREQ_ARM)
    differing = {k for k in set(base) | set(boxreq) if base.get(k) != boxreq.get(k)}
    assert differing == {"prompt_config"}
    assert base["training"] == boxreq["training"]
    assert base["prompt_config"]["entry"] != boxreq["prompt_config"]["entry"]


@pytest.mark.parametrize("seat", ["alice", "bob"])
def test_the_debaters_read_identical_prompts(seat):
    base, boxreq = _library(_arm(BASE_ARM)), _library(_arm(BOXREQ_ARM))
    assert base.system[seat] == boxreq.system[seat]
    assert base.preamble.get(seat) == boxreq.preamble.get(seat)


def test_only_the_judge_cues_differ():
    base, boxreq = _library(_arm(BASE_ARM)), _library(_arm(BOXREQ_ARM))
    assert {k for k in base.slots if base.slots[k] != boxreq.slots[k]} == {
        "deliberation",
        "verdict",
    }


def test_the_boxreq_judge_is_never_handed_the_extracted_position():
    """The base pack tells the judge the answer value, which comes from the
    relaxed fallback and exists even when the solution has no box."""
    assert "<OPPONENT_POSITION>" in _judge_text(_library(_arm(BASE_ARM)))
    assert "<OPPONENT_POSITION>" not in _judge_text(_library(_arm(BOXREQ_ARM)))


def test_the_boxreq_judge_contract_names_the_boxed_form():
    """The literal LaTeX form the strict extractor requires, which the base
    pack never shows the judge — it only ever says "boxed answer"."""
    assert "\\boxed{" not in _judge_text(_library(_arm(BASE_ARM)))
    assert "\\boxed{" in _judge_text(_library(_arm(BOXREQ_ARM)))


def test_the_boxreq_pack_keeps_the_parent_judge_reasoning_stages():
    """Only block 0 of judge_system is overridden; the by-index merge must
    leave the evaluation and standard-of-proof blocks in place."""
    card = _library(_arm(BOXREQ_ARM)).system["judge"]
    assert "<evaluation-steps>" in card
    assert "<standard-of-proof>" in card


def test_the_judge_preamble_still_opens_on_the_problem_statement():
    """tests/test_debate_env.py pins this at index 0 of the judge's first user
    message; the override restates block 0 and must not have drifted."""
    base, boxreq = _library(_arm(BASE_ARM)), _library(_arm(BOXREQ_ARM))
    assert base.preamble["judge"][0].splitlines()[:3] == boxreq.preamble["judge"][0].splitlines()[:3]


def test_the_shared_prompt_file_identity_is_unchanged():
    path = Path(_arm(BASE_ARM)["prompt_config"]["file_path"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PUBLISHED_MATH_PROMPT_SHA
