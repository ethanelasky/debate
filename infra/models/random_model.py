"""Random model: gibberish speeches and coin-flip verdicts. Test fixture only.

Limited to the Debater_A/Debater_B name family. Judge outputs carry a JSON
verdict envelope because the judge path parses the winner out of `speech`.
"""

from __future__ import annotations

import random
import string
from typing import Optional

from infra.models.base import (
    DEFAULT_DEBATER_A_NAME,
    DEFAULT_DEBATER_B_NAME,
    DEFAULT_MAX_NEW_TOKENS,
    Model,
    ModelInput,
    ModelResponse,
    SpeechStructure,
)


class RandomModel(Model):
    def __init__(self, alias: str, is_debater: bool = False, seed: Optional[int] = None, **kwargs):
        """seed pins the stream so tests can assert exact outputs."""
        super().__init__(alias=alias, is_debater=is_debater)
        self.seed = seed
        self._rng = random.Random(seed)

    def predict(
        self,
        inputs: list[list[ModelInput]],
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        num_return_sequences: int = 1,
        speech_structure: SpeechStructure = SpeechStructure.OPEN_ENDED,
        **kwargs,
    ) -> list[ModelResponse]:
        out = []
        for conversation in inputs:
            prompt = "\n".join(mi.content for mi in conversation)
            for _ in range(num_return_sequences):
                out.append(self._one(prompt, max_new_tokens, speech_structure))
        return out

    def _one(self, prompt: str, max_new_tokens: int, speech_structure: SpeechStructure) -> ModelResponse:
        a_odds = self._rng.random()
        winner = DEFAULT_DEBATER_A_NAME if a_odds > 0.5 else DEFAULT_DEBATER_B_NAME
        verdict = f'{{"winner": "{winner}", "confidence": {round(max(a_odds, 1 - a_odds), 3)}}}'
        if speech_structure == SpeechStructure.DECISION:
            return ModelResponse(
                speech=verdict,
                decision=winner,
                probabilistic_decision={
                    DEFAULT_DEBATER_A_NAME: a_odds,
                    DEFAULT_DEBATER_B_NAME: 1 - a_odds,
                },
                prompt=prompt,
            )
        text = self._gibberish(max_new_tokens)
        # A judge asked open-ended still has to emit something parseable.
        return ModelResponse(
            speech=text if self.is_debater else f"{text} {verdict}",
            prompt=prompt,
        )

    def _gibberish(self, max_new_tokens: int) -> str:
        words = [
            "".join(self._rng.choices(string.ascii_lowercase, k=self._rng.randrange(1, 8)))
            for _ in range(self._rng.randrange(1, max(2, min(max_new_tokens, 64))))
        ]
        return " ".join(words) + " <quote>This is not real</quote>."

    def copy(self, alias: Optional[str] = None, is_debater: Optional[bool] = None, **kwargs) -> "RandomModel":
        return RandomModel(
            alias=alias or self.alias,
            is_debater=self.is_debater if is_debater is None else is_debater,
            seed=self.seed,
        )
