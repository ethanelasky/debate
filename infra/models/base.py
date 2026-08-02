"""Inference-model layer: types + Model ABC.

Salvaged from ~/ai-debate/ai_infra/models/model.py, slimmed. Kept: the batch
predict() contract, ModelInput/ModelResponse, the YAML-facing ModelSettings
(field-compatible with existing experiment YAMLs), sampling profiles.
Dropped: the SAMPLING_CAPABILITIES declaration machinery, effective-sampling
provenance stamping, BoN scoring, probe hyperparams, generation-ledger fields.
"""

from __future__ import annotations

from abc import ABC
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DEFAULT_DEBATER_A_NAME = "Debater_A"
DEFAULT_DEBATER_B_NAME = "Debater_B"
DEFAULT_MAX_NEW_TOKENS = 1000


class ModelType(Enum):
    RANDOM = 1
    OPENAI = 4
    OFFLINE = 5
    HUMAN = 6
    ANTHROPIC = 10
    FIREWORKS = 13
    ALIBABA = 14
    TINKER = 15
    GOOGLE = 16
    LOCAL = 17
    OPENROUTER = 18


class RoleType(Enum):
    SYSTEM = 1
    USER = 2
    ASSISTANT = 3

    @property
    def api_name(self) -> str:
        return self.name.lower()


class SpeechStructure(Enum):
    OPEN_ENDED = 1
    DECISION = 2


class ModelInput(BaseModel):
    role: RoleType
    content: str

    @field_validator("role", mode="before")
    @classmethod
    def coerce_role(cls, v):
        if isinstance(v, str):
            return RoleType[v.upper()]
        return v


class ModelResponse(BaseModel):
    """One generation. Most fields are optional capture channels; wrappers fill
    what their API exposes."""

    speech: str = ""
    decision: str = ""  # judge decisions: Debater_A / Debater_B / ""
    probabilistic_decision: Optional[dict[str, float]] = None
    response_tokens: list[int] = Field(default_factory=list)
    response_logprobs: list[Optional[float]] = Field(default_factory=list)
    # Per-position token STRINGS from OpenAI-compatible logprobs.content
    # payloads (those APIs return strings, not ids); length-aligned with
    # response_logprobs when set. The judge-logit extraction prefers this.
    response_token_texts: Optional[list[str]] = None
    prompt_tokens: list[int] = Field(default_factory=list)
    prompt: str = ""
    failed: bool = False
    raw_response: Optional[str] = None
    stop_reason: Optional[Literal["stop", "length"]] = None
    thinking: Optional[str] = None  # e.g. Qwen <think> content
    structured_decision: Optional[dict[str, Any]] = None
    served_provider: Optional[str] = None  # OpenRouter: who actually served it
    generation_id: Optional[str] = None


class GenerationParams(BaseModel):
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = 0.5
    top_p: float = 1.0
    reasoning_effort: Optional[Literal["low", "medium", "high"]] = None
    thinking_budget_tokens: Optional[int] = None

    # Old YAMLs may carry dead knobs (repetition_penalty, do_sample, ...);
    # accept and ignore them rather than failing the load.
    model_config = ConfigDict(extra="ignore")


class SamplingProfile(BaseModel):
    """None defers to the wrapper default; a set value is authoritative."""

    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    repetition_penalty: Optional[float] = None

    model_config = ConfigDict(extra="forbid")


class SamplingProfiles(BaseModel):
    train: SamplingProfile = Field(default_factory=SamplingProfile)
    eval: SamplingProfile = Field(default_factory=SamplingProfile)


SamplingBinding = Literal["train", "eval"]


class ModelSettings(BaseModel):
    """The YAML-facing model spec (agents.*.model_settings in experiment
    configs). Field-compatible with the prior repo's YAMLs; typo'd knobs still
    fail the load (extra='forbid')."""

    model_type: str | ModelType = ModelType.RANDOM
    model_file_path: Optional[str] = None
    alias: str
    override_prompt: Optional[str] = None
    nucleus: bool = True  # accepted for YAML compat; unused
    is_human: bool = False
    offline_file_path: Optional[str] = None
    served: bool = False
    require_quote_validation: bool = True
    generation_params: GenerationParams = Field(default_factory=GenerationParams)
    sampling: SamplingProfiles = Field(default_factory=SamplingProfiles)
    reasoning_effort: Optional[Literal["low", "medium", "high"]] = None
    thinking_budget_tokens: Optional[int] = None
    max_new_tokens: Optional[int] = None
    peft_base_model: Optional[str] = None
    lora_rank: Optional[int] = None  # trained seats: adapter rank for the backend
    enable_thinking: Optional[bool] = None
    base_url: Optional[str] = None
    require_token_fidelity: bool = False
    capture_token_logprobs: bool = False
    require_token_logprobs: bool = False
    # OpenRouter routing pins (honored by OpenRouterModel only).
    provider_order: Optional[list[str]] = None
    quantizations: Optional[list[str]] = None
    allow_fallbacks: Optional[bool] = None
    data_collection: Optional[Literal["allow", "deny"]] = None

    model_config = ConfigDict(
        protected_namespaces=("protect_me_",),
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def verify_exclusive_modes(cls, values):
        if isinstance(values, dict):
            count = sum([bool(values.get("is_human")), bool(values.get("served"))]) + (
                1 if values.get("offline_file_path") else 0
            )
            if count > 1:
                raise ValueError("at most one of is_human, served, offline_file_path")
        return values

    @field_validator("alias", mode="before")
    @classmethod
    def validate_alias(cls, alias):
        return str(alias)

    @field_validator("model_type", mode="before")
    @classmethod
    def validate_model_type(cls, model_type):
        if isinstance(model_type, str):
            return ModelType[model_type.upper()]
        return model_type

    @property
    def resolved_reasoning_effort(self) -> Optional[str]:
        return self.reasoning_effort if self.reasoning_effort is not None else self.generation_params.reasoning_effort

    @property
    def resolved_thinking_budget_tokens(self) -> Optional[int]:
        if self.thinking_budget_tokens is not None:
            return self.thinking_budget_tokens
        return self.generation_params.thinking_budget_tokens

    @property
    def resolved_max_new_tokens(self) -> int:
        return self.max_new_tokens if self.max_new_tokens is not None else self.generation_params.max_new_tokens


def resolved_sampling_profile(settings: ModelSettings, binding: SamplingBinding = "eval") -> SamplingProfile:
    """Resolve sampling values for a binding; eval falls back to train, train
    falls back to legacy generation_params."""
    train = settings.sampling.train.model_copy(
        update={
            "temperature": (
                settings.sampling.train.temperature
                if settings.sampling.train.temperature is not None
                else settings.generation_params.temperature
            ),
            "top_p": (
                settings.sampling.train.top_p
                if settings.sampling.train.top_p is not None
                else settings.generation_params.top_p
            ),
        }
    )
    if binding == "train":
        return train
    if binding != "eval":
        raise ValueError(f"unknown sampling binding {binding!r}")
    ev = settings.sampling.eval
    return ev.model_copy(
        update={
            "temperature": ev.temperature if ev.temperature is not None else train.temperature,
            "top_p": ev.top_p if ev.top_p is not None else train.top_p,
        }
    )


class Model(ABC):
    def __init__(self, alias: str, is_debater: bool = False):
        self.alias = alias
        self.is_debater = is_debater

    def predict(
        self,
        inputs: list[list[ModelInput]],
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        num_return_sequences: int = 1,
        **kwargs,
    ) -> list[ModelResponse]:
        """Batch generate. inputs is one conversation per entry. With
        num_return_sequences=n the flat return has inputs[i]'s samples at
        [i*n, (i+1)*n)."""
        raise NotImplementedError

    def decode_tokens(self, tokens: list[int]) -> str:
        raise NotImplementedError(f"{type(self).__name__} has no tokenizer")

    def supports_server_side_sampling(self) -> bool:
        return False

    def copy(self, is_debater: Optional[bool] = None, **kwargs) -> "Model":
        return self

    def can_merge(self, other: "Model") -> bool:
        return other == self

    def merge(self, other: "Model") -> "Model":
        if self.can_merge(other):
            return self
        raise ValueError(f"cannot merge {self.alias} with {other.alias}")
