"""ModelSettings -> Model instantiation (port of the old ModelUtils)."""

from __future__ import annotations

from typing import Optional

from infra.models.base import Model, ModelSettings, ModelType, resolved_sampling_profile


def resolve_model_class(model_type: ModelType) -> Optional[type[Model]]:
    from infra.models.alibaba_model import AlibabaModel
    from infra.models.anthropic_model import AnthropicModel
    from infra.models.fireworks_model import FireworksModel
    from infra.models.google_model import GoogleModel
    from infra.models.local_model import LocalModel
    from infra.models.openai_model import OpenAIModel
    from infra.models.openrouter_model import OpenRouterModel
    from infra.models.random_model import RandomModel

    return {
        ModelType.RANDOM: RandomModel,
        ModelType.OPENAI: OpenAIModel,
        ModelType.ANTHROPIC: AnthropicModel,
        ModelType.FIREWORKS: FireworksModel,
        ModelType.LOCAL: LocalModel,
        ModelType.OPENROUTER: OpenRouterModel,
        ModelType.ALIBABA: AlibabaModel,
        ModelType.GOOGLE: GoogleModel,
    }.get(model_type)


def instantiate_model(
    settings: ModelSettings, is_debater: bool = True, binding: str = "eval"
) -> Optional[Model]:
    """Build a model from YAML-parsed settings. Returns None for types that
    are never directly instantiated (offline, human)."""
    mt = settings.model_type
    if mt in (ModelType.OFFLINE, ModelType.HUMAN):
        return None
    if mt == ModelType.TINKER:
        from infra.models.tinker_model import TinkerModel

        profile = resolved_sampling_profile(settings, binding)
        kwargs = dict(
            alias=settings.alias,
            is_debater=is_debater,
            temperature=profile.temperature,
            top_p=profile.top_p,
            max_tokens=settings.resolved_max_new_tokens,
            enable_thinking=settings.enable_thinking,
        )
        mfp = settings.model_file_path
        if not mfp:
            raise ValueError("model_type: tinker requires model_file_path")
        # Routing (old repo, run-verified): tinker:// URI -> checkpoint;
        # a '/' path -> HF base model; anything else -> legacy checkpoint.
        if mfp.startswith("tinker://"):
            return TinkerModel.from_checkpoint(checkpoint_path=mfp, **kwargs)
        if "/" in mfp:
            return TinkerModel.from_base_model(base_model=mfp, **kwargs)
        return TinkerModel.from_checkpoint(checkpoint_path=mfp, **kwargs)

    common = dict(alias=settings.alias, is_debater=is_debater)
    endpoint = settings.model_file_path

    if mt == ModelType.RANDOM:
        from infra.models.random_model import RandomModel

        return RandomModel(**common)
    if mt == ModelType.OPENAI:
        from infra.models.openai_model import OpenAIModel

        return OpenAIModel(**common, endpoint=endpoint, reasoning_effort=settings.resolved_reasoning_effort)
    if mt == ModelType.ANTHROPIC:
        from infra.models.anthropic_model import AnthropicModel

        return AnthropicModel(
            **common, endpoint=endpoint, thinking_budget_tokens=settings.resolved_thinking_budget_tokens
        )
    if mt == ModelType.FIREWORKS:
        from infra.models.fireworks_model import FireworksModel

        return FireworksModel(**common, endpoint=endpoint, reasoning_effort=settings.resolved_reasoning_effort)
    if mt == ModelType.LOCAL:
        from infra.models.local_model import LocalModel

        kwargs = dict(
            common,
            endpoint=endpoint,
            reasoning_effort=settings.resolved_reasoning_effort,
            capture_token_logprobs=settings.capture_token_logprobs,
        )
        if settings.enable_thinking is not None:
            kwargs["enable_thinking"] = settings.enable_thinking
        if settings.base_url is not None:
            kwargs["base_url"] = settings.base_url
        return LocalModel(**kwargs)
    if mt == ModelType.OPENROUTER:
        from infra.models.openrouter_model import OpenRouterModel

        kwargs = dict(
            common,
            endpoint=endpoint,
            reasoning_effort=settings.resolved_reasoning_effort,
            capture_token_logprobs=settings.capture_token_logprobs,
            require_token_logprobs=settings.require_token_logprobs,
            provider_order=settings.provider_order,
            quantizations=settings.quantizations,
            allow_fallbacks=settings.allow_fallbacks,
            data_collection=settings.data_collection,
        )
        if settings.enable_thinking is not None:
            kwargs["enable_thinking"] = settings.enable_thinking
        if settings.base_url is not None:
            kwargs["base_url"] = settings.base_url
        return OpenRouterModel(**kwargs)
    if mt == ModelType.ALIBABA:
        from infra.models.alibaba_model import AlibabaModel

        kwargs = dict(common, endpoint=endpoint, reasoning_effort=settings.resolved_reasoning_effort)
        if settings.enable_thinking is not None:
            kwargs["enable_thinking"] = settings.enable_thinking
        return AlibabaModel(**kwargs)
    if mt == ModelType.GOOGLE:
        from infra.models.google_model import GoogleModel

        return GoogleModel(**common, endpoint=endpoint)

    raise ValueError(f"unknown model type {mt}")
