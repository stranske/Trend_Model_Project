import os
from pathlib import Path

from trend_analysis.config.patch import apply_and_diff
from trend_analysis.llm import (
    ConfigPatchChain,
    LLMProviderConfig,
    build_config_patch_prompt,
    create_llm,
)
from trend_analysis.llm.schema import load_compact_schema


def _resolve_api_key(provider: str) -> str | None:
    override = os.environ.get("TREND_LLM_API_KEY")
    if override:
        return override
    provider_key = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
    env_var = provider_key.get(provider, "")
    return os.environ.get(env_var) if env_var else None


def main() -> None:
    instruction = "use risk parity weighting"
    config_path = Path("config/demo.yml")
    provider = os.environ.get("TREND_LLM_PROVIDER", "openai").lower()

    config = LLMProviderConfig(
        provider=provider,
        model=os.environ.get("TREND_LLM_MODEL", "gpt-4o-mini"),
        api_key=_resolve_api_key(provider),
        base_url=os.environ.get("TREND_LLM_BASE_URL"),
        organization=os.environ.get("TREND_LLM_ORG"),
    )
    llm = create_llm(config)
    chain = ConfigPatchChain.from_env(
        llm=llm,
        schema=load_compact_schema(),
        prompt_builder=build_config_patch_prompt,
    )

    patch = chain.run(
        current_config=config_path.read_text(encoding="utf-8"), instruction=instruction
    )
    _, diff = apply_and_diff(config_path, patch)

    print(patch.summary)
    print(diff)


if __name__ == "__main__":
    main()
