"""Demo for NL-driven config patches with provider setup, retries, and logging."""

from __future__ import annotations

import json
import logging
import os
from typing import Iterable

from trend_analysis.config.patch import (
    apply_config_patch,
    diff_configs,
    parse_config_patch_with_retries,
)
from trend_analysis.llm import LLMProviderConfig, create_llm
from trend_analysis.llm.prompts import (
    build_config_patch_prompt,
    format_config_for_prompt,
)
from trend_analysis.llm.schema import load_compact_schema, select_schema_sections


def build_provider_configs() -> list[LLMProviderConfig]:
    """Build provider configurations from environment defaults."""

    return [
        LLMProviderConfig(
            provider="openai",
            model=os.environ.get("TREND_OPENAI_MODEL", "gpt-4o-mini"),
            api_key=os.environ.get("OPENAI_API_KEY"),
        ),
        LLMProviderConfig(
            provider="anthropic",
            model=os.environ.get("TREND_ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
        ),
    ]


def demo_provider_setup(logger: logging.Logger) -> list[tuple[LLMProviderConfig, object]]:
    """Create provider configs, instantiate clients, and log outcomes."""

    llms: list[tuple[LLMProviderConfig, object]] = []
    for config in build_provider_configs():
        try:
            llm = create_llm(config)
        except RuntimeError as exc:
            logger.error("LLM provider '%s' failed to initialize: %s", config.provider, exc)
        else:
            logger.info(
                "LLM provider '%s' initialized with model '%s'.", config.provider, config.model
            )
            llms.append((config, llm))
    return llms


def demo_provider_usage(
    logger: logging.Logger, llms: Iterable[tuple[LLMProviderConfig, object]]
) -> None:
    """Show how to invoke providers with an optional live call."""

    run_live = os.environ.get("TREND_DEMO_RUN_LLM", "").lower() in {"1", "true", "yes"}
    for config, llm in llms:
        if not run_live:
            logger.info(
                "Skipping live call for provider '%s'. Set TREND_DEMO_RUN_LLM=1 to enable.",
                config.provider,
            )
            continue
        try:
            response = llm.invoke("Provide a one-sentence summary for a trend model config demo.")
        except Exception as exc:
            logger.error("LLM provider '%s' invocation failed: %s", config.provider, exc)
        else:
            logger.info("LLM provider '%s' sample response: %s", config.provider, response)


def demo_patch_retry_workflow(logger: logging.Logger) -> None:
    """Simulate patch parsing for two providers, including a retry scenario."""

    current_config = {"analysis": {"top_n": 10, "target_vol": 0.12}}
    instruction = "Increase top_n to 25 to expand the selection universe."
    schema = select_schema_sections(load_compact_schema(), instruction)
    prompt = build_config_patch_prompt(
        current_config=format_config_for_prompt(current_config),
        allowed_schema=json.dumps(schema, indent=2, ensure_ascii=True),
        instruction=instruction,
    )
    logger.info("Prompt preview:\n%s", prompt[:400])

    scenarios = [
        (
            LLMProviderConfig(provider="openai", model="gpt-4o-mini"),
            [
                json.dumps(
                    {
                        "operations": [
                            {
                                "op": "set",
                                "path": "analysis.top_n",
                                "value": 25,
                                "rationale": "Increase the selection count.",
                            }
                        ],
                        "summary": "Increase top_n to 25.",
                    }
                )
            ],
            1,
        ),
        (
            LLMProviderConfig(provider="anthropic", model="claude-3-5-sonnet-20241022"),
            [
                "not-json-response",
                json.dumps(
                    {
                        "operations": [
                            {
                                "op": "set",
                                "path": "analysis.top_n",
                                "value": 25,
                                "rationale": "Increase the selection count.",
                            }
                        ],
                        "summary": "Increase top_n to 25 after retry.",
                    }
                ),
            ],
            2,
        ),
    ]

    for config, response_sequence, retries in scenarios:
        logger.info(
            "Parsing patch with provider '%s' (model '%s', retries=%s).",
            config.provider,
            config.model,
            retries,
        )
        responses = iter(response_sequence)

        def response_provider(attempt: int, last_error: Exception | None) -> str:
            if last_error:
                logger.info(
                    "Provider '%s' retry attempt %s after error: %s",
                    config.provider,
                    attempt + 1,
                    last_error,
                )
            else:
                logger.info("Provider '%s' attempt %s.", config.provider, attempt + 1)
            return next(responses)

        patch = parse_config_patch_with_retries(response_provider, retries=retries, logger=logger)
        logger.info("Parsed patch summary for '%s': %s", config.provider, patch.summary)
        updated = apply_config_patch(current_config, patch)
        logger.info(
            "Config diff for '%s':\n%s", config.provider, diff_configs(current_config, updated)
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger("nl-config-demo")
    logger.info("Starting NL config demo.")
    llms = demo_provider_setup(logger)
    demo_provider_usage(logger, llms)
    demo_patch_retry_workflow(logger)


if __name__ == "__main__":
    main()
