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


def demo_provider_setup(logger: logging.Logger) -> None:
    """Create two provider configs and log success or failure."""

    configs: Iterable[LLMProviderConfig] = [
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

    for config in configs:
        try:
            create_llm(config)
        except RuntimeError as exc:
            logger.error("LLM provider '%s' failed to initialize: %s", config.provider, exc)
        else:
            logger.info(
                "LLM provider '%s' initialized with model '%s'.", config.provider, config.model
            )


def demo_patch_retry_workflow(logger: logging.Logger) -> None:
    """Simulate a patch response that fails once, then succeeds on retry."""

    current_config = {"analysis": {"top_n": 10, "target_vol": 0.12}}
    instruction = "Increase top_n to 25 to expand the selection universe."
    schema = select_schema_sections(load_compact_schema(), instruction)
    prompt = build_config_patch_prompt(
        current_config=format_config_for_prompt(current_config),
        allowed_schema=json.dumps(schema, indent=2, ensure_ascii=True),
        instruction=instruction,
    )
    logger.info("Prompt preview:\n%s", prompt[:400])

    responses = iter(
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
                    "summary": "Increase top_n to 25.",
                }
            ),
        ]
    )

    def response_provider(attempt: int, last_error: Exception | None) -> str:
        _ = attempt, last_error
        return next(responses)

    patch = parse_config_patch_with_retries(response_provider, retries=1, logger=logger)
    logger.info("Parsed patch summary: %s", patch.summary)
    updated = apply_config_patch(current_config, patch)
    logger.info("Config diff:\n%s", diff_configs(current_config, updated))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger("nl-config-demo")
    logger.info("Starting NL config demo.")
    demo_provider_setup(logger)
    demo_patch_retry_workflow(logger)


if __name__ == "__main__":
    main()
