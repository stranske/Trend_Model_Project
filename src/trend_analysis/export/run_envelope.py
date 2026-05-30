"""Build and write the single replayable run-envelope JSON (``run_contract``).

The *run envelope* is an export-time projection that ties together the data
already produced by a run into one independently re-runnable JSON object:
who/what/why (``actor``/``intent``), validated input hashes, named outputs (a
reference to the existing artifact ``manifest.json`` rather than a duplicate),
collected ``warnings``, ``cost_latency``, ``provenance`` and the run
``diagnostic``.

It deliberately **references** the existing
:func:`trend_analysis.reporting.run_artifacts.write_run_artifacts` manifest and
does not replace :class:`trend_analysis.api.RunResult` or either existing
manifest. See ``run_envelope_schema.json`` for the validated shape.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from trend_analysis.util.hash import normalise_for_json, sha256_config, sha256_text

if TYPE_CHECKING:  # pragma: no cover - typing only
    from trend_analysis.api import RunResult

SCHEMA_VERSION = "trend.run_envelope/1"


@dataclass
class RunEnvelope:
    """In-memory representation of the run envelope.

    Mirrors the JSON shape validated by ``run_envelope_schema.json``. Use
    :func:`to_run_envelope` to build the serialisable ``dict`` from a
    :class:`~trend_analysis.api.RunResult`.
    """

    run_id: str
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    provenance: dict[str, Any]
    cost_latency: dict[str, Any]
    warnings: list[dict[str, Any]] = field(default_factory=list)
    diagnostic: dict[str, Any] | None = None
    actor: str | None = None
    intent: str | None = None
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-friendly envelope mapping (omitting empty actor/intent)."""

        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "provenance": self.provenance,
            "cost_latency": self.cost_latency,
            "warnings": self.warnings,
            "diagnostic": self.diagnostic,
        }
        if self.actor is not None:
            payload["actor"] = self.actor
        if self.intent is not None:
            payload["intent"] = self.intent
        return payload


def _serialise_diagnostic(diagnostic: Any) -> dict[str, Any] | None:
    """Return a JSON-friendly projection of a ``DiagnosticPayload`` (or ``None``)."""

    if diagnostic is None:
        return None
    if dataclasses.is_dataclass(diagnostic) and not isinstance(diagnostic, type):
        return cast("dict[str, Any]", normalise_for_json(dataclasses.asdict(diagnostic)))
    if hasattr(diagnostic, "model_dump"):
        return cast("dict[str, Any]", normalise_for_json(diagnostic.model_dump()))
    if isinstance(diagnostic, dict):
        return cast("dict[str, Any]", normalise_for_json(diagnostic))
    return {"message": str(diagnostic)}


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):  # pragma: no cover - defensive
        return {}


def to_run_envelope(
    result: RunResult,
    *,
    config: Any,
    manifest_path: Path | str,
    timings: dict[str, float] | None = None,
    warnings: list[dict[str, Any]] | None = None,
    actor: str | None = None,
    intent: str | None = None,
) -> dict[str, Any]:
    """Build the run-envelope ``dict`` for *result*, cross-linking the manifest.

    Parameters
    ----------
    result:
        The :class:`~trend_analysis.api.RunResult` from :func:`run_simulation`.
    config:
        The configuration object/mapping (hashed via :func:`sha256_config`).
    manifest_path:
        Path to the existing ``manifest.json`` produced by
        :func:`write_run_artifacts`; the envelope references it and reuses its
        ``input_sha256``/``git_hash``/artifact names rather than duplicating them.
    timings:
        Optional timing mapping (e.g. ``{"wall_ms": 12.3}``). Falls back to
        ``result.timings`` when omitted.
    warnings:
        Optional structured warnings. Falls back to ``result.warnings`` when
        omitted.
    actor, intent:
        Optional caller-supplied provenance describing who ran it and why.
    """

    manifest_path = Path(manifest_path)
    manifest = _load_manifest(manifest_path)

    config_sha256 = sha256_config(config)
    input_sha256 = manifest.get("input_sha256")
    seed = getattr(result, "seed", None)

    # Content-addressed run_id, identical scheme to export/bundle.py:75-81.
    run_id_src = "|".join(
        part
        for part in (
            input_sha256 if isinstance(input_sha256, str) else None,
            config_sha256,
            str(seed) if seed is not None else "",
        )
        if part
    )
    run_id = sha256_text(run_id_src)

    artifacts = manifest.get("artifacts", [])
    artifact_names = [
        a.get("name")
        for a in artifacts
        if isinstance(a, dict) and a.get("name") is not None
    ]

    effective_timings = timings if timings is not None else (getattr(result, "timings", None) or {})
    wall_ms = effective_timings.get("wall_ms")
    cost_latency: dict[str, Any] = {"wall_ms": float(wall_ms) if wall_ms is not None else None}
    peak_rss_kb = effective_timings.get("peak_rss_kb")
    if peak_rss_kb is not None:
        cost_latency["peak_rss_kb"] = peak_rss_kb

    effective_warnings = (
        warnings if warnings is not None else (getattr(result, "warnings", None) or [])
    )

    envelope = RunEnvelope(
        run_id=run_id,
        inputs={"config_sha256": config_sha256, "input_sha256": input_sha256},
        outputs={"manifest": manifest_path.name, "artifacts": artifact_names},
        provenance={
            "git_hash": manifest.get("git_hash"),
            "environment": normalise_for_json(getattr(result, "environment", {}) or {}),
        },
        cost_latency=cost_latency,
        warnings=[normalise_for_json(w) for w in effective_warnings],
        diagnostic=_serialise_diagnostic(getattr(result, "diagnostic", None)),
        actor=actor,
        intent=intent,
    )
    return envelope.to_dict()


def write_run_envelope(
    result: RunResult,
    *,
    config: Any,
    manifest_path: Path | str,
    run_dir: Path | str | None = None,
    timings: dict[str, float] | None = None,
    warnings: list[dict[str, Any]] | None = None,
    actor: str | None = None,
    intent: str | None = None,
) -> Path:
    """Write ``run_envelope.json`` next to the manifest and return its path."""

    manifest_path = Path(manifest_path)
    target_dir = Path(run_dir) if run_dir is not None else manifest_path.parent
    envelope = to_run_envelope(
        result,
        config=config,
        manifest_path=manifest_path,
        timings=timings,
        warnings=warnings,
        actor=actor,
        intent=intent,
    )
    out_path = target_dir / "run_envelope.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
    return out_path
