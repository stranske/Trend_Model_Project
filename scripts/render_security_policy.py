"""Render a comprehensive security policy document."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def render_security_policy() -> str:
    lines = [
        "# Security Policy",
        "",
        "## Reporting a Vulnerability",
        "",
        "Report suspected vulnerabilities by emailing security@example.com.",
        "Include a description, reproduction steps, affected versions, and any",
        "mitigations already attempted. We will acknowledge receipt within 72 hours.",
        "",
        "## Supported Versions",
        "",
        "- Latest release series",
        "- Most recent minor version on the previous major release",
        "",
        "## Security Controls",
        "",
        "- Input validation is enforced on configuration parsing and CLI arguments.",
        "- Path traversal protections ensure file access stays within expected roots.",
        "- Sensitive runtime confirmations require explicit user approval in CLI, API,",
        "  and Streamlit entry points.",
        "- CI workflows avoid executing untrusted code with elevated privileges.",
        "- Dependency updates are locked and reviewed via pinned lockfiles.",
        "",
        "## Potential Vulnerabilities",
        "",
        "- Malicious configuration files that attempt to escape configured paths.",
        "- Prompt-injection or obfuscated inputs that bypass safety checks.",
        "- Supply-chain risks from compromised dependencies or build artifacts.",
        "- Unauthorized access to model output or sensitive report data.",
        "",
        "## Mitigation Strategies",
        "",
        "- Enforce allowlists and canonicalize paths before file operations.",
        "- Validate configuration keys against schema definitions with clear errors.",
        "- Add test coverage for injection vectors and confirmation gates.",
        "- Use minimal privileges for automation and avoid reusing elevated tokens.",
        "- Rotate and audit dependencies; verify hashes where applicable.",
        "",
        "## Security Testing",
        "",
        "- Unit tests cover path traversal, config validation, and prompt hygiene.",
        "- Integration tests validate confirmation flows across interfaces.",
        "",
        "## Disclosure Policy",
        "",
        "We follow coordinated disclosure. Do not publicly disclose vulnerabilities",
        "until we have had a reasonable opportunity to investigate and respond.",
        "",
    ]
    return "\n".join(lines)


def _write_output(output_path: Path, payload: str) -> None:
    output_path.write_text(payload, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a comprehensive security policy document.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path; defaults to stdout when omitted.",
    )
    args = parser.parse_args(argv)
    payload = render_security_policy()
    if args.output:
        _write_output(args.output, payload)
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
