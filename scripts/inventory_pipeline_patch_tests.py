from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

PIPELINE_MODULE = "trend_analysis.pipeline"
IMPORTLIB_NAMES = {"import_module"}


@dataclass(frozen=True)
class PatchHit:
    path: Path
    line: int
    column: int
    kind: str
    snippet: str


def _iter_python_files(paths: Sequence[Path]) -> Iterable[Path]:
    for root in paths:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if path.name.startswith("."):
                continue
            yield path


def _extract_import_aliases(tree: ast.AST) -> tuple[set[str], set[str]]:
    pipeline_aliases: set[str] = set()
    trend_package_aliases: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == PIPELINE_MODULE and alias.asname:
                    pipeline_aliases.add(alias.asname)
                if alias.name == "trend_analysis":
                    trend_package_aliases.add(alias.asname or alias.name)
                if alias.name == PIPELINE_MODULE and not alias.asname:
                    trend_package_aliases.add("trend_analysis")
        elif isinstance(node, ast.ImportFrom):
            if node.module == "trend_analysis":
                for alias in node.names:
                    if alias.name == "pipeline":
                        pipeline_aliases.add(alias.asname or alias.name)
                    if alias.name == "trend_analysis":
                        trend_package_aliases.add(alias.asname or alias.name)
    return pipeline_aliases, trend_package_aliases


def _extract_dynamic_aliases(tree: ast.AST) -> set[str]:
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            target = None
            value = None
            if isinstance(node, ast.Assign):
                if not node.targets:
                    continue
                target = node.targets[0]
                value = node.value
            else:
                target = node.target
                value = node.value
            if not isinstance(target, ast.Name) or not isinstance(value, ast.Call):
                continue
            func = value.func
            if isinstance(func, ast.Attribute) and func.attr in IMPORTLIB_NAMES:
                if func.attr == "import_module" and _is_pipeline_string(value.args):
                    aliases.add(target.id)
            if isinstance(func, ast.Name) and func.id in IMPORTLIB_NAMES:
                if _is_pipeline_string(value.args):
                    aliases.add(target.id)
    return aliases


def _is_pipeline_string(nodes: Sequence[ast.AST]) -> bool:
    if not nodes:
        return False
    arg = nodes[0]
    return (
        isinstance(arg, ast.Constant)
        and isinstance(arg.value, str)
        and arg.value == PIPELINE_MODULE
    )


def _is_pipeline_ref(
    node: ast.AST,
    *,
    pipeline_aliases: set[str],
    trend_package_aliases: set[str],
) -> bool:
    if isinstance(node, ast.Name):
        return node.id in pipeline_aliases
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name) and node.attr == "pipeline":
            return node.value.id in trend_package_aliases
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.startswith(PIPELINE_MODULE)
    return False


def _get_source_line(source_lines: list[str], line_no: int) -> str:
    if 0 < line_no <= len(source_lines):
        return source_lines[line_no - 1].rstrip()
    return ""


def _collect_patch_hits(path: Path) -> list[PatchHit]:
    source = path.read_text(encoding="utf-8")
    source_lines = source.splitlines()
    tree = ast.parse(source)
    pipeline_aliases, trend_package_aliases = _extract_import_aliases(tree)
    pipeline_aliases |= _extract_dynamic_aliases(tree)

    hits: list[PatchHit] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "setattr":
            if node.args and _is_pipeline_ref(
                node.args[0],
                pipeline_aliases=pipeline_aliases,
                trend_package_aliases=trend_package_aliases,
            ):
                line = getattr(node, "lineno", 0)
                column = getattr(node, "col_offset", 0)
                hits.append(
                    PatchHit(
                        path=path,
                        line=line,
                        column=column,
                        kind="monkeypatch.setattr",
                        snippet=_get_source_line(source_lines, line),
                    )
                )
        if isinstance(func, ast.Name) and func.id == "patch":
            if node.args and _is_pipeline_ref(
                node.args[0],
                pipeline_aliases=pipeline_aliases,
                trend_package_aliases=trend_package_aliases,
            ):
                line = getattr(node, "lineno", 0)
                column = getattr(node, "col_offset", 0)
                hits.append(
                    PatchHit(
                        path=path,
                        line=line,
                        column=column,
                        kind="patch",
                        snippet=_get_source_line(source_lines, line),
                    )
                )
        if isinstance(func, ast.Attribute) and func.attr == "patch":
            if node.args and _is_pipeline_ref(
                node.args[0],
                pipeline_aliases=pipeline_aliases,
                trend_package_aliases=trend_package_aliases,
            ):
                line = getattr(node, "lineno", 0)
                column = getattr(node, "col_offset", 0)
                hits.append(
                    PatchHit(
                        path=path,
                        line=line,
                        column=column,
                        kind="patch",
                        snippet=_get_source_line(source_lines, line),
                    )
                )

    return hits


def _format_hits(hits: Sequence[PatchHit]) -> str:
    lines: list[str] = []
    for hit in sorted(hits, key=lambda h: (str(h.path), h.line, h.column)):
        lines.append(f"{hit.path}:{hit.line}:{hit.column}: {hit.kind}: {hit.snippet}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inventory tests patching trend_analysis.pipeline symbols.",
    )
    parser.add_argument(
        "--paths",
        nargs="*",
        default=["tests"],
        help="Paths to scan (default: tests).",
    )
    args = parser.parse_args()

    roots = [Path(p) for p in args.paths]
    hits: list[PatchHit] = []
    for path in _iter_python_files(roots):
        hits.extend(_collect_patch_hits(path))

    print(_format_hits(hits))
    print(f"\nTotal hits: {len(hits)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
