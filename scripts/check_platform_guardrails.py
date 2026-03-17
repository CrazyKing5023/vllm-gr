#!/usr/bin/env python3
# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""Static guardrails for platform-specific leakage into shared layers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import ast
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "vllm_gr"

ALLOWED_PLATFORM_DIRS = {
    Path("vllm_gr/utils"),
    Path("vllm_gr/ops"),
    Path("vllm_gr/worker"),
}

FILE_RULE_ALLOWLIST: dict[Path, set[str]] = {
    Path("vllm_gr/engine/arg_utils.py"): {"current_platform"},
}


@dataclass(frozen=True)
class GuardrailViolation:
    path: Path
    line: int
    rule: str
    detail: str


def _full_attr_name(node: ast.AST) -> str | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return None


class PlatformLeakVisitor(ast.NodeVisitor):
    def __init__(self, rel_path: Path):
        self.rel_path = rel_path
        self.allowed_rules = FILE_RULE_ALLOWLIST.get(rel_path, set())
        self.violations: list[tuple[int, str, str]] = []

    def _record(self, node: ast.AST, rule: str, detail: str) -> None:
        if rule in self.allowed_rules:
            return
        self.violations.append((getattr(node, "lineno", 1), rule, detail))

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "torch_npu" or alias.name.startswith("torch_npu."):
                self._record(node, "torch_npu", f"direct import: {alias.name}")
            if alias.name == "mxrec" or alias.name.startswith("mxrec."):
                self._record(node, "mxrec", f"direct import: {alias.name}")
            if alias.name in {"fbgemm", "fbgemm_gpu"} or alias.name.startswith("fbgemm."):
                self._record(node, "fbgemm", f"direct import: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if module == "torch_npu" or module.startswith("torch_npu."):
            self._record(node, "torch_npu", f"from import: {module}")
        if module == "mxrec" or module.startswith("mxrec."):
            self._record(node, "mxrec", f"from import: {module}")
        if module in {"fbgemm", "fbgemm_gpu"} or module.startswith("fbgemm."):
            self._record(node, "fbgemm", f"from import: {module}")
        if module == "vllm.platforms":
            for alias in node.names:
                if alias.name == "current_platform":
                    self._record(
                        node,
                        "current_platform",
                        "shared layer importing current_platform",
                    )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        full_name = _full_attr_name(node.func)
        if full_name == "torch.cuda.is_available":
            self._record(node, "is_available", "torch.cuda.is_available()")
        elif full_name == "torch.npu.is_available":
            self._record(node, "is_available", "torch.npu.is_available()")
        elif full_name and full_name.startswith("torch.ops.fbgemm."):
            self._record(node, "fbgemm", full_name)
        elif full_name and full_name.startswith("torch.ops.mxrec."):
            self._record(node, "mxrec", full_name)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        full_name = _full_attr_name(node)
        if full_name == "current_platform.device_type":
            self._record(
                node,
                "current_platform",
                "shared layer reading current_platform.device_type",
            )
        elif full_name and full_name.startswith("torch.ops.fbgemm."):
            self._record(node, "fbgemm", full_name)
        elif full_name and full_name.startswith("torch.ops.mxrec."):
            self._record(node, "mxrec", full_name)
        self.generic_visit(node)


def is_shared_python_file(path: Path, root: Path) -> bool:
    rel_path = path.relative_to(root)
    if path.suffix != ".py":
        return False
    if any(part == "__pycache__" for part in rel_path.parts):
        return False
    return not any(
        rel_path.is_relative_to(allowed_root)
        for allowed_root in ALLOWED_PLATFORM_DIRS
    )


def collect_guardrail_violations(root: Path = REPO_ROOT) -> list[GuardrailViolation]:
    violations: list[GuardrailViolation] = []
    for path in sorted((root / "vllm_gr").rglob("*.py")):
        if not is_shared_python_file(path, root):
            continue

        rel_path = path.relative_to(root)
        tree = ast.parse(path.read_text(), filename=str(rel_path))
        visitor = PlatformLeakVisitor(rel_path)
        visitor.visit(tree)

        for line, rule, detail in visitor.violations:
            violations.append(
                GuardrailViolation(
                    path=rel_path,
                    line=line,
                    rule=rule,
                    detail=detail,
                )
            )
    return violations


def format_violations(violations: list[GuardrailViolation]) -> str:
    lines = [
        (
            f"{violation.path}:{violation.line}: "
            f"[{violation.rule}] {violation.detail}"
        )
        for violation in violations
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check for platform-specific leakage in shared vllm_gr layers",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root to scan (defaults to current repo root)",
    )
    args = parser.parse_args()

    violations = collect_guardrail_violations(args.root.resolve())
    if not violations:
        print("Platform guardrails: OK")
        return 0

    print("Platform guardrails: violations detected", file=sys.stderr)
    print(format_violations(violations), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
