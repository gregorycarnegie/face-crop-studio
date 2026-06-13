#!/usr/bin/env python3
"""Rank likely "god files" by size, complexity, and breadth metrics."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


SOURCE_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cs",
    ".go",
    ".h",
    ".hpp",
    ".java",
    ".js",
    ".jsx",
    ".kt",
    ".py",
    ".rs",
    ".ts",
    ".tsx",
}

SKIP_DIRS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".svn",
    ".venv",
    "__pycache__",
    "build",
    "coverage",
    "dist",
    "node_modules",
    "target",
    "vendor",
}

DEFAULT_MIN_SCORE = 100.0
REASON_THRESHOLDS = {
    "total_lines": 500,
    "code_lines": 350,
    "functions": 25,
    "types": 12,
    "impl_blocks": 15,
    "decision_points": 90,
    "imports": 45,
    "fanout": 25,
    "max_function_lines": 120,
}


@dataclass(frozen=True)
class Finding:
    path: str
    score: float
    total_lines: int
    code_lines: int
    functions: int
    types: int
    impl_blocks: int
    decision_points: int
    imports: int
    fanout: int
    max_function_lines: int
    reasons: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="*", default=["."], help="Files or directories to scan.")
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    parser.add_argument("--top", type=int, default=0, help="Print only the top N findings.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    parser.add_argument("--include-tests", action="store_true", help="Include test-only files.")
    parser.add_argument(
        "--extensions",
        default=",".join(sorted(SOURCE_EXTENSIONS)),
        help="Comma-separated source extensions to scan.",
    )
    parser.add_argument(
        "--skip-dir",
        action="append",
        default=[],
        help="Additional directory name to skip. Can be passed more than once.",
    )
    return parser.parse_args()


def source_files(
    roots: list[str],
    extensions: set[str],
    skip_dirs: set[str],
    include_tests: bool,
) -> list[Path]:
    self_path = Path(__file__).resolve()
    files: list[Path] = []
    seen: set[Path] = set()

    for raw_root in roots:
        root = Path(raw_root)
        candidates = [root] if root.is_file() else root.rglob("*")
        for path in candidates:
            if not path.is_file() or path.suffix not in extensions:
                continue
            resolved = path.resolve()
            if resolved == self_path or resolved in seen:
                continue
            if any(part in skip_dirs for part in path.parts):
                continue
            if not include_tests and looks_test_only(path):
                continue
            seen.add(resolved)
            files.append(path)

    return files


def looks_test_only(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    name = path.name.lower()
    return (
        "tests" in parts
        or "__tests__" in parts
        or name == "tests.rs"
        or name.startswith("test_")
        or name.endswith(("_test.py", ".test.ts", ".test.tsx", ".spec.ts", ".spec.tsx"))
    )


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", lambda match: "\n" * match.group(0).count("\n"), text, flags=re.S)
    lines = []
    for line in text.splitlines():
        line = re.sub(r"//.*", "", line)
        line = re.sub(r"#.*", "", line)
        lines.append(line)
    return "\n".join(lines)


def count_functions(code: str, suffix: str) -> int:
    if suffix == ".py":
        return len(re.findall(r"(?m)^\s*(?:async\s+)?def\s+[A-Za-z_]\w*\s*\(", code))
    if suffix in {".js", ".jsx", ".ts", ".tsx"}:
        return len(re.findall(r"\bfunction\s+[A-Za-z_$][\w$]*\s*\(", code)) + len(
            re.findall(r"(?m)^\s*(?:export\s+)?(?:const|let|var)\s+[A-Za-z_$][\w$]*\s*=", code)
        )
    return len(re.findall(r"\bfn\s+[A-Za-z_]\w*\s*(?:<|\()", code))


def count_types(code: str, suffix: str) -> int:
    if suffix == ".py":
        return len(re.findall(r"(?m)^\s*class\s+[A-Za-z_]\w*", code))
    if suffix in {".js", ".jsx", ".ts", ".tsx"}:
        return len(re.findall(r"\b(?:class|interface|type)\s+[A-Za-z_$][\w$]*", code))
    return len(re.findall(r"\b(?:struct|enum|trait|type)\s+[A-Za-z_]\w*", code))


def count_imports(code: str, suffix: str) -> int:
    if suffix == ".py":
        return len(re.findall(r"(?m)^\s*(?:from\s+\S+\s+import|import)\s+", code))
    if suffix in {".js", ".jsx", ".ts", ".tsx"}:
        return len(re.findall(r"(?m)^\s*import\s+", code))
    return len(re.findall(r"(?m)^\s*use\s+", code))


def count_fanout(code: str, suffix: str) -> int:
    modules: set[str] = set()
    if suffix == ".py":
        pattern = r"(?m)^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))"
        for match in re.finditer(pattern, code):
            modules.add((match.group(1) or match.group(2)).split(".")[0])
        return len(modules)

    if suffix in {".js", ".jsx", ".ts", ".tsx"}:
        for match in re.finditer(r"""(?m)^\s*import\s+.*?\s+from\s+['"]([^'"]+)['"]""", code):
            modules.add(match.group(1))
        return len(modules)

    for match in re.finditer(r"(?m)^\s*use\s+([^;]+);", code):
        path = match.group(1).strip().split("{", 1)[0].strip(": ")
        parts = path.split("::")
        if parts[0] in {"crate", "self", "super"} and len(parts) > 1:
            modules.add("::".join(parts[:2]))
        elif parts:
            modules.add(parts[0])
    return len(modules)


def count_decisions(code: str) -> int:
    keywords = re.findall(r"\b(?:if|else\s+if|match|for|while|loop|switch|case|catch)\b", code)
    return len(keywords) + code.count("&&") + code.count("||") + code.count("?")


def max_function_lines(code: str, suffix: str) -> int:
    lines = code.splitlines()
    starts = [i for i, line in enumerate(lines) if looks_like_function_start(line, suffix)]
    best = 0

    for start in starts:
        depth = 0
        saw_body = False
        for end in range(start, len(lines)):
            if "{" in lines[end]:
                saw_body = True
            depth += lines[end].count("{") - lines[end].count("}")
            if saw_body and depth <= 0:
                best = max(best, end - start + 1)
                break

    if suffix == ".py":
        for index, start in enumerate(starts):
            end = starts[index + 1] if index + 1 < len(starts) else len(lines)
            best = max(best, end - start)

    return best


def looks_like_function_start(line: str, suffix: str) -> bool:
    if suffix == ".py":
        return bool(re.match(r"^\s*(?:async\s+)?def\s+[A-Za-z_]\w*\s*\(", line))
    if suffix in {".js", ".jsx", ".ts", ".tsx"}:
        return bool(re.search(r"\bfunction\s+[A-Za-z_$][\w$]*\s*\(", line))
    return bool(re.search(r"\bfn\s+[A-Za-z_]\w*\s*(?:<|\()", line))


def score(metrics: dict[str, int]) -> float:
    value = 0.0
    value += max(0, metrics["total_lines"] - 300) * 0.20
    value += max(0, metrics["code_lines"] - 220) * 0.30
    value += max(0, metrics["functions"] - 15) * 4.00
    value += max(0, metrics["types"] - 6) * 6.00
    value += max(0, metrics["impl_blocks"] - 8) * 3.00
    value += max(0, metrics["decision_points"] - 50) * 1.20
    value += max(0, metrics["imports"] - 25) * 1.50
    value += max(0, metrics["fanout"] - 15) * 2.50
    value += max(0, metrics["max_function_lines"] - 80) * 0.80
    return round(value, 1)


def analyze(path: Path, repo_root: Path, min_score: float) -> Finding | None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    code = strip_comments(text)
    metrics = {
        "total_lines": len(text.splitlines()),
        "code_lines": sum(1 for line in code.splitlines() if line.strip()),
        "functions": count_functions(code, path.suffix),
        "types": count_types(code, path.suffix),
        "impl_blocks": len(re.findall(r"\bimpl(?:\s*<[^>{;]*>)?\s+", code)) if path.suffix == ".rs" else 0,
        "decision_points": count_decisions(code),
        "imports": count_imports(code, path.suffix),
        "fanout": count_fanout(code, path.suffix),
        "max_function_lines": max_function_lines(code, path.suffix),
    }
    file_score = score(metrics)
    if file_score < min_score:
        return None

    labels = {
        "total_lines": "total lines",
        "code_lines": "code lines",
        "functions": "functions",
        "types": "types",
        "impl_blocks": "impl blocks",
        "decision_points": "decision points",
        "imports": "imports",
        "fanout": "import fanout",
        "max_function_lines": "longest function",
    }
    reasons = [
        f"{labels[key]} {metrics[key]} >= {limit}"
        for key, limit in REASON_THRESHOLDS.items()
        if metrics[key] >= limit
    ]
    reasons.append(f"score {file_score:.1f} >= {min_score:.1f}")

    return Finding(
        path=path.resolve().relative_to(repo_root).as_posix(),
        score=file_score,
        reasons=reasons,
        **metrics,
    )


def print_table(findings: list[Finding]) -> None:
    if not findings:
        print("No god files found with the current thresholds.")
        return

    header = "score  lines  code  fn  type  impl  branch  use  fanout  max_fn  path"
    print(header)
    print("-" * len(header))
    for item in findings:
        print(
            f"{item.score:5.1f}  {item.total_lines:5d}  {item.code_lines:4d}  "
            f"{item.functions:2d}  {item.types:4d}  {item.impl_blocks:4d}  "
            f"{item.decision_points:6d}  {item.imports:3d}  {item.fanout:6d}  "
            f"{item.max_function_lines:6d}  {item.path}"
        )
        print(f"       reasons: {'; '.join(item.reasons)}")

    print("\nFiles:")
    for item in findings:
        print(item.path)


def main() -> int:
    args = parse_args()
    repo_root = Path.cwd().resolve()
    extensions = {
        ext if ext.startswith(".") else f".{ext}"
        for ext in (part.strip() for part in args.extensions.split(","))
        if ext
    }
    paths = source_files(args.roots, extensions, SKIP_DIRS | set(args.skip_dir), args.include_tests)
    findings = [item for path in paths if (item := analyze(path, repo_root, args.min_score))]
    findings.sort(key=lambda item: (-item.score, -item.total_lines, item.path))
    if args.top:
        findings = findings[: args.top]

    if args.json:
        print(json.dumps([asdict(item) for item in findings], indent=2))
    else:
        print_table(findings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
