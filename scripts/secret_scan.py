"""Secret-scan CI gate for RoadGuard AI (task 16.6).

Fails (exit 1) if any tracked text file appears to contain a hardcoded secret -
AWS access keys, private keys, generic api/secret assignments, or Slack tokens.
Keeps credentials out of source and (via the frontend glob) out of UI bundles
(Requirement 18.5). Offline, no dependencies beyond the stdlib.

Usage:
    python scripts/secret_scan.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (name, compiled pattern). Patterns are intentionally conservative.
_PATTERNS = [
    ("aws_access_key_id", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("aws_secret_access_key", re.compile(r"aws_secret_access_key\s*=\s*['\"]?[A-Za-z0-9/+=]{40}")),
    ("private_key_block", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")),
    ("slack_token", re.compile(r"xox[baprs]-[0-9A-Za-z-]{10,}")),
    ("generic_secret_assign", re.compile(r"(?i)(secret|password|passwd|api[_-]?key|token)\s*[:=]\s*['\"][^'\"\n]{12,}['\"]")),
]

# Only scan source-ish text; skip vendored/build/binary dirs.
_INCLUDE_SUFFIXES = {".py", ".ts", ".tsx", ".js", ".jsx", ".json", ".md", ".yaml", ".yml", ".sh", ".env", ".cfg", ".ini", ".toml"}
_SKIP_DIRS = {".git", "node_modules", "dist", ".pytest_cache", "__pycache__", "cdk.out", ".venv", "venv"}

# Allowlist: placeholders/examples that are not real secrets.
_ALLOW = re.compile(r"(?i)(example|placeholder|dummy|FILL|<[^>]+>|xxx|changeme|your[-_])")


def _iter_files():
    for p in ROOT.rglob("*"):
        if p.is_dir():
            continue
        if any(part in _SKIP_DIRS for part in p.parts):
            continue
        if p.suffix.lower() not in _INCLUDE_SUFFIXES:
            continue
        yield p


def scan() -> list[str]:
    findings: list[str] = []
    for path in _iter_files():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if _ALLOW.search(line):
                continue
            for name, pat in _PATTERNS:
                if pat.search(line):
                    rel = path.relative_to(ROOT)
                    findings.append(f"{rel}:{lineno}: possible {name}")
    return findings


def main() -> int:
    findings = scan()
    if findings:
        print("[secret_scan] FAIL - possible secrets detected:", file=sys.stderr)
        for f in findings:
            print("  " + f, file=sys.stderr)
        return 1
    print("[secret_scan] OK - no hardcoded secrets detected in tracked source.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
