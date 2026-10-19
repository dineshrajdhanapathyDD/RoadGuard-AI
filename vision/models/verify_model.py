"""Verify the pinned ONNX model manifest and artifact (task 14.1 CI check).

Checks, with NO network access, that:
  1. ``MODEL_MANIFEST.md`` exists, is well-formed, and names a License and a
     Source URL (so the pin is attributable and license-verified);
  2. IF the local ONNX artifact exists, its SHA-256 matches the manifest's
     recorded ``Expected SHA-256``.

The model is OPTIONAL (the Classical_Path is the default, and the Mock_Runtime
backbone does not need it), so this verifier PASSES when the artifact is absent
and only fails on a present-but-mismatched artifact or a malformed/licenseless
manifest. Run via ``python -m vision.models.verify_model``.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parent
_MANIFEST = _DIR / "MODEL_MANIFEST.md"
_ARTIFACT = _DIR / "road_damage_yolov11s.onnx"


class ModelManifestError(RuntimeError):
    """Raised when the manifest is missing/malformed or the artifact mismatches."""


def _parse_manifest(text: str) -> dict[str, str]:
    """Pull the key fields out of the manifest's pin table (honest, minimal parse)."""
    fields: dict[str, str] = {}
    for key in ("License", "Source URL", "Expected SHA-256", "Local artifact"):
        m = re.search(rf"\|\s*{re.escape(key)}\s*\|\s*(.+?)\s*\|", text)
        if m:
            fields[key] = m.group(1).strip()
    return fields


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify() -> str:
    if not _MANIFEST.exists():
        raise ModelManifestError(f"model manifest missing: {_MANIFEST}")
    text = _MANIFEST.read_text(encoding="utf-8")
    fields = _parse_manifest(text)

    license_ = fields.get("License", "")
    if not license_ or "FILL" in license_:
        raise ModelManifestError("manifest does not name a verified License")
    if "CC-BY" not in license_ and "MIT" not in license_ and "Apache" not in license_:
        # Not a hard fail on vocabulary, but require a real, recognizable license token.
        raise ModelManifestError(f"manifest License is not a recognized permissive license: {license_!r}")
    if not fields.get("Source URL", "").startswith("http"):
        raise ModelManifestError("manifest does not record a Source URL")

    if not _ARTIFACT.exists():
        return (
            f"OK (manifest valid; license={license_}); ONNX artifact absent - "
            "Model_Path optional, Classical_Path is the default."
        )

    expected = fields.get("Expected SHA-256", "")
    if "FILL" in expected or not re.fullmatch(r"[0-9a-fA-F]{64}", expected or ""):
        raise ModelManifestError(
            "ONNX artifact is present but the manifest has no concrete Expected SHA-256; "
            "record the fetched artifact's sha256 in MODEL_MANIFEST.md"
        )
    actual = _sha256(_ARTIFACT)
    if actual.lower() != expected.lower():
        raise ModelManifestError(
            f"ONNX artifact SHA-256 mismatch: manifest={expected}, actual={actual}"
        )
    return f"OK (manifest valid; license={license_}; artifact sha256 matches)"


def main() -> int:
    try:
        print("[verify_model] " + verify())
        return 0
    except ModelManifestError as exc:
        print(f"[verify_model] FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
