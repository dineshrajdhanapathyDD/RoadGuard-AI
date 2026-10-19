"""OpenCV 5 verification smoke check.

Requirement 2.1 / Competition 4: the platform must run a *real* OpenCV 5
pipeline, and the reproducible deliverables must run without AWS credentials.

This module imports ``cv2``, reads the actually-installed version, and asserts
that the major version is 5. It fails loudly (non-zero exit, raised error) when
the requirement is not met, and it reports the real installed version rather
than silently passing. It performs no network or AWS calls, so it is a valid
credential-free smoke check.

Usage::

    python -m vision.verify_opencv      # exits 0 if OpenCV major == 5, else 1
    verify-opencv                       # console-script entry point (pyproject)

Programmatic use::

    from vision.verify_opencv import check_opencv_version, OpenCVVersionError
    info = check_opencv_version()       # raises OpenCVVersionError if not v5
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

REQUIRED_MAJOR = 5


class OpenCVVersionError(RuntimeError):
    """Raised when the installed OpenCV does not meet the major-version-5 requirement."""


@dataclass(frozen=True)
class OpenCVInfo:
    """The actual, observed OpenCV installation state."""

    version: str
    major: int
    meets_requirement: bool
    import_ok: bool
    detail: str


def _parse_major(version: str) -> int:
    """Return the leading integer of a version string, or -1 if unparseable."""
    head = version.strip().split(".", 1)[0]
    try:
        return int(head)
    except ValueError:
        return -1


def inspect_opencv() -> OpenCVInfo:
    """Import cv2 and report the real installed state without raising.

    Never fabricates a version: on import failure the reported version is the
    explicit string "unavailable" and ``import_ok`` is False.
    """
    try:
        import cv2  # noqa: PLC0415 - imported lazily so inspection reports honestly
    except Exception as exc:  # pragma: no cover - environment-dependent
        return OpenCVInfo(
            version="unavailable",
            major=-1,
            meets_requirement=False,
            import_ok=False,
            detail=f"could not import cv2: {exc!r}",
        )

    version = str(getattr(cv2, "__version__", "unavailable"))
    major = _parse_major(version)
    meets = major == REQUIRED_MAJOR
    if meets:
        detail = f"OpenCV {version} satisfies the major-version-{REQUIRED_MAJOR} requirement."
    else:
        detail = (
            f"OpenCV {version} (major={major}) does NOT satisfy the "
            f"major-version-{REQUIRED_MAJOR} requirement. Install opencv-python==5.0.0.93."
        )
    return OpenCVInfo(
        version=version,
        major=major,
        meets_requirement=meets,
        import_ok=True,
        detail=detail,
    )


def check_opencv_version() -> OpenCVInfo:
    """Return the installation info, raising OpenCVVersionError if not OpenCV 5.

    This is the "fail loudly" entry point for programmatic callers and tests.
    """
    info = inspect_opencv()
    if not info.meets_requirement:
        raise OpenCVVersionError(info.detail)
    return info


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Prints the real version and returns 0 only if OpenCV 5."""
    _ = argv  # no flags today; kept for a stable console-script signature
    info = inspect_opencv()
    stream = sys.stdout if info.meets_requirement else sys.stderr
    print(f"[verify_opencv] installed cv2 version : {info.version}", file=stream)
    print(f"[verify_opencv] installed cv2 major   : {info.major}", file=stream)
    print(f"[verify_opencv] requires major        : {REQUIRED_MAJOR}", file=stream)
    print(f"[verify_opencv] meets requirement     : {info.meets_requirement}", file=stream)
    print(f"[verify_opencv] {info.detail}", file=stream)

    if not info.meets_requirement:
        print("[verify_opencv] FAIL: OpenCV 5 requirement not met.", file=sys.stderr)
        return 1
    print("[verify_opencv] OK: OpenCV 5 verified.", file=stream)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
