"""Tests for the OpenCV 5 verification smoke check (Requirement 2.1, Competition 4)."""

from __future__ import annotations

import pytest

from vision.verify_opencv import (
    REQUIRED_MAJOR,
    OpenCVInfo,
    OpenCVVersionError,
    _parse_major,
    check_opencv_version,
    inspect_opencv,
    main,
)


def test_required_major_is_five():
    assert REQUIRED_MAJOR == 5


def test_parse_major_valid():
    assert _parse_major("5.0.0") == 5
    assert _parse_major("4.11.0") == 4


def test_parse_major_unparseable_returns_negative_one():
    assert _parse_major("not-a-version") == -1


def test_inspect_reports_installed_opencv_is_major_5():
    info = inspect_opencv()
    assert info.import_ok, f"cv2 did not import: {info.detail}"
    # The actual installed OpenCV must be major version 5 (never fabricated).
    assert info.major == 5, f"installed OpenCV is {info.version}, expected major 5"
    assert info.meets_requirement is True
    assert info.version.startswith("5."), info.version


def test_check_opencv_version_passes_on_v5():
    info = check_opencv_version()
    assert isinstance(info, OpenCVInfo)
    assert info.meets_requirement


def test_check_raises_loudly_when_not_v5(monkeypatch):
    fake = OpenCVInfo(
        version="4.11.0",
        major=4,
        meets_requirement=False,
        import_ok=True,
        detail="OpenCV 4.11.0 (major=4) does NOT satisfy the requirement.",
    )
    monkeypatch.setattr("vision.verify_opencv.inspect_opencv", lambda: fake)
    with pytest.raises(OpenCVVersionError):
        check_opencv_version()


def test_main_returns_zero_on_v5():
    assert main([]) == 0


def test_main_returns_nonzero_when_not_v5(monkeypatch):
    fake = OpenCVInfo(
        version="unavailable",
        major=-1,
        meets_requirement=False,
        import_ok=False,
        detail="could not import cv2",
    )
    monkeypatch.setattr("vision.verify_opencv.inspect_opencv", lambda: fake)
    assert main([]) == 1
