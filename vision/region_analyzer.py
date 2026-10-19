"""Region analyzer: region-scoped detection with bounds enforcement.

Vision Pipeline Design, ``region_analyzer.py`` row of the component-responsibilities
table:

    | Input                 | Output                    | Failure behavior                              |
    | frame + region bbox, method | region-scoped ``Detection[]`` | Region outside bounds -> ``ValidationFailure`` (8.3) |

This component backs the agent's ``analyze_region`` tool (Requirement 8). Given a
sampled :class:`~vision.frame_sampler.Frame` and a :data:`Bbox` region boundary
(``x, y, w, h`` in pixels), it:

* **Validates the region against the real frame bounds first** (8.3). A region
  with a non-positive size, a negative origin, or an extent that spills past the
  frame's width/height is rejected with a :func:`~vision.result_schema.failed_result`
  and a clear ``failure_reason`` -- **no detection is attempted** on an
  out-of-bounds region. The failed result carries an empty ``detections`` list
  by construction, so an out-of-bounds region can never become a finding
  (2.6, 6.3, Competition 3).
* **Crops the region from the frame and delegates detection to a detector
  seam** (8.1, 8.2). The crop is the real ROI pixels; the analyzer owns *no*
  detection logic of its own. It calls a :class:`RegionDetector` -- the same
  interface :mod:`vision.damage_detector` (task 4.2) will implement -- so the
  classical and model paths compose here without duplication. Detections
  returned in crop-local coordinates are translated back into frame coordinates
  so every ``bbox_px`` is anchored to the source frame (8.2).

The detector seam
-----------------

Detection is injected, not hard-coded, through the :class:`RegionDetector`
``Protocol``::

    class RegionDetector(Protocol):
        def detect(self, image: np.ndarray, method: Method) -> list[Detection]: ...

Resolution order for the detector used by :func:`analyze_region`:

1. An explicit ``detector`` argument (dependency injection; this is how the
   vision façade and tests wire the real classical/model detector).
2. Otherwise a lazy import of :func:`vision.damage_detector.detect_regions`
   (the function task 4.2 lands). The import is lazy and guarded so this module
   stays importable and testable *before* ``damage_detector`` exists.
3. If neither is available, a guarded :data:`_UnavailableDetector` stub that
   analyzes nothing and attaches a ``poor_input_quality`` :class:`QualityFlag`
   explaining that no detector is wired yet. It never fabricates a detection
   (3.3), so the seam degrades honestly rather than crashing the pipeline.

When ``damage_detector`` lands (4.2/14.2), no change is needed here: the lazy
import resolves to the real detector, or the façade injects it explicitly.

This module performs no network or AWS work and is importable with no AWS
credentials (Requirement 19.4, Competition 4).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from vision.frame_sampler import Frame
from vision.result_schema import (
    Detection,
    Method,
    QualityFlag,
    AnalysisResult,
    detection,
    failed_result,
    pixel_measurement,
    quality_flag,
    success_result,
)

#: A region boundary in pixel space: ``(x, y, w, h)``. Same shape as a
#: ``Detection``'s ``bbox_px``; ``x, y`` is the top-left origin and ``w, h`` the
#: width/height, all in pixels (design "region bbox").
Bbox = tuple[int, int, int, int]


@runtime_checkable
class RegionDetector(Protocol):
    """The detection seam the region analyzer delegates to.

    This is the interface :mod:`vision.damage_detector` (task 4.2) implements:
    given a real image (here, the cropped ROI) and a selected ``method``,
    return pixel-space :class:`~vision.result_schema.Detection` records in the
    coordinate frame of the *image passed in* (i.e. crop-local). The analyzer
    translates those into source-frame coordinates.

    Keeping detection behind this ``Protocol`` means the analyzer composes with
    the classical path now and the model path later without duplicating or
    importing detection logic eagerly.
    """

    def detect(self, image: np.ndarray, method: Method) -> list[Detection]:
        """Return crop-local detections for ``image`` using ``method``."""
        ...


class _UnavailableDetector:
    """Guarded stub used when no real detector is wired yet (task 4.2 pending).

    Analyzes nothing and signals the gap honestly via a ``poor_input_quality``
    quality flag on the result. It never fabricates a detection (3.3), so the
    region seam degrades gracefully and the pipeline/tests still run before
    :mod:`vision.damage_detector` exists.
    """

    reason = "no damage detector wired yet (vision.damage_detector pending task 4.2)"

    def detect(self, image: np.ndarray, method: Method) -> list[Detection]:  # noqa: ARG002
        return []


def _resolve_detector(detector: RegionDetector | None) -> tuple[RegionDetector, bool]:
    """Resolve the detector to use, returning ``(detector, is_real)``.

    Prefers an injected ``detector``; else lazily imports the real
    ``vision.damage_detector`` detector; else falls back to the guarded stub.
    ``is_real`` is ``False`` only for the stub, so the caller can attach an
    honest quality flag.
    """
    if detector is not None:
        return detector, True

    # Lazy, guarded import so this module is importable before task 4.2 lands.
    try:  # pragma: no cover - exercised once damage_detector exists
        from vision import damage_detector as _dd  # type: ignore[attr-defined]

        detect_fn = getattr(_dd, "detect_regions", None)
        if callable(detect_fn):

            class _FnDetector:
                def detect(self, image: np.ndarray, method: Method) -> list[Detection]:
                    return list(detect_fn(image, method))

            return _FnDetector(), True
        # A class-based detector is also acceptable if the module exposes one.
        detector_cls = getattr(_dd, "DamageDetector", None)
        if detector_cls is not None:
            instance = detector_cls()
            if isinstance(instance, RegionDetector):
                return instance, True
    except Exception:
        # Not importable yet (or import-time failure): degrade to the stub.
        pass

    return _UnavailableDetector(), False


def _validate_region(region: Bbox, width: int, height: int) -> str | None:
    """Return a failure reason if ``region`` is not fully within the frame.

    Enforces Requirement 8.3: ``x < 0``, ``y < 0``, ``w <= 0``, ``h <= 0``, or
    an extent spilling past the frame (``x + w > width`` or ``y + h > height``)
    is out of bounds. Returns ``None`` when the region is valid.
    """
    if len(region) != 4:
        return f"region must be a 4-tuple (x, y, w, h), got {region!r}"
    x, y, w, h = region
    if w <= 0 or h <= 0:
        return f"region has non-positive size (w={w}, h={h}); nothing to analyze"
    if x < 0 or y < 0:
        return f"region origin is negative (x={x}, y={y}); outside frame bounds"
    if x + w > width or y + h > height:
        return (
            f"region ({x},{y},{w},{h}) spills past frame bounds "
            f"{width}x{height} (x+w={x + w}, y+h={y + h})"
        )
    return None


def _frame_dimensions(frame: Frame) -> tuple[int, int]:
    """Return ``(width, height)`` of a decoded frame's real pixels.

    ``Frame.image`` is a BGR ndarray shaped ``(height, width, channels)``.
    """
    assert frame.image is not None  # guarded by the caller (frame.ok)
    height, width = frame.image.shape[0], frame.image.shape[1]
    return int(width), int(height)


def _translate_to_frame(det: Detection, offset_x: int, offset_y: int) -> Detection:
    """Shift a crop-local detection's bbox back into source-frame coordinates.

    The detector sees only the cropped ROI, so its ``bbox_px`` origin is
    relative to the crop. Adding the region origin anchors the detection to the
    source frame (8.2). Rebuilds through the :func:`~vision.result_schema.detection`
    factory so validation and the pixel-labeled measurement are preserved.
    """
    x, y, w, h = det["bbox_px"]
    frame_bbox: Bbox = (x + offset_x, y + offset_y, w, h)
    return detection(
        det["hazard_type"],
        frame_bbox,
        det["confidence"],
        pixel_measurement(w, h, area_px=det["measurement"]["area_px"]),
    )


def analyze_region(
    frame: Frame,
    region: Bbox,
    method: Method,
    *,
    detector: RegionDetector | None = None,
) -> AnalysisResult:
    """Run region-scoped detection on ``frame`` within ``region`` (Requirement 8).

    Validates ``region`` against the frame's real bounds first; if the region
    falls outside the frame (negative origin, non-positive size, or an extent
    spilling past width/height) this returns a
    :func:`~vision.result_schema.failed_result` validation failure and analyzes
    nothing (8.3). Otherwise it crops the ROI, delegates detection to the
    resolved :class:`RegionDetector` seam, translates the detections back into
    source-frame coordinates, and returns a
    :func:`~vision.result_schema.success_result` with the region's detections
    and any quality flags (8.1, 8.2).

    Args:
        frame: The sampled :class:`~vision.frame_sampler.Frame` to analyze. A
            frame that failed to decode (``not frame.ok``) is itself a failure
            and is surfaced as such -- there are no real pixels to analyze.
        region: The :data:`Bbox` region boundary ``(x, y, w, h)`` in pixels.
        method: The selected detection method recorded on the result (4.3).
        detector: Optional explicit :class:`RegionDetector` (dependency
            injection). When omitted, the real detector is resolved lazily with
            a guarded stub fallback (see module docstring).

    Returns:
        A region-scoped :class:`~vision.result_schema.AnalysisResult`.
    """
    # A frame that could not be decoded has no real pixels to analyze. Carry its
    # decode quality flags onto an honest failed result (no fabricated pixels).
    if not frame.ok:
        return failed_result(
            method=method,
            frame_ref=frame.frame_ref,
            source_timestamp_ms=_ts(frame),
            failure_reason="frame did not decode; no pixels to analyze in region",
            quality_flags=list(frame.quality_flags),
        )

    width, height = _frame_dimensions(frame)

    # 8.3: validate bounds BEFORE any analysis. Out of bounds -> validation
    # failure with no detection output (failed_result carries empty detections).
    reason = _validate_region(region, width, height)
    if reason is not None:
        return failed_result(
            method=method,
            frame_ref=frame.frame_ref,
            source_timestamp_ms=_ts(frame),
            failure_reason=reason,
        )

    x, y, w, h = region
    roi = frame.image[y : y + h, x : x + w]  # real ROI pixels

    resolved, is_real = _resolve_detector(detector)
    local_detections = resolved.detect(roi, method)

    # Anchor crop-local detections back to source-frame coordinates (8.2).
    detections = [_translate_to_frame(d, x, y) for d in local_detections]

    quality_flags: list[QualityFlag] = []
    if not is_real:
        quality_flags.append(
            quality_flag("poor_input_quality", _UnavailableDetector.reason)
        )

    return success_result(
        method=method,
        frame_ref=frame.frame_ref,
        source_timestamp_ms=_ts(frame),
        detections=detections,
        quality_flags=quality_flags,
    )


def _ts(frame: Frame) -> int:
    """Coerce a frame's ``source_timestamp_ms`` to an int for the result schema.

    A sampled frame's timestamp is a real millisecond int, or the UNAVAILABLE
    sentinel when the backend reported no position. The result schema stores an
    int; an unavailable timestamp is recorded as ``0`` (first-frame-equivalent)
    without fabricating a non-zero position.
    """
    ts = frame.source_timestamp_ms
    return int(ts) if isinstance(ts, int) else 0


# --------------------------------------------------------------------------- #
# Self-validation (import-time smoke check of bounds + delegation seam)
# --------------------------------------------------------------------------- #


def _self_check() -> None:
    """Assert bounds enforcement (8.3) and the detection seam (8.1, 8.2).

    Lightweight guard with real ndarrays and no mocks, not the property/unit
    suite (optional task 4.8). Runs under ``python -m``.
    """
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    frame = Frame(
        frame_ref="clip#frame_00000000",
        frame_index=0,
        source_timestamp_ms=0,
        image=image,
        quality_flags=[],
    )

    # In-bounds region on a flat (featureless) frame. With the real classical
    # detector now wired (vision.damage_detector landed in task 4.2), the seam
    # resolves to it: a flat ROI has no edges/texture, so the result is a
    # success with zero detections and NO stub quality flag (the detector is
    # real, not the _UnavailableDetector stub).
    ok = analyze_region(frame, (10, 20, 50, 60), "classical")
    assert ok["status"] == "success"
    assert ok["detections"] == []
    assert not any(f["code"] == "poor_input_quality" for f in ok["quality_flags"])
    assert ok["method"] == "classical"

    # Out-of-bounds cases (8.3): each must be a validation failure, no analysis.
    for bad_region in [
        (-1, 0, 10, 10),  # negative x
        (0, -1, 10, 10),  # negative y
        (300, 0, 50, 10),  # x + w (350) > width (320)
        (0, 200, 10, 60),  # y + h (260) > height (240)
        (0, 0, 0, 10),  # non-positive width
        (0, 0, 10, 0),  # non-positive height
    ]:
        res = analyze_region(frame, bad_region, "classical")
        assert res["status"] == "failed", f"expected failure for {bad_region}"
        assert res["detections"] == [], f"out-of-bounds must analyze nothing: {bad_region}"
        assert res["failure_reason"], f"missing failure_reason for {bad_region}"

    # Injected detector seam: crop-local detection is translated to frame coords.
    class _StubDetector:
        def detect(self, img: np.ndarray, method: Method) -> list[Detection]:
            h, w = img.shape[0], img.shape[1]
            # one detection filling the crop, in crop-local coordinates
            return [detection("pothole", (0, 0, w, h), 0.77, pixel_measurement(w, h))]

    injected = analyze_region(frame, (10, 20, 50, 60), "model", detector=_StubDetector())
    assert injected["status"] == "success"
    assert len(injected["detections"]) == 1
    det = injected["detections"][0]
    # crop-local (0,0,50,60) translated by region origin (10,20) -> (10,20,50,60)
    assert det["bbox_px"] == (10, 20, 50, 60), det["bbox_px"]
    assert det["confidence"] == 0.77
    assert injected["quality_flags"] == []  # real detector -> no stub flag

    # A frame that failed to decode is surfaced as a failure, not analyzed.
    bad_frame = Frame(
        frame_ref="clip#frame_00000001",
        frame_index=1,
        source_timestamp_ms=100,
        image=None,
        quality_flags=[quality_flag("decode_failure", "frame 1 undecodable")],
    )
    dec = analyze_region(bad_frame, (0, 0, 10, 10), "classical")
    assert dec["status"] == "failed"
    assert dec["detections"] == []
    assert any(f["code"] == "decode_failure" for f in dec["quality_flags"])


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    _self_check()
    print("[region_analyzer] OK: region bounds enforced (8.3) and detection seam composes.")
