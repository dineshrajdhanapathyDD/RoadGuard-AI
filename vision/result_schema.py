"""Structured result schema for the RoadGuard vision pipeline.

This module defines the shared output types every Vision_Pipeline component
produces and the agent reasons over, plus the factory helpers that enforce the
responsible-use invariants *by construction* so they cannot be violated by a
caller.

See .kiro/specs/roadguard-ai/design.md "Structured Result Schema
(result_schema.py)".

Responsible-use safeguards (Requirement 3) are enforced at the schema level:

* Measurements are **always** pixel-labeled (``unit="pixels"``) and never carry
  a physical conversion: ``physical`` is the literal ``"unavailable"`` because
  no calibration reference is supplied (3.1, 3.2).
* Any missing value is the explicit :data:`UNAVAILABLE` sentinel string rather
  than a fabricated number (3.3, 13.4).
* A ``failed`` :class:`AnalysisResult` carries an **empty** ``detections`` list
  by construction (via :func:`failed_result`), so a failed analysis can never
  be promoted into a successful finding (2.6, 6.2, 6.3).

The data shapes are ``TypedDict``s (matching the design), which keeps them
trivially JSON-serializable for the trace/finding/DynamoDB persistence layer.
The invariant that a ``TypedDict`` alone cannot express -- "failed implies no
detections" -- is guaranteed by the factory functions below, which are the
sanctioned way to build these records.

The module performs no OpenCV, network, or AWS work and is importable with no
AWS credentials (Requirement 19.4, Competition 4).
"""

from __future__ import annotations

from typing import Literal, TypedDict, get_args

# --------------------------------------------------------------------------- #
# Sentinels and closed vocabularies
# --------------------------------------------------------------------------- #

#: Explicit "no value" indicator. Never a fabricated number (3.3, 13.4).
UNAVAILABLE = "unavailable"

#: Hazard categories the pipeline may report (4.6). Potholes and surface damage
#: are produced on both paths; cracks and debris where the active path supports
#: them.
HazardType = Literal["pothole", "surface_damage", "crack", "debris"]

#: Detection method recorded in the trace for every analysis (4.3).
Method = Literal["classical", "model"]

#: Overall outcome of an analysis. ``failed`` can never become a finding (2.6).
Status = Literal["success", "failed"]

#: Quality-flag codes attached to uncertain or failed observations (2.5).
QualityCode = Literal["decode_failure", "low_confidence", "poor_input_quality"]

#: Comparison outcomes across two or more evidence items (9.2).
Agreement = Literal["agreement", "disagreement", "unavailable"]

# Tuple views of the Literal vocabularies, for runtime validation.
HAZARD_TYPES: tuple[str, ...] = get_args(HazardType)
METHODS: tuple[str, ...] = get_args(Method)
QUALITY_CODES: tuple[str, ...] = get_args(QualityCode)


# --------------------------------------------------------------------------- #
# Data shapes (match design.md exactly)
# --------------------------------------------------------------------------- #


class Measurement(TypedDict):
    """A pixel-space measurement of a detected region (3.1, 3.2).

    ``unit`` is always ``"pixels"`` and ``physical`` is always ``"unavailable"``
    -- the pipeline never converts pixel area into physical dimensions without a
    calibration reference. Build these with :func:`pixel_measurement`.
    """

    unit: Literal["pixels"]
    area_px: int
    width_px: int
    height_px: int
    physical: Literal["unavailable"]


class QualityFlag(TypedDict):
    """An uncertainty/failure indicator attached to an observation (2.5)."""

    code: QualityCode
    detail: str


class Detection(TypedDict):
    """One detected hazard region in pixel space (2.2)."""

    hazard_type: HazardType
    bbox_px: tuple[int, int, int, int]  # x, y, w, h in pixels
    confidence: float  # [0.0, 1.0]
    measurement: Measurement


class AnalysisResult(TypedDict):
    """The structured output of a frame or region analysis (2.2, 2.4, 6.2).

    A ``failed`` result (``status="failed"``) always carries an empty
    ``detections`` list -- enforce this with :func:`failed_result`.
    """

    status: Status
    method: Method
    frame_ref: str
    source_timestamp_ms: int  # preserved link to video position (2.4)
    detections: list[Detection]  # empty on failure, never fabricated (6.3)
    quality_flags: list[QualityFlag]
    failure_reason: str | None
    evidence_ref: str | None  # S3 key of annotated frame (2.3)


class FrameRef(TypedDict):
    """An ordered reference to a sampled frame with its source timestamp (7.2)."""

    frame_ref: str
    source_timestamp_ms: int


class ComparisonResult(TypedDict):
    """The outcome of comparing two or more evidence items (9.2, 9.3).

    On success ``status="success"`` and ``agreement`` is ``"agreement"`` or
    ``"disagreement"``. When fewer than two valid evidence items are referenced
    the result is a validation failure: ``status="failed"``, ``agreement`` is
    the :data:`UNAVAILABLE` literal, and ``failure_reason`` explains why (9.3).
    """

    status: Status
    agreement: Agreement
    evidence_refs: list[str]
    detail: str
    failure_reason: str | None


# --------------------------------------------------------------------------- #
# Factory helpers -- the sanctioned, invariant-preserving constructors
# --------------------------------------------------------------------------- #


def pixel_measurement(width_px: int, height_px: int, *, area_px: int | None = None) -> Measurement:
    """Build a pixel-labeled :class:`Measurement` (3.1, 3.2).

    ``unit`` is always ``"pixels"`` and ``physical`` is always
    ``"unavailable"`` -- there is no code path that converts to physical units.
    ``area_px`` defaults to the bounding-box area ``width_px * height_px`` when
    not supplied explicitly.
    """
    if width_px < 0 or height_px < 0:
        raise ValueError(f"pixel dimensions must be non-negative, got {width_px}x{height_px}")
    resolved_area = width_px * height_px if area_px is None else area_px
    if resolved_area < 0:
        raise ValueError(f"area_px must be non-negative, got {resolved_area}")
    return Measurement(
        unit="pixels",
        area_px=resolved_area,
        width_px=width_px,
        height_px=height_px,
        physical="unavailable",
    )


def quality_flag(code: QualityCode, detail: str) -> QualityFlag:
    """Build a :class:`QualityFlag`, validating the code against the vocabulary."""
    if code not in QUALITY_CODES:
        raise ValueError(f"unknown quality-flag code {code!r}; expected one of {QUALITY_CODES}")
    return QualityFlag(code=code, detail=detail)


def detection(
    hazard_type: HazardType,
    bbox_px: tuple[int, int, int, int],
    confidence: float,
    measurement: Measurement,
) -> Detection:
    """Build a validated :class:`Detection`.

    Enforces the real-value guarantees: ``hazard_type`` must be in the closed
    vocabulary, ``bbox_px`` must be a 4-tuple, and ``confidence`` must be a real
    number in ``[0.0, 1.0]`` -- never fabricated or out of range (3.3, 6.2).
    """
    if hazard_type not in HAZARD_TYPES:
        raise ValueError(f"unknown hazard_type {hazard_type!r}; expected one of {HAZARD_TYPES}")
    if len(bbox_px) != 4:
        raise ValueError(f"bbox_px must be a 4-tuple (x, y, w, h), got {bbox_px!r}")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence must be in [0.0, 1.0], got {confidence!r}")
    return Detection(
        hazard_type=hazard_type,
        bbox_px=(int(bbox_px[0]), int(bbox_px[1]), int(bbox_px[2]), int(bbox_px[3])),
        confidence=float(confidence),
        measurement=measurement,
    )


def success_result(
    *,
    method: Method,
    frame_ref: str,
    source_timestamp_ms: int,
    detections: list[Detection] | None = None,
    quality_flags: list[QualityFlag] | None = None,
    evidence_ref: str | None = None,
) -> AnalysisResult:
    """Build a successful :class:`AnalysisResult` (2.2, 2.4).

    A successful result has ``status="success"`` and ``failure_reason=None``.
    Detections and quality flags default to empty lists; a success with zero
    detections is legitimate (the frame simply had no hazard).
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    return AnalysisResult(
        status="success",
        method=method,
        frame_ref=frame_ref,
        source_timestamp_ms=int(source_timestamp_ms),
        detections=list(detections) if detections else [],
        quality_flags=list(quality_flags) if quality_flags else [],
        failure_reason=None,
        evidence_ref=evidence_ref,
    )


def failed_result(
    *,
    method: Method,
    frame_ref: str,
    source_timestamp_ms: int,
    failure_reason: str,
    quality_flags: list[QualityFlag] | None = None,
    evidence_ref: str | None = None,
) -> AnalysisResult:
    """Build a failed :class:`AnalysisResult` with an empty detections list (2.6, 6.3).

    This is the only sanctioned way to produce a ``status="failed"`` result.
    ``detections`` is empty **by construction**: there is no parameter to pass
    detections, so a failed result can never carry (fabricated) detections and
    therefore can never be promoted into a successful finding (Competition 3).
    A non-empty ``failure_reason`` is required so the failure is surfaced
    honestly rather than hidden.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    if not failure_reason:
        raise ValueError("failed_result requires a non-empty failure_reason")
    return AnalysisResult(
        status="failed",
        method=method,
        frame_ref=frame_ref,
        source_timestamp_ms=int(source_timestamp_ms),
        detections=[],  # invariant: failed => no detections (2.6, 6.3)
        quality_flags=list(quality_flags) if quality_flags else [],
        failure_reason=failure_reason,
        evidence_ref=evidence_ref,
    )


def frame_ref(reference: str, source_timestamp_ms: int) -> FrameRef:
    """Build a :class:`FrameRef` pairing a reference with its source timestamp (7.2)."""
    return FrameRef(frame_ref=reference, source_timestamp_ms=int(source_timestamp_ms))


def comparison_result(
    agreement: Literal["agreement", "disagreement"],
    evidence_refs: list[str],
    detail: str,
) -> ComparisonResult:
    """Build a successful :class:`ComparisonResult` for two or more evidence items (9.2).

    Raises ``ValueError`` if fewer than two evidence references are supplied; a
    comparison is only meaningful across at least two items. For the "<2 valid
    items" case use :func:`comparison_failure` to produce a validation failure
    (9.3).
    """
    if len(evidence_refs) < 2:
        raise ValueError(
            "comparison_result requires at least two evidence refs; "
            "use comparison_failure for the fewer-than-two case (9.3)"
        )
    if agreement not in ("agreement", "disagreement"):
        raise ValueError(f"agreement must be 'agreement' or 'disagreement', got {agreement!r}")
    return ComparisonResult(
        status="success",
        agreement=agreement,
        evidence_refs=list(evidence_refs),
        detail=detail,
        failure_reason=None,
    )


def comparison_failure(evidence_refs: list[str], failure_reason: str) -> ComparisonResult:
    """Build a validation-failure :class:`ComparisonResult` (9.3).

    Used when fewer than two valid evidence items are referenced. ``agreement``
    is the explicit :data:`UNAVAILABLE` literal rather than a fabricated verdict.
    """
    if not failure_reason:
        raise ValueError("comparison_failure requires a non-empty failure_reason")
    return ComparisonResult(
        status="failed",
        agreement="unavailable",
        evidence_refs=list(evidence_refs),
        detail=failure_reason,
        failure_reason=failure_reason,
    )


# --------------------------------------------------------------------------- #
# Self-validation (import-time smoke check of the responsible-use invariants)
# --------------------------------------------------------------------------- #


def _self_check() -> None:
    """Assert the responsible-use invariants hold. Runs under ``python -m``.

    This is a lightweight guard, not the property-test suite (tasks 2.4/2.5).
    """
    m = pixel_measurement(10, 20)
    assert m["unit"] == "pixels"
    assert m["physical"] == UNAVAILABLE
    assert m["area_px"] == 200

    ok = success_result(
        method="classical",
        frame_ref="frame_0",
        source_timestamp_ms=0,
        detections=[detection("pothole", (1, 2, 3, 4), 0.9, m)],
    )
    assert ok["status"] == "success"
    assert ok["detections"][0]["confidence"] == 0.9

    bad = failed_result(
        method="model",
        frame_ref="frame_1",
        source_timestamp_ms=100,
        failure_reason="decode_failure",
    )
    # The central invariant: failed => empty detections, by construction (2.6, 6.3).
    assert bad["status"] == "failed"
    assert bad["detections"] == []
    assert bad["failure_reason"]

    cmp_ok = comparison_result("agreement", ["e1", "e2"], "both show a pothole")
    assert cmp_ok["status"] == "success"
    cmp_bad = comparison_failure(["e1"], "fewer than two valid evidence items")
    assert cmp_bad["status"] == "failed"
    assert cmp_bad["agreement"] == UNAVAILABLE


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    _self_check()
    print("[result_schema] OK: responsible-use invariants hold.")
