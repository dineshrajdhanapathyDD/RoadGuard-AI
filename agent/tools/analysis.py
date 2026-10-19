"""The ``analyze_frame`` and ``analyze_region`` agent tools (Task 6.4).

These are the two **perception** tools of the seven constrained Agent_Tools:
they are how the agent obtains *real* OpenCV output to reason over. Both are
thin, well-typed boundaries over the :class:`vision.pipeline.VisionPipeline`
façade -- they validate the decider's free-form arguments into a typed input,
delegate to the façade, and return the façade's structured
:class:`~vision.result_schema.AnalysisResult` unchanged.

Design mapping ("The Seven Agent Tools" table)
----------------------------------------------

| Tool             | Input                              | Output          | Failure behavior                                   |
| ``analyze_frame``  | ``frame_ref``, ``method``            | ``AnalysisResult`` | ``status=failed`` + reason, no fabricated detections (6.3) |
| ``analyze_region`` | ``frame_ref``, ``region_bbox``, ``method`` | region ``AnalysisResult`` | region out of bounds -> validation failure, no analysis (8.3) |

Requirements wired here
-----------------------

* **6.1 / 8.1 - input schema.** :func:`analyze_frame` accepts a frame reference
  and a selected detection method; :func:`analyze_region` additionally accepts a
  region boundary. The typed inputs (:class:`AnalyzeFrameInput`,
  :class:`AnalyzeRegionInput`) validate those fields: ``frame_ref`` must be a
  non-empty string, ``method`` must be one of ``{"classical", "model"}``, and a
  region must be a 4-tuple of integers.
* **6.2 / 8.2 - output schema.** Both return the façade's
  :class:`~vision.result_schema.AnalysisResult`, which carries the detected
  regions, per-detection confidence, the source frame timestamp, any
  ``Quality_Flag``, and the detection ``method`` used (the façade guarantees
  this shape).
* **6.3 - honest failure, no fabricated detections.** The façade already returns
  a :func:`~vision.result_schema.failed_result` (empty detections *by
  construction*) on any analysis failure. These tools add an **input-validation
  layer in front**: invalid arguments do not reach the façade and are not raised
  as exceptions into the loop -- they are returned as the same honest
  ``failed_result`` with a clear reason and empty detections. A failed analysis
  therefore can never become a finding (2.6, Competition 3).
* **8.3 - region bounds.** Out-of-bounds enforcement lives in the façade /
  ``region_analyzer`` (task 4.4). :func:`analyze_region` forwards a *well-formed*
  region to the façade and returns the façade's validation-failure result; the
  tool does not re-implement bounds checking, it only guarantees the argument is
  structurally a region before forwarding.

The tool contract (so the loop can dispatch into these)
-------------------------------------------------------

Each tool has the uniform signature ``tool(pipeline, arguments) -> AnalysisResult``:

* ``pipeline`` is the injected :class:`vision.pipeline.VisionPipeline` the
  Vision_Worker constructed for this inspection.
* ``arguments`` is the :attr:`agent.state.AgentDecision.arguments` mapping
  (plain ``dict``) the decider produced for this iteration.

The loop (task 6.8) looks up the tool by :class:`~agent.state.ToolName` and calls
it with ``(pipeline, decision.arguments)``. The tool owns parsing/validating
``arguments`` into its typed input, so the loop stays agnostic of each tool's
schema and only the seven named tools are ever invoked (5.5).

This module performs no OpenCV, network, or AWS work on import and is importable
with no AWS credentials (Requirement 19.4, Competition 4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from vision.pipeline import VisionPipelineProtocol
from vision.region_analyzer import Bbox
from vision.result_schema import METHODS, AnalysisResult, Method, failed_result


class ToolInputError(ValueError):
    """Raised internally when tool arguments fail validation.

    Callers of :func:`analyze_frame` / :func:`analyze_region` never see this: it
    is caught at the tool boundary and converted into an honest
    :func:`~vision.result_schema.failed_result` so the agent loop is driven by
    result status, not exceptions (6.3). It is exported for the few call sites
    (e.g. the loop or tests) that want to validate arguments up front.
    """


# --------------------------------------------------------------------------- #
# Shared argument parsing helpers
# --------------------------------------------------------------------------- #


def _require_frame_ref(arguments: Mapping[str, Any]) -> str:
    """Extract and validate a non-empty ``frame_ref`` string (6.1, 8.1)."""
    frame_ref = arguments.get("frame_ref")
    if not isinstance(frame_ref, str) or not frame_ref:
        raise ToolInputError(
            f"frame_ref must be a non-empty string, got {frame_ref!r}"
        )
    return frame_ref


def _require_method(arguments: Mapping[str, Any]) -> Method:
    """Extract and validate the detection ``method`` against the vocabulary (6.1, 8.1).

    Must be one of ``{"classical", "model"}`` (``vision.result_schema.METHODS``).
    """
    method = arguments.get("method")
    if method not in METHODS:
        raise ToolInputError(
            f"method must be one of {METHODS}, got {method!r}"
        )
    return method  # type: ignore[return-value]  # narrowed by the membership check


def _require_region(arguments: Mapping[str, Any]) -> Bbox:
    """Extract and validate a region as a 4-tuple of ints ``(x, y, w, h)`` (8.1).

    Accepts the ``region`` or ``region_bbox`` key (either name the decider may
    use) given as any length-4 sequence of integers. This checks only that the
    argument is *structurally* a region; whether the region falls inside the
    frame is enforced downstream by the façade / region_analyzer (8.3).
    """
    region = arguments.get("region_bbox", arguments.get("region"))
    if region is None:
        raise ToolInputError("a region is required (key 'region_bbox' or 'region')")
    try:
        values = list(region)
    except TypeError as exc:
        raise ToolInputError(f"region must be a sequence, got {region!r}") from exc
    if len(values) != 4:
        raise ToolInputError(
            f"region must be a 4-tuple (x, y, w, h), got {region!r}"
        )
    coerced: list[int] = []
    for name, value in zip(("x", "y", "w", "h"), values):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ToolInputError(
                f"region component {name} must be an int, got {value!r}"
            )
        coerced.append(value)
    return (coerced[0], coerced[1], coerced[2], coerced[3])


# --------------------------------------------------------------------------- #
# Typed input schemas
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AnalyzeFrameInput:
    """Validated input for :func:`analyze_frame` (6.1).

    Attributes:
        frame_ref: Reference to the sampled frame to analyze (non-empty).
        method: The selected detection method, ``"classical"`` or ``"model"``.
    """

    frame_ref: str
    method: Method

    @classmethod
    def from_arguments(cls, arguments: Mapping[str, Any]) -> "AnalyzeFrameInput":
        """Parse+validate a decider ``arguments`` mapping, or raise :class:`ToolInputError`."""
        if not isinstance(arguments, Mapping):
            raise ToolInputError(
                f"arguments must be a mapping, got {type(arguments).__name__}"
            )
        return cls(
            frame_ref=_require_frame_ref(arguments),
            method=_require_method(arguments),
        )


@dataclass(frozen=True)
class AnalyzeRegionInput:
    """Validated input for :func:`analyze_region` (8.1).

    Attributes:
        frame_ref: Reference to the sampled frame containing the region.
        region: The region boundary ``(x, y, w, h)`` in pixels.
        method: The selected detection method, ``"classical"`` or ``"model"``.
    """

    frame_ref: str
    region: Bbox
    method: Method

    @classmethod
    def from_arguments(cls, arguments: Mapping[str, Any]) -> "AnalyzeRegionInput":
        """Parse+validate a decider ``arguments`` mapping, or raise :class:`ToolInputError`."""
        if not isinstance(arguments, Mapping):
            raise ToolInputError(
                f"arguments must be a mapping, got {type(arguments).__name__}"
            )
        return cls(
            frame_ref=_require_frame_ref(arguments),
            region=_require_region(arguments),
            method=_require_method(arguments),
        )


# --------------------------------------------------------------------------- #
# The two perception tools
# --------------------------------------------------------------------------- #


def analyze_frame(
    pipeline: VisionPipelineProtocol, arguments: Mapping[str, Any]
) -> AnalysisResult:
    """``analyze_frame`` tool: analyze a whole frame (Requirement 6).

    Validates ``arguments`` into an :class:`AnalyzeFrameInput`
    (``frame_ref`` + ``method``, 6.1), then delegates to
    :meth:`vision.pipeline.VisionPipeline.analyze_frame`, returning its
    :class:`~vision.result_schema.AnalysisResult` unchanged. That result carries
    the detected regions, per-detection confidence, the source timestamp, any
    ``Quality_Flag``, and the ``method`` used (6.2); the façade guarantees a
    failed analysis has empty detections (6.3).

    Invalid arguments do not reach the façade and are not raised into the loop:
    they are returned as an honest :func:`~vision.result_schema.failed_result`
    with a clear reason and empty detections, so the agent reasons over result
    status rather than catching exceptions (6.3).

    Args:
        pipeline: The injected vision façade for this inspection.
        arguments: The decider's :attr:`~agent.state.AgentDecision.arguments`.

    Returns:
        The façade's :class:`~vision.result_schema.AnalysisResult`.
    """
    try:
        parsed = AnalyzeFrameInput.from_arguments(arguments)
    except ToolInputError as exc:
        return _invalid_input_result(arguments, f"analyze_frame input invalid: {exc}")
    return pipeline.analyze_frame(parsed.frame_ref, parsed.method)


def analyze_region(
    pipeline: VisionPipelineProtocol, arguments: Mapping[str, Any]
) -> AnalysisResult:
    """``analyze_region`` tool: analyze a region within a frame (Requirement 8).

    Validates ``arguments`` into an :class:`AnalyzeRegionInput`
    (``frame_ref`` + ``region_bbox`` + ``method``, 8.1), then delegates to
    :meth:`vision.pipeline.VisionPipeline.analyze_region`, returning its
    :class:`~vision.result_schema.AnalysisResult` unchanged. The result carries
    the region's detection output, per-detection confidence, and any
    ``Quality_Flag`` (8.2).

    Bounds enforcement is the façade's job: a region that falls outside the frame
    yields a validation-failure result with no analysis (8.3). This tool forwards
    a structurally valid region and returns that failure unchanged. Invalid
    arguments (missing/malformed region, bad method) are returned as an honest
    :func:`~vision.result_schema.failed_result`, never raised (6.3/8.3 spirit).

    Args:
        pipeline: The injected vision façade for this inspection.
        arguments: The decider's :attr:`~agent.state.AgentDecision.arguments`.

    Returns:
        The façade's region :class:`~vision.result_schema.AnalysisResult`.
    """
    try:
        parsed = AnalyzeRegionInput.from_arguments(arguments)
    except ToolInputError as exc:
        return _invalid_input_result(arguments, f"analyze_region input invalid: {exc}")
    return pipeline.analyze_region(parsed.frame_ref, parsed.region, parsed.method)


# --------------------------------------------------------------------------- #
# Honest failure for invalid tool input
# --------------------------------------------------------------------------- #


def _invalid_input_result(
    arguments: Mapping[str, Any], reason: str
) -> AnalysisResult:
    """Build an honest :func:`~vision.result_schema.failed_result` for bad input.

    The failed result carries empty detections by construction (2.6, 6.3) so an
    invalid-argument call can never become a finding. ``frame_ref`` and
    ``method`` are echoed back when they are present and well-formed enough to
    report; otherwise safe placeholders are used so the trace is still truthful
    about what was requested. ``source_timestamp_ms`` is ``0`` because no frame
    was resolved -- there is no real timestamp to report, and we never fabricate
    one.
    """
    raw_ref = arguments.get("frame_ref") if isinstance(arguments, Mapping) else None
    frame_ref = raw_ref if isinstance(raw_ref, str) and raw_ref else "unavailable"

    raw_method = arguments.get("method") if isinstance(arguments, Mapping) else None
    method: Method = raw_method if raw_method in METHODS else "classical"  # type: ignore[assignment]

    return failed_result(
        method=method,
        frame_ref=frame_ref,
        source_timestamp_ms=0,
        failure_reason=reason,
    )


# --------------------------------------------------------------------------- #
# Self-validation (import-time smoke check against a real VisionPipeline)
# --------------------------------------------------------------------------- #


def _self_check() -> None:
    """Wire the two tools over a real :class:`VisionPipeline` on a generated clip.

    Minimal real-OpenCV guard (no mocks), not the property-test suite
    (tasks 6.9-6.17). Runs under ``python -m agent.tools.analysis``:

    * ``analyze_frame`` on a resolved frame -> success with the method recorded;
    * an unknown ``frame_ref`` -> failed with empty detections (6.3);
    * ``analyze_region`` out of bounds -> failed with empty detections (8.3);
    * invalid arguments (bad method, missing region) -> honest failed result,
      never an exception (6.3).
    """
    import tempfile
    from pathlib import Path

    from vision.annotation import LocalEvidenceStore
    from vision.frame_sampler import SamplingPolicy
    from vision.pipeline import SelectionCriteria, VisionPipeline
    from vision.samples.make_samples import make_samples

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        clips = make_samples(tmp_path)
        pothole_clip = next(c for c in clips if "pothole" in c.name)

        pipeline = VisionPipeline(
            str(pothole_clip),
            inspection_id="insp-tools-selfcheck",
            policy=SamplingPolicy.every_n_frames(5),
            store=LocalEvidenceStore(tmp_path / "artifacts"),
        )

        refs = pipeline.select_frames(SelectionCriteria(), count=2)
        assert refs, "select_frames returned nothing from a real clip"
        first_ref = refs[0]["frame_ref"]

        # analyze_frame tool: success, method recorded (6.1, 6.2).
        ok = analyze_frame(pipeline, {"frame_ref": first_ref, "method": "classical"})
        assert ok["status"] == "success", ok["failure_reason"]
        assert ok["method"] == "classical", "selected method must be recorded (4.3)"
        assert ok["frame_ref"] == first_ref

        # analyze_frame tool: unknown frame_ref -> failed, empty detections (6.3).
        miss = analyze_frame(
            pipeline, {"frame_ref": "no-such#frame_99999999", "method": "classical"}
        )
        assert miss["status"] == "failed"
        assert miss["detections"] == [], "failed analysis must have empty detections"
        assert miss["failure_reason"]

        # analyze_frame tool: invalid input is an honest failure, not an exception.
        bad_method = analyze_frame(pipeline, {"frame_ref": first_ref, "method": "bogus"})
        assert bad_method["status"] == "failed"
        assert bad_method["detections"] == []
        no_ref = analyze_frame(pipeline, {"method": "classical"})
        assert no_ref["status"] == "failed"
        assert no_ref["detections"] == []

        # analyze_region tool: out-of-bounds -> validation failure, no analysis (8.3).
        oob = analyze_region(
            pipeline,
            {"frame_ref": first_ref, "region_bbox": (10_000, 10_000, 50, 50), "method": "classical"},
        )
        assert oob["status"] == "failed"
        assert oob["detections"] == [], "out-of-bounds region must not analyze (8.3)"

        # analyze_region tool: a valid in-bounds region succeeds (classical path).
        in_bounds = analyze_region(
            pipeline,
            {"frame_ref": first_ref, "region": (0, 0, 80, 80), "method": "classical"},
        )
        assert in_bounds["status"] == "success", in_bounds["failure_reason"]

        # analyze_region tool: malformed region -> honest failure, not an exception.
        bad_region = analyze_region(
            pipeline, {"frame_ref": first_ref, "region": (1, 2, 3), "method": "classical"}
        )
        assert bad_region["status"] == "failed"
        assert bad_region["detections"] == []


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    _self_check()
    print(
        "[agent.tools.analysis] OK: analyze_frame/analyze_region validate input "
        "and delegate to the VisionPipeline façade; success records the method; "
        "unknown frame_ref and out-of-bounds region fail with empty detections "
        "(6.1-6.3, 8.1-8.3); invalid input fails honestly rather than raising."
    )
