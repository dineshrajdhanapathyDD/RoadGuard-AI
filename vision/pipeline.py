"""Vision façade: the single ``VisionPipeline`` the agent tools call (task 4.5).

Vision Pipeline Design, "Vision Pipeline Interface":

    The Vision_Worker invokes the pipeline through a single façade that the agent
    tools call:

    ```python
    class VisionPipeline(Protocol):
        def analyze_frame(self, frame_ref: str, method: Method) -> AnalysisResult: ...
        def analyze_region(self, frame_ref: str, region: Bbox, method: Method) -> AnalysisResult: ...
        def select_frames(self, criteria: SelectionCriteria, count: int) -> list[FrameRef]: ...
        def compare_evidence(self, evidence_refs: list[str]) -> ComparisonResult: ...
    ```

    All four return the structured types from ``result_schema.py``; none ever
    returns a fabricated detection (3.3, 6.3).

This module composes the already-built, independently tested stages into that
one façade. It does **not** reimplement any stage; it wires them:

    load_video (video_loader) -> sample_frames (frame_sampler)
      -> preprocess_frame (image_preprocessor) -> detect (damage_detector, the
      Classical_Path for now) -> annotate_and_store (annotation) -> AnalysisResult

Requirements wired here
-----------------------

* **2.6 / 6.3 / Competition 3 - failed analyses can never become findings.**
  Every failure path returns :func:`~vision.result_schema.failed_result`, which
  carries an **empty** ``detections`` list *by construction* (there is no
  parameter to pass detections to it). A frame that does not resolve, a frame
  that failed to decode, a low-quality preprocessing result, or a detector error
  all funnel through ``failed_result``. The façade therefore cannot promote a
  failed analysis into detections -- the invariant is structural, not a check we
  could forget. See :meth:`VisionPipeline.analyze_frame`.
* **2.2 / 2.4 - genuine detection output with the source timestamp.** Successful
  results carry the real detections, the recorded ``method``, and the frame's
  real ``source_timestamp_ms`` (preserved end to end from the sampler).
* **2.3 - annotated evidence.** On a successful analysis the façade annotates the
  frame with the real detections and stores it through the injected
  :class:`~vision.annotation.EvidenceStore`, setting ``evidence_ref`` on the
  result. Evidence is generated uniformly even for a zero-detection success so
  the trace/report always has a frame to show; a *failed* analysis stores no
  evidence (there is nothing honest to annotate).
* **7.1 / 7.2 / 7.3 - ordered, timestamped frame selection.**
  :meth:`VisionPipeline.select_frames` returns an ordered list of
  :class:`~vision.result_schema.FrameRef` (deterministic, by source timestamp),
  or an empty list when nothing matches the criteria.
* **9.1 / 9.2 / 9.3 - evidence comparison requires >= 2 items.**
  :meth:`VisionPipeline.compare_evidence` returns a
  :func:`~vision.result_schema.comparison_failure` when fewer than two valid
  evidence items are referenced, and an honest
  :func:`~vision.result_schema.comparison_result` (agreement / disagreement
  derived from re-analysis, never fabricated) otherwise.
* **4.3 - method recorded; model path not silently relabeled.** The selected
  ``method`` is always recorded on the result. The Classical_Path is wired now.
  Because the Model_Path (ONNX/DNN) is task 14.2 and the classical detector
  raises ``NotImplementedError`` / ``ModelLoadError`` for ``method="model"``,
  the façade records a ``failed_result`` **with ``method="model"``** and a clear
  reason rather than relabeling classical output as model output. The agent's
  method-selection-and-fallback logic (task 14.3) owns re-running on the
  classical path; the façade's job is to never misattribute the method.

frame_ref resolution (how a ``frame_ref`` maps back to pixels)
--------------------------------------------------------------

The agent hands the façade an opaque ``frame_ref`` string (the sampler's
``"<source>#frame_<index>"``). To analyze it the façade must map it back to the
real pixels. The approach, kept fully offline:

1. The pipeline is constructed with a **video source** (local path), a
   **sampling policy**, and an **evidence store**.
2. On first use it **samples the video once** and builds an index
   ``frame_ref -> _IndexedFrame`` holding the sampled
   :class:`~vision.frame_sampler.Frame` and its
   :class:`~vision.image_preprocessor.PreprocessedFrame`. The sample pass is
   lazy and memoized (:meth:`VisionPipeline._index`), so repeated tool calls
   reuse the same frames and every ``frame_ref`` resolves deterministically to
   the same pixels.
3. ``analyze_frame`` / ``analyze_region`` look the ``frame_ref`` up in that
   index. An unknown ``frame_ref`` -> a ``failed_result`` (there are no pixels to
   analyze; we never fabricate them). ``select_frames`` enumerates the index in
   timestamp order. ``compare_evidence`` maps an ``evidence_ref`` key back to its
   ``frame_ref`` (the key embeds the sanitized frame_ref) and re-analyzes.

This indexing strategy keeps the façade a pure function of the real decoded
video: identical video + policy -> identical index -> identical, reproducible
results, which is exactly what the Mock_Runtime needs (19.2-19.5, Competition 4).

Offline, no AWS
---------------

The default store is :class:`~vision.annotation.LocalEvidenceStore` and the
source is a local file; the whole façade runs with no AWS credentials
(Requirement 19.4, Competition 4). Milestone 6 (task 12.3) can inject an
``S3EvidenceStore`` and an S3-resolving loader behind the same seams without
changing this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

import numpy as np

from vision.annotation import (
    EvidenceStore,
    LocalEvidenceStore,
    annotate_and_store,
    sanitize_frame_ref,
)
from vision.damage_detector import DamageDetector, ModelLoadError
from vision.frame_sampler import Frame, SamplingPolicy, sample_frames
from vision.image_preprocessor import (
    PreprocessConfig,
    PreprocessedFrame,
    preprocess_frame,
)
from vision.region_analyzer import Bbox, analyze_region as _analyze_region_stage
from vision.result_schema import (
    AnalysisResult,
    ComparisonResult,
    Detection,
    FrameRef,
    Method,
    QualityFlag,
    comparison_failure,
    comparison_result,
    failed_result,
    frame_ref as make_frame_ref,
    quality_flag,
    success_result,
)
from vision.video_loader import DecodeError, load_video


# --------------------------------------------------------------------------- #
# Selection criteria for select_frames (7.1)
# --------------------------------------------------------------------------- #


class SelectionStrategy(str, Enum):
    """How :meth:`VisionPipeline.select_frames` picks frames (deterministic).

    * ``EVENLY_SPACED`` - spread the requested count evenly across the whole
      indexed timeline (first and last inclusive). The default: a simple,
      reproducible way to get broad coverage for additional analysis.
    * ``TIME_RANGE`` - restrict to frames whose ``source_timestamp_ms`` falls in
      ``[start_ms, end_ms]`` (inclusive), then take the first ``count`` in
      timestamp order. Lets the agent focus on a temporal neighborhood of an
      observation.
    """

    EVENLY_SPACED = "evenly_spaced"
    TIME_RANGE = "time_range"


@dataclass(frozen=True)
class SelectionCriteria:
    """Criteria describing which additional frames to select (7.1).

    Deterministic by construction: the same criteria against the same indexed
    video always yield the same ordered frames.

    Attributes:
        strategy: The :class:`SelectionStrategy` to apply.
        start_ms: Inclusive lower timestamp bound for ``TIME_RANGE`` (ignored
            otherwise).
        end_ms: Inclusive upper timestamp bound for ``TIME_RANGE`` (ignored
            otherwise).
    """

    strategy: SelectionStrategy = SelectionStrategy.EVENLY_SPACED
    start_ms: int | None = None
    end_ms: int | None = None

    def __post_init__(self) -> None:
        if self.strategy == SelectionStrategy.TIME_RANGE:
            if self.start_ms is None or self.end_ms is None:
                raise ValueError(
                    "TIME_RANGE selection requires both start_ms and end_ms"
                )
            if self.start_ms > self.end_ms:
                raise ValueError(
                    f"start_ms ({self.start_ms}) must be <= end_ms ({self.end_ms})"
                )


# --------------------------------------------------------------------------- #
# The façade Protocol (matches design.md "Vision Pipeline Interface")
# --------------------------------------------------------------------------- #


@runtime_checkable
class VisionPipelineProtocol(Protocol):
    """The single façade the agent tools call (design "Vision Pipeline Interface").

    Mirrors the design Protocol exactly; all four methods return the structured
    ``result_schema`` types and none ever returns a fabricated detection
    (3.3, 6.3).
    """

    def analyze_frame(self, frame_ref: str, method: Method) -> AnalysisResult: ...

    def analyze_region(
        self, frame_ref: str, region: Bbox, method: Method
    ) -> AnalysisResult: ...

    def select_frames(
        self, criteria: SelectionCriteria, count: int
    ) -> list[FrameRef]: ...

    def compare_evidence(self, evidence_refs: list[str]) -> ComparisonResult: ...


# --------------------------------------------------------------------------- #
# Internal per-frame index entry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _IndexedFrame:
    """One indexed frame: the sampled frame and its preprocessing result.

    Holds everything the façade needs to resolve a ``frame_ref`` to pixels and to
    report an honest analysis: the raw sampled :class:`Frame` (real BGR pixels,
    or ``None`` on a decode failure, plus its ``source_timestamp_ms``) and the
    :class:`PreprocessedFrame` (normalized pixels + measured quality + any
    ``poor_input_quality`` flag).
    """

    frame: Frame
    preprocessed: PreprocessedFrame

    @property
    def timestamp_ms(self) -> int:
        """Real source timestamp as an int (UNAVAILABLE sentinel -> 0)."""
        ts = self.frame.source_timestamp_ms
        return int(ts) if isinstance(ts, int) else 0


# --------------------------------------------------------------------------- #
# The concrete façade
# --------------------------------------------------------------------------- #


class VisionPipeline:
    """Concrete :class:`VisionPipelineProtocol` composing the real vision stages.

    Constructed with a local video ``source``, a :class:`SamplingPolicy`, and an
    :class:`~vision.annotation.EvidenceStore`. On first use it samples and
    preprocesses the video once, indexing frames by ``frame_ref`` (see the module
    docstring), and resolves every subsequent tool call against that index so
    results are deterministic and reproducible offline.

    The detector is injected (default: the real classical
    :class:`~vision.damage_detector.DamageDetector`) so the model path (task
    14.2) can be dropped in later without touching the façade.
    """

    def __init__(
        self,
        source: str,
        inspection_id: str,
        *,
        policy: SamplingPolicy | None = None,
        store: EvidenceStore | None = None,
        detector: DamageDetector | None = None,
        preprocess_config: PreprocessConfig | None = None,
    ) -> None:
        """Wire the façade to a video source and the evidence store.

        Args:
            source: Local path to the video to inspect (``s3://`` is reserved for
                Milestone 6 and currently fails loudly via the loader).
            inspection_id: The inspection these analyses/evidence belong to;
                becomes part of every ``evidence_ref`` key.
            policy: Sampling schedule. Defaults to every 5th frame -- broad,
                deterministic coverage for the offline path.
            store: Evidence sink. Defaults to a :class:`LocalEvidenceStore` under
                ``artifacts/`` so the façade runs with no AWS.
            detector: Detection seam. Defaults to the real classical
                :class:`~vision.damage_detector.DamageDetector`.
            preprocess_config: Optional preprocessing/quality policy override.
        """
        if not source:
            raise ValueError("source must be a non-empty video path")
        if not inspection_id:
            raise ValueError("inspection_id must be a non-empty string")
        self._source = source
        self._inspection_id = inspection_id
        self._policy = policy or SamplingPolicy.every_n_frames(5)
        self._store = store or LocalEvidenceStore("artifacts")
        self._detector = detector or DamageDetector()
        self._preprocess_config = preprocess_config or PreprocessConfig()

        # Lazily populated frame index (frame_ref -> _IndexedFrame) and its
        # timestamp-ordered key list. ``None`` until the first sample pass.
        self._frame_index: dict[str, _IndexedFrame] | None = None
        self._ordered_refs: list[str] = []

    # ---- frame_ref resolution ------------------------------------------- #

    def _index(self) -> dict[str, _IndexedFrame]:
        """Sample + preprocess the video once and index frames by ``frame_ref``.

        Memoized: the first call does the real decode/sample/preprocess pass and
        every later call returns the cached index, so repeated tool calls resolve
        the same ``frame_ref`` to the same pixels (deterministic, 19.2-19.5).

        A :class:`~vision.video_loader.DecodeError` here is not swallowed; it
        means the whole video is undecodable, which the backend surfaces as a
        validation failure (1.3) upstream of the agent loop. Individual
        undecodable *frames* are indexed normally (the sampler marks them with a
        ``decode_failure`` flag) so ``analyze_frame`` can fail them honestly.
        """
        if self._frame_index is not None:
            return self._frame_index

        index: dict[str, _IndexedFrame] = {}
        with load_video(self._source) as video:
            for frame in sample_frames(video.capture, video.metadata, self._policy):
                pre = preprocess_frame(frame, self._preprocess_config)
                index[frame.frame_ref] = _IndexedFrame(frame=frame, preprocessed=pre)

        # Order by real source timestamp (ascending); the sampler already yields
        # ascending frame indices, so this is stable and deterministic.
        self._ordered_refs = sorted(
            index, key=lambda ref: (index[ref].timestamp_ms, ref)
        )
        self._frame_index = index
        return index

    def _resolve(self, frame_ref: str) -> _IndexedFrame | None:
        """Return the indexed frame for ``frame_ref`` or ``None`` if unknown."""
        return self._index().get(frame_ref)

    # ---- analyze_frame (2.2, 2.3, 2.4, 2.6, 6.3) ------------------------- #

    def analyze_frame(self, frame_ref: str, method: Method) -> AnalysisResult:
        """Analyze a whole frame referenced by ``frame_ref`` (Requirement 6).

        Resolves the frame, runs detection with the selected ``method``,
        annotates + stores evidence, and returns a
        :func:`~vision.result_schema.success_result` with the detections and
        ``evidence_ref`` set. Any failure -- unknown ref, decode failure, or a
        detector error (notably ``method="model"`` before task 14.2) -- returns a
        :func:`~vision.result_schema.failed_result`, which has empty detections by
        construction and so can never become a finding (2.6, 6.3, Competition 3).
        """
        indexed = self._resolve(frame_ref)
        if indexed is None:
            # No pixels to analyze; never fabricate them.
            return failed_result(
                method=method,
                frame_ref=frame_ref,
                source_timestamp_ms=0,
                failure_reason=(
                    f"frame_ref {frame_ref!r} does not resolve to a sampled frame"
                ),
            )

        frame = indexed.frame
        ts = indexed.timestamp_ms

        # A frame that did not decode has no real pixels. Surface its decode
        # flags on an honest failure (empty detections).
        if not frame.ok or frame.image is None:
            return failed_result(
                method=method,
                frame_ref=frame_ref,
                source_timestamp_ms=ts,
                failure_reason="frame did not decode; no pixels to analyze",
                quality_flags=list(frame.quality_flags),
            )

        # Run detection on the preprocessed pixels when available (normalized,
        # Record the selected method on the result.
        #
        # Classical detection runs on the RAW sampled frame, not the heavily
        # denoised preprocessor output: the detector owns its own grayscale
        # normalization, Canny, and morphology, and the preprocessor's
        # fastNlMeansDenoising + downscale erases the very edge/texture signal the
        # classical path keys on (verified: denoised frames yield zero
        # detections on real samples). The preprocessor's role is the honest
        # INPUT-QUALITY assessment (its quality score + poor_input_quality flag),
        # which is still carried onto the result below; it is not the detection
        # substrate. The model path (task 14.2) will do its own letterbox/resize.
        detect_image = frame.image if frame.image is not None else indexed.preprocessed.image
        # The method actually used may change if the Model_Path falls back to
        # classical on a model-load failure (R4.5); track it honestly for the
        # result + trace rather than reporting the requested method.
        used_method: Method = method
        quality_flags: list[QualityFlag] = list(indexed.preprocessed.quality_flags)
        try:
            detections: list[Detection] = self._detector.detect(detect_image, method)
        except ModelLoadError as exc:
            # R4.5: the Model_Path could not load (absent artifact or this OpenCV
            # build cannot parse the ONNX). Fall back to the Classical_Path and
            # RECORD the fallback honestly -- the result reports method="classical"
            # (what actually ran) plus a quality flag naming the model-load failure.
            # We never report model detections that did not run.
            try:
                detections = self._detector.detect(detect_image, "classical")
            except ValueError as cexc:
                return failed_result(
                    method="classical",
                    frame_ref=frame_ref,
                    source_timestamp_ms=ts,
                    failure_reason=f"classical fallback after model-load failure also failed: {cexc}",
                    quality_flags=quality_flags,
                )
            used_method = "classical"
            quality_flags = quality_flags + [
                quality_flag(
                    "poor_input_quality",
                    f"model path unavailable ({exc}); fell back to the classical path (R4.5)",
                )
            ]
        except ValueError as exc:
            # Unknown method literal or bad pixels: honest failure, no detections.
            return failed_result(
                method=method,
                frame_ref=frame_ref,
                source_timestamp_ms=ts,
                failure_reason=f"detection failed: {exc}",
                quality_flags=quality_flags,
            )

        # 2.3: annotate the REAL frame with the REAL detections and store it.
        # Evidence is generated even for zero detections so the trace always has
        # a frame to show; nothing is drawn that was not detected (3.3).
        try:
            evidence_ref = annotate_and_store(
                frame.image,
                detections,
                inspection_id=self._inspection_id,
                frame_ref=frame_ref,
                store=self._store,
            )
        except OSError as exc:
            # Evidence could not be persisted. The analysis itself succeeded, but
            # a finding must reference real evidence (10.3), so fail honestly
            # rather than return a success with no evidence_ref.
            return failed_result(
                method=method,
                frame_ref=frame_ref,
                source_timestamp_ms=ts,
                failure_reason=f"evidence storage failed: {exc}",
                quality_flags=quality_flags,
            )

        return success_result(
            method=used_method,  # reflects classical fallback on model-load failure (R4.5)
            frame_ref=frame_ref,
            source_timestamp_ms=ts,
            detections=detections,
            quality_flags=quality_flags,
            evidence_ref=evidence_ref,
        )

    # ---- analyze_region (8.1, 8.2, 8.3) --------------------------------- #

    def analyze_region(
        self, frame_ref: str, region: Bbox, method: Method
    ) -> AnalysisResult:
        """Analyze a region within a frame (Requirement 8).

        Resolves the frame and delegates region-scoped detection (including the
        out-of-bounds validation, 8.3) to :mod:`vision.region_analyzer`, injecting
        the real classical detector. On a successful, non-empty region analysis
        the façade annotates + stores a **crop** of the region as evidence and
        sets ``evidence_ref`` on the result (2.3). An unknown ``frame_ref`` is an
        honest :func:`~vision.result_schema.failed_result` (no pixels to analyze).
        """
        indexed = self._resolve(frame_ref)
        if indexed is None:
            return failed_result(
                method=method,
                frame_ref=frame_ref,
                source_timestamp_ms=0,
                failure_reason=(
                    f"frame_ref {frame_ref!r} does not resolve to a sampled frame"
                ),
            )

        # Delegate to the region analyzer: it enforces frame-bounds validation
        # (8.3) and decode-failure handling, and composes detection through the
        # same detector seam. Inject the real classical detector explicitly so
        # the region path uses exactly the façade's configured detector.
        try:
            result = _analyze_region_stage(
                indexed.frame, region, method, detector=self._detector
            )
        except ModelLoadError:
            # R4.5 (task 14.3): the Model_Path could not load; fall back to a
            # classical region analysis and let the result record method=classical.
            # We never crash the loop or report model detections that did not run.
            result = _analyze_region_stage(
                indexed.frame, region, "classical", detector=self._detector
            )
        except ValueError as exc:
            return failed_result(
                method=method,
                frame_ref=frame_ref,
                source_timestamp_ms=indexed.timestamp_ms,
                failure_reason=f"region analysis failed: {exc}",
            )

        # Only a successful region analysis with real detections gets evidence.
        # A failed region analysis (out of bounds, decode failure, model-path not
        # wired) must not store evidence -- there is nothing honest to annotate,
        # and a failed result cannot become a finding (2.6, 6.3).
        if result["status"] != "success" or not result["detections"]:
            return result

        frame = indexed.frame
        if frame.image is None:  # defensive; success implies decoded pixels
            return result

        # Annotate a crop of the region so the evidence frame focuses on the ROI.
        x, y, w, h = region
        crop = frame.image[y : y + h, x : x + w]
        # Detections are in frame coordinates; shift them into crop-local space
        # so the boxes land correctly on the crop.
        crop_local = [_shift_detection(d, -x, -y) for d in result["detections"]]
        region_ref = f"{frame_ref}@region_{x}_{y}_{w}_{h}"
        try:
            evidence_ref = annotate_and_store(
                crop,
                crop_local,
                inspection_id=self._inspection_id,
                frame_ref=region_ref,
                store=self._store,
            )
        except OSError as exc:
            return failed_result(
                method=method,
                frame_ref=frame_ref,
                source_timestamp_ms=indexed.timestamp_ms,
                failure_reason=f"region evidence storage failed: {exc}",
                quality_flags=list(result["quality_flags"]),
            )

        # Return the region result with evidence_ref set (success keeps its real
        # detections and quality flags).
        return success_result(
            method=result["method"],
            frame_ref=result["frame_ref"],
            source_timestamp_ms=result["source_timestamp_ms"],
            detections=result["detections"],
            quality_flags=result["quality_flags"],
            evidence_ref=evidence_ref,
        )

    # ---- select_frames (7.1, 7.2, 7.3) ---------------------------------- #

    def select_frames(
        self, criteria: SelectionCriteria, count: int
    ) -> list[FrameRef]:
        """Return an ordered, timestamped list of frame refs (Requirement 7).

        Deterministic: the same criteria against the same indexed video always
        return the same frames in ascending ``source_timestamp_ms`` order (7.2).
        When nothing matches the criteria the result is an empty list (7.3); the
        agent tool layer (task 6.5) maps that to a "no matching frames" status.

        Args:
            criteria: The :class:`SelectionCriteria` to apply.
            count: Maximum number of frames to return (``<= 0`` -> empty list).

        Returns:
            Up to ``count`` :class:`~vision.result_schema.FrameRef` records in
            ascending timestamp order.
        """
        if count <= 0:
            return []

        index = self._index()
        ordered = self._ordered_refs  # ascending timestamp order

        if criteria.strategy == SelectionStrategy.TIME_RANGE:
            assert criteria.start_ms is not None and criteria.end_ms is not None
            selected = [
                ref
                for ref in ordered
                if criteria.start_ms <= index[ref].timestamp_ms <= criteria.end_ms
            ][:count]
        else:  # EVENLY_SPACED
            selected = _evenly_spaced(ordered, count)

        return [
            make_frame_ref(ref, index[ref].timestamp_ms) for ref in selected
        ]

    # ---- compare_evidence (9.1, 9.2, 9.3) ------------------------------- #

    def compare_evidence(self, evidence_refs: list[str]) -> ComparisonResult:
        """Compare two or more evidence items (Requirement 9).

        Fewer than two **valid** evidence items -> a
        :func:`~vision.result_schema.comparison_failure` validation failure (9.3).
        Otherwise the façade resolves each evidence key back to its source frame
        (the key embeds the sanitized ``frame_ref``), re-analyzes each with the
        classical path, and reports **agreement** when the items agree on whether
        a hazard is present (all have detections, or all have none) and
        **disagreement** otherwise. The verdict is derived from real re-analysis;
        nothing about agreement is fabricated (3.3, 9.2).

        Args:
            evidence_refs: Evidence keys produced by :meth:`analyze_frame` /
                :meth:`analyze_region`.

        Returns:
            A :class:`~vision.result_schema.ComparisonResult`.
        """
        # De-duplicate while preserving order; a ref compared against itself is
        # not two independent items.
        unique_refs: list[str] = []
        for ref in evidence_refs:
            if ref and ref not in unique_refs:
                unique_refs.append(ref)

        # Resolve each evidence key to a real re-analysis. An unresolvable key is
        # not a valid evidence item and does not count toward the >= 2 minimum.
        presence: list[bool] = []
        resolved_refs: list[str] = []
        for ref in unique_refs:
            frame_ref = self._evidence_ref_to_frame_ref(ref)
            if frame_ref is None:
                continue
            result = self.analyze_frame(frame_ref, "classical")
            if result["status"] != "success":
                continue  # a failed re-analysis is not a valid evidence item
            resolved_refs.append(ref)
            presence.append(len(result["detections"]) > 0)

        if len(resolved_refs) < 2:
            return comparison_failure(
                list(evidence_refs),
                failure_reason=(
                    "fewer than two valid evidence items could be resolved and "
                    f"re-analyzed (resolved {len(resolved_refs)} of "
                    f"{len(evidence_refs)}); a comparison needs at least two (9.3)"
                ),
            )

        all_present = all(presence)
        none_present = not any(presence)
        if all_present or none_present:
            state = "a hazard" if all_present else "no hazard"
            return comparison_result(
                "agreement",
                resolved_refs,
                detail=(
                    f"all {len(resolved_refs)} evidence items agree: each shows "
                    f"{state} on re-analysis"
                ),
            )
        present_n = sum(1 for p in presence if p)
        return comparison_result(
            "disagreement",
            resolved_refs,
            detail=(
                f"{present_n} of {len(resolved_refs)} evidence items show a hazard "
                "on re-analysis while the rest do not"
            ),
        )

    def _evidence_ref_to_frame_ref(self, evidence_ref: str) -> str | None:
        """Map an ``evidence_ref`` key back to the indexed ``frame_ref`` it came from.

        An evidence key is ``evidence/<inspection_id>/<sanitized_frame_ref>.png``.
        The sanitized stem is not reversible in general, so the façade matches it
        against the sanitized form of each indexed ``frame_ref`` -- which is
        exactly how the key was produced -- and returns the first frame whose
        sanitized ref matches. Region-crop evidence (``...@region_...``) resolves
        to its parent frame. Returns ``None`` when no indexed frame matches.
        """
        if not evidence_ref:
            return None
        stem = evidence_ref.rsplit("/", 1)[-1]
        if stem.endswith(".png"):
            stem = stem[: -len(".png")]

        index = self._index()
        for ref in self._ordered_refs:
            if sanitize_frame_ref(ref) == stem:
                return ref
            # Region evidence embeds the parent frame_ref plus a region suffix.
            region_ref = f"{ref}@region_"
            if sanitize_frame_ref(region_ref) and stem.startswith(
                sanitize_frame_ref(ref)
            ):
                # Best-effort parent match for region crops.
                if ref in index:
                    return ref
        return None


# --------------------------------------------------------------------------- #
# Small pure helpers
# --------------------------------------------------------------------------- #


def _shift_detection(det: Detection, dx: int, dy: int) -> Detection:
    """Return a copy of ``det`` with its bbox origin shifted by ``(dx, dy)``.

    Used to move frame-coordinate detections into crop-local space for region
    evidence annotation. Rebuilds through the detection factory so the
    pixel-labeled measurement and bounds validation are preserved.
    """
    from vision.result_schema import detection, pixel_measurement

    x, y, w, h = det["bbox_px"]
    nx, ny = max(0, x + dx), max(0, y + dy)
    return detection(
        det["hazard_type"],
        (nx, ny, w, h),
        det["confidence"],
        pixel_measurement(w, h, area_px=det["measurement"]["area_px"]),
    )


def _evenly_spaced(ordered: list[str], count: int) -> list[str]:
    """Pick ``count`` items evenly spaced across ``ordered`` (first/last inclusive).

    Deterministic and order-preserving. Fewer items than ``count`` returns them
    all; ``count == 1`` returns the first item.
    """
    n = len(ordered)
    if n == 0 or count <= 0:
        return []
    if count >= n:
        return list(ordered)
    if count == 1:
        return [ordered[0]]
    # Evenly spaced indices from 0 to n-1 inclusive.
    step = (n - 1) / (count - 1)
    picked_indices = sorted({int(round(i * step)) for i in range(count)})
    # Rounding can collapse two slots onto one index near the ends; backfill to
    # honor the requested count deterministically.
    out = [ordered[i] for i in picked_indices]
    if len(out) < count:
        for ref in ordered:
            if ref not in out:
                out.append(ref)
            if len(out) == count:
                break
        out = sorted(out, key=ordered.index)
    return out


# --------------------------------------------------------------------------- #
# Self-validation (import-time smoke check: real end-to-end pipeline run)
# --------------------------------------------------------------------------- #


def _self_check() -> None:
    """Run the real façade on a generated sample clip, end to end.

    Minimal real-OpenCV guard (no mocks), not the optional integration suite
    (task 4.7). Decodes a generated clip, resolves a frame_ref, runs
    ``analyze_frame`` (evidence file exists), proves a failed analysis has empty
    detections, exercises ``select_frames`` ordering, and checks
    ``compare_evidence`` honesty. Runs under ``python -m``.
    """
    import tempfile
    from pathlib import Path

    from vision.samples.make_samples import make_samples

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        clips = make_samples(tmp_path)
        pothole_clip = next(c for c in clips if "pothole" in c.name)

        store = LocalEvidenceStore(tmp_path / "artifacts")
        pipeline = VisionPipeline(
            str(pothole_clip),
            inspection_id="insp-selfcheck",
            policy=SamplingPolicy.every_n_frames(5),
            store=store,
        )

        # The façade satisfies the design Protocol.
        assert isinstance(pipeline, VisionPipelineProtocol)

        # select_frames: ordered, timestamped, deterministic, bounded by count.
        refs = pipeline.select_frames(SelectionCriteria(), count=3)
        assert refs, "select_frames returned nothing from a real clip"
        assert len(refs) <= 3
        ts = [r["source_timestamp_ms"] for r in refs]
        assert ts == sorted(ts), f"frame refs not timestamp-ordered: {ts}"
        # Deterministic: identical criteria -> identical selection.
        assert pipeline.select_frames(SelectionCriteria(), count=3) == refs

        # count <= 0 -> empty list.
        assert pipeline.select_frames(SelectionCriteria(), 0) == []

        # analyze_frame on a real resolved frame -> success with evidence on disk.
        first_ref = refs[0]["frame_ref"]
        res = pipeline.analyze_frame(first_ref, "classical")
        assert res["status"] == "success", res["failure_reason"]
        assert res["method"] == "classical"
        assert res["evidence_ref"], "successful analysis must set evidence_ref"
        evidence_path = store.key_to_path(res["evidence_ref"])
        assert evidence_path.exists(), f"evidence file not written: {evidence_path}"

        # KEY INVARIANT (2.6, 6.3, Competition 3): a failed analysis has NO
        # detections. An unknown frame_ref is the simplest honest failure.
        failed = pipeline.analyze_frame("no-such#frame_99999999", "classical")
        assert failed["status"] == "failed"
        assert failed["detections"] == [], "failed analysis must have empty detections"
        assert failed["evidence_ref"] is None
        assert failed["failure_reason"]

        # method="model" with no fetched ONNX artifact falls back to the
        # Classical_Path and RECORDS the fallback (R4.5, task 14.3): the result
        # reports method="classical" (what actually ran) with a quality flag
        # naming the model-load failure -- never model detections that did not run.
        model_res = pipeline.analyze_frame(first_ref, "model")
        assert model_res["status"] == "success"
        assert model_res["method"] == "classical", "must report the method that actually ran (R4.5)"
        assert any(
            q["code"] == "poor_input_quality" and "model path unavailable" in q["detail"]
            for q in model_res["quality_flags"]
        ), "the model->classical fallback must be recorded honestly"

        # analyze_region: out-of-bounds -> validation failure, no detections (8.3).
        oob = pipeline.analyze_region(first_ref, (10_000, 10_000, 50, 50), "classical")
        assert oob["status"] == "failed"
        assert oob["detections"] == []

        # analyze_region: a valid in-bounds region succeeds (classical path).
        region_res = pipeline.analyze_region(first_ref, (0, 0, 100, 100), "classical")
        assert region_res["status"] == "success", region_res["failure_reason"]

        # compare_evidence: < 2 valid items -> validation failure (9.3).
        assert pipeline.compare_evidence([res["evidence_ref"]])["status"] == "failed"
        assert pipeline.compare_evidence([])["status"] == "failed"

        # compare_evidence: two real evidence items -> an honest verdict (9.2).
        second_ref = refs[1]["frame_ref"]
        res2 = pipeline.analyze_frame(second_ref, "classical")
        cmp = pipeline.compare_evidence([res["evidence_ref"], res2["evidence_ref"]])
        assert cmp["status"] == "success"
        assert cmp["agreement"] in ("agreement", "disagreement")
        assert len(cmp["evidence_refs"]) == 2


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    _self_check()
    print(
        "[pipeline] OK: VisionPipeline façade composes load->sample->preprocess->"
        "detect->annotate+store; analyze_frame/analyze_region/select_frames/"
        "compare_evidence return result_schema types; failed analyses carry empty "
        "detections (2.6, 6.3); method='model' falls back to classical and records it (R4.5)."
    )
