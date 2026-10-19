"""Failure-handling hardening audit (task 16.3).

Exercises each failure path from the design's error-handling table and asserts
it is recorded HONESTLY - empty detections on failure, an explicit reason/flag,
and no fabricated values - across the real modules. Offline, no AWS.

Error-handling table rows covered:
  invalid upload -> validation reason (upload_handler, infra)   [covered separately]
  frame decode failure           -> Quality_Flag
  low quality / poor input       -> Quality_Flag
  failed vision analysis         -> cannot become a finding
  model load failure             -> classical fallback, recorded
  region out of bounds           -> validation failure, no detections
  fewer than 2 evidence          -> validation failure
  tool fails past retry cap      -> escalate or complete
  missing data value             -> explicit UNAVAILABLE
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from vision.result_schema import UNAVAILABLE, failed_result, success_result
from vision.frame_sampler import Frame, SamplingPolicy
from vision.image_preprocessor import preprocess_frame
from vision.region_analyzer import analyze_region
from vision.result_schema import quality_flag as _qf


def _pipeline(tmp: Path):
    from vision.annotation import LocalEvidenceStore
    from vision.pipeline import VisionPipeline
    from vision.samples.make_samples import make_samples

    clips = make_samples(tmp / "clips")
    pot = next(c for c in clips if "pothole" in c.name)
    return VisionPipeline(
        str(pot), inspection_id="audit",
        policy=SamplingPolicy.every_n_frames(5),
        store=LocalEvidenceStore(tmp / "art"),
    )


def test_decode_failure_frame_carries_quality_flag_and_no_pixels():
    # frame_sampler emits a decode_failure flag; preprocessor propagates it.
    f = Frame(frame_ref="x#frame_1", frame_index=1, source_timestamp_ms=0,
              image=None, quality_flags=[_qf("decode_failure", "undecodable")])
    pre = preprocess_frame(f)
    assert not pre.ok and pre.image is None
    assert any(q["code"] == "decode_failure" for q in pre.quality_flags)
    assert pre.quality_score == 0.0  # not fabricated


def test_poor_input_quality_is_flagged_not_fabricated():
    dark_blur = np.zeros((240, 320, 3), dtype=np.uint8)  # black, zero texture
    f = Frame(frame_ref="x#frame_0", frame_index=0, source_timestamp_ms=0,
              image=dark_blur, quality_flags=[])
    pre = preprocess_frame(f)
    assert pre.is_low_quality
    assert any(q["code"] == "poor_input_quality" for q in pre.quality_flags)


def test_failed_analysis_cannot_become_a_finding():
    bad = failed_result(method="classical", frame_ref="f", source_timestamp_ms=0,
                        failure_reason="decode_failure")
    assert bad["status"] == "failed" and bad["detections"] == []  # empty by construction


def test_model_load_failure_falls_back_to_classical_and_records_it():
    from vision.pipeline import SelectionCriteria
    with tempfile.TemporaryDirectory() as t:
        pipe = _pipeline(Path(t))
        ref = pipe.select_frames(SelectionCriteria(), 1)[0]["frame_ref"]
        res = pipe.analyze_frame(ref, "model")  # no ONNX artifact -> classical fallback
        assert res["status"] == "success" and res["method"] == "classical"
        assert any("model path unavailable" in q["detail"] for q in res["quality_flags"])


def test_region_out_of_bounds_is_validation_failure_with_no_detections():
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    frame = Frame(frame_ref="x#frame_0", frame_index=0, source_timestamp_ms=0,
                  image=img, quality_flags=[])
    res = analyze_region(frame, (10_000, 10_000, 50, 50), "classical")
    assert res["status"] == "failed" and res["detections"] == []
    assert res["failure_reason"]


def test_fewer_than_two_evidence_is_validation_failure():
    from agent.tools.selection import compare_visual_evidence

    class _P:
        def compare_evidence(self, refs):  # pragma: no cover - not reached with <2
            raise AssertionError("should not delegate with < 2 refs")
    res = compare_visual_evidence(_P(), {"evidence_refs": ["only-one"]})
    assert res["status"] == "failed" and res["agreement"] == "unavailable"
    assert res["failure_reason"]


def test_tool_failure_past_retry_cap_escalates_or_completes():
    # A decider that always analyzes a missing frame -> analyze_frame keeps
    # failing; past the retry cap the loop escalates then completes.
    from agent.decider_service import MockDecider
    from agent.state import AgentDecision, InspectionState, ToolName
    from agent.store import InMemoryStore
    from agent.loop import run_inspection

    class _MissPipeline:
        def analyze_frame(self, frame_ref, method):
            return failed_result(method=method, frame_ref=frame_ref,
                                 source_timestamp_ms=0, failure_reason="decode_failure")
        def analyze_region(self, frame_ref, region, method):
            return failed_result(method=method, frame_ref=frame_ref,
                                 source_timestamp_ms=0, failure_reason="decode_failure")
        def select_frames(self, criteria, count):
            from vision.result_schema import frame_ref as fr
            return [fr("f", 0)]
        def compare_evidence(self, refs):
            from vision.result_schema import comparison_failure
            return comparison_failure(refs, "n/a")

    store = InMemoryStore()
    rep = run_inspection(_MissPipeline(), store, "retry",
                         decider=MockDecider(),
                         state=InspectionState(max_iterations=10, max_retries_per_tool=2),
                         seed_frame_ref="f", aws_credentials_available=False)
    assert rep["status"] == "completed"  # loop terminates honestly
    trace = store.get_inspection_bundle("retry")["trace"]
    tools = {e["tool"] for e in trace}
    # escalation and/or completion are recorded; nothing saved from failed analysis
    assert "request_human_review" in tools or "complete_inspection" in tools
    assert rep["finding_count"] == 0


def test_missing_value_is_explicit_unavailable():
    r = success_result(method="classical", frame_ref="f", source_timestamp_ms=0, detections=[])
    # evidence_ref unset -> None (explicit), measurement physical always unavailable
    from vision.result_schema import pixel_measurement
    m = pixel_measurement(10, 20)
    assert m["physical"] == UNAVAILABLE and m["unit"] == "pixels"
