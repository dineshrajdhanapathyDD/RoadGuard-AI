"""Shared inspection orchestration: download -> OpenCV pipeline -> agent loop -> persist.

This is the SINGLE implementation of a full cloud inspection, extracted so both
the (legacy) Fargate ``vision/worker_main.py`` and the consolidated AgentCore
entrypoint call the same code. It:

  1. verifies OpenCV 5,
  2. downloads the uploaded video from S3,
  3. runs the REAL OpenCV 5 Vision_Pipeline + the bounded agent loop (the loop
     resolves its reasoning backend via ``resolve_decider`` -- Amazon Nova via
     Strands when ``RUNTIME!=mock`` and credentials resolve, else the mock),
  4. persists trace/findings/review/metrics + a completed META to the DynamoDB
     single table, and annotated evidence frames to S3.

It is deliberately free of any Fargate- or AgentCore-specific assumption: it
takes plain ``(inspection_id, object_key, bucket)`` and returns a summary dict.
"""

from __future__ import annotations

import tempfile
import time
import traceback
from pathlib import Path
from typing import Any


def _log(msg: str) -> None:
    print(f"[inspection_job] {msg}", flush=True)


def run_inspection_job(inspection_id: str, object_key: str, bucket: str) -> dict[str, Any]:
    """Run one full inspection and persist results. Returns a summary dict.

    Honest failure: on a download or inspection error the inspection META is
    marked ``failed`` with the reason and the summary carries ``ok=False``; the
    function never raises for an operational failure (the caller gets a result).
    """
    # 1. Verify OpenCV 5 up front (fail loudly if the image is wrong).
    from vision.verify_opencv import check_opencv_version
    info = check_opencv_version()
    _log(f"OpenCV {info.version} verified (major {info.major})")

    if not inspection_id or not object_key or not bucket:
        return {"ok": False, "error": "missing inspection_id/object_key/bucket"}

    import boto3
    from vision.frame_sampler import SamplingPolicy
    from vision.pipeline import VisionPipeline
    from agent.loop import run_inspection
    from agent.state import InspectionState

    s3 = boto3.client("s3")

    with tempfile.TemporaryDirectory() as tmp:
        local_video = str(Path(tmp) / Path(object_key).name)
        _log(f"downloading s3://{bucket}/{object_key}")
        try:
            s3.download_file(bucket, object_key, local_video)
        except Exception as exc:
            _mark_failed(inspection_id, f"video download failed: {exc}")
            return {"ok": False, "inspection_id": inspection_id, "error": f"download failed: {exc}"}

        store = _DynamoAgentStore()
        evidence_store = _S3EvidenceStore(s3, bucket)
        pipeline = VisionPipeline(
            local_video,
            inspection_id=inspection_id,
            policy=SamplingPolicy.every_n_frames(5),
            store=evidence_store,
        )

        import os
        _log(f"running inspection {inspection_id} (runtime={os.environ.get('RUNTIME','mock')})")
        started = time.time()
        try:
            report = run_inspection(
                pipeline, store, inspection_id,
                state=InspectionState(max_iterations=20, max_retries_per_tool=2),
            )
        except Exception as exc:
            _log("inspection crashed:\n" + traceback.format_exc())
            _mark_failed(inspection_id, f"inspection error: {exc}")
            return {"ok": False, "inspection_id": inspection_id, "error": f"inspection error: {exc}"}
        latency_ms = int((time.time() - started) * 1000)

        from agent.metrics import compute_operational_metrics
        bundle = store.get_inspection_bundle(inspection_id)
        metrics = compute_operational_metrics(
            bundle.get("trace", []),
            finding_count=report["finding_count"],
            escalation_count=report["escalation_count"],
            latency_ms=latency_ms,
        )
        store.put_metrics(inspection_id, metrics, upload_ts=str(int(started)))

        summary = {
            "ok": True,
            "inspection_id": inspection_id,
            "status": report["status"],
            "finding_count": report["finding_count"],
            "escalation_count": report["escalation_count"],
            "trace_length": report["trace_length"],
            "latency_ms": latency_ms,
        }
        _log(
            f"done: status={summary['status']} findings={summary['finding_count']} "
            f"escalations={summary['escalation_count']} trace={summary['trace_length']} "
            f"latency_ms={latency_ms}"
        )
        return summary


def _mark_failed(inspection_id: str, reason: str) -> None:
    """Best-effort: record an honest failure on the inspection META."""
    try:
        from dynamo_store import DynamoStore  # flat copy in the image
    except Exception:
        try:
            from infra.handlers.dynamo_store import DynamoStore  # local dev
        except Exception:
            return
    try:
        store = DynamoStore()
        bundle = store.get_inspection_bundle(inspection_id)
        meta = bundle.get("meta")
        rec = dict(meta) if isinstance(meta, dict) else {"inspection_id": inspection_id}
        rec.update({"inspection_id": inspection_id, "status": "failed",
                    "stop_reason": reason, "progress": 100})
        store.put_inspection(rec)
    except Exception:  # pragma: no cover
        pass


class _DynamoAgentStore:
    """Adapts the handlers' DynamoStore to the agent.store.Store interface."""

    def __init__(self) -> None:
        try:
            from dynamo_store import DynamoStore  # image layout
        except Exception:
            from infra.handlers.dynamo_store import DynamoStore  # local dev
        self._s = DynamoStore()
        self._finding_seq = 0
        self._review_seq = 0

    def put_inspection(self, record):
        self._s.put_inspection(dict(record))

    def append_trace(self, inspection_id, entry):
        self._s.append_trace(inspection_id, dict(entry))

    def put_finding(self, inspection_id, finding):
        self._s.put_finding(inspection_id, dict(finding), self._finding_seq)
        self._finding_seq += 1

    def enqueue_review(self, inspection_id, item):
        self._s.enqueue_review(inspection_id, dict(item), self._review_seq)
        self._review_seq += 1

    def put_metrics(self, inspection_id, metrics, **kw):
        self._s.put_metrics(inspection_id, dict(metrics), upload_ts=kw.get("upload_ts", ""))

    def get_inspection_bundle(self, inspection_id):
        return self._s.get_inspection_bundle(inspection_id)


class _S3EvidenceStore:
    """EvidenceStore that writes annotated frames to S3 under evidence/<id>/."""

    def __init__(self, s3_client, bucket: str) -> None:
        self._s3 = s3_client
        self._bucket = bucket

    def save(self, inspection_id: str, frame_ref: str, image) -> str:
        import cv2
        from vision.annotation import evidence_key
        key = evidence_key(inspection_id, frame_ref)
        ok, buf = cv2.imencode(".png", image)
        if not ok:
            raise OSError(f"failed to encode evidence PNG for {frame_ref}")
        self._s3.put_object(Bucket=self._bucket, Key=key, Body=buf.tobytes(),
                            ContentType="image/png")
        return key
