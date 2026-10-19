"""Vision_Worker entrypoint: run ONE real inspection end to end on Fargate.

This is what the deployed Fargate task runs (Milestone 6 task 12.5): it takes an
inspection id + the S3 object key of an uploaded video, downloads the video,
runs the REAL OpenCV 5 pipeline + the bounded agent loop, and persists the
trace, findings, review queue, report, and metrics to DynamoDB. It is the
component that makes cloud video processing actually happen (not a placeholder).

Inputs (from the ECS task environment / command):
  INSPECTION_ID   - the inspection to process (DynamoDB partition key)
  OBJECT_KEY      - the S3 key of the uploaded video (uploads/<id>/<name>)
  EVIDENCE_BUCKET - S3 bucket for video input + annotated-frame evidence
  ROADGUARD_TABLE - DynamoDB single table
  RUNTIME         - "mock" (default) or a live reasoning runtime

It reuses the SAME vision pipeline and agent loop as the offline backbone; only
the Store (DynamoDB) and the video source (S3) are the cloud implementations, so
behavior matches the local Mock_Runtime exactly (deterministic, honest).

Verifies OpenCV 5 on boot, then processes. Exits non-zero on an unrecoverable
error so the ECS task surfaces failure honestly (never a silent success).
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import traceback
from pathlib import Path


def _log(msg: str) -> None:
    print(f"[vision_worker] {msg}", flush=True)


def main() -> int:
    # 1. Verify OpenCV 5 up front (fail loudly if the image is wrong).
    from vision.verify_opencv import check_opencv_version
    try:
        info = check_opencv_version()
        _log(f"OpenCV {info.version} verified (major {info.major})")
    except Exception as exc:  # pragma: no cover - image-build guard
        _log(f"OpenCV 5 verification failed: {exc}")
        return 2

    inspection_id = os.environ.get("INSPECTION_ID", "").strip()
    object_key = os.environ.get("OBJECT_KEY", "").strip()
    bucket = os.environ.get("EVIDENCE_BUCKET", "").strip()
    if not inspection_id or not object_key or not bucket:
        _log("missing INSPECTION_ID / OBJECT_KEY / EVIDENCE_BUCKET")
        return 2

    import boto3
    from vision.frame_sampler import SamplingPolicy
    from vision.pipeline import VisionPipeline
    from vision.annotation import EvidenceStore
    from agent.loop import run_inspection
    from agent.state import InspectionState

    s3 = boto3.client("s3")

    # 2. Download the uploaded video to local scratch for OpenCV decoding.
    with tempfile.TemporaryDirectory() as tmp:
        local_video = str(Path(tmp) / Path(object_key).name)
        _log(f"downloading s3://{bucket}/{object_key}")
        try:
            s3.download_file(bucket, object_key, local_video)
        except Exception as exc:
            _log(f"failed to download video: {exc}")
            _mark_failed(inspection_id, f"video download failed: {exc}")
            return 1

        # 3. Build the cloud-backed collaborators: an S3 evidence store and the
        #    DynamoDB store. Both honor the same seams as the offline fakes.
        store = _DynamoAgentStore()
        evidence_store = _S3EvidenceStore(s3, bucket)

        pipeline = VisionPipeline(
            local_video,
            inspection_id=inspection_id,
            policy=SamplingPolicy.every_n_frames(5),
            store=evidence_store,
        )

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
            return 1
        latency_ms = int((time.time() - started) * 1000)

        # 4. Persist operational metrics (detection metrics need labeled GT, so
        #    they stay the explicit UNAVAILABLE sentinel here).
        from agent.metrics import compute_operational_metrics
        bundle = store.get_inspection_bundle(inspection_id)
        metrics = compute_operational_metrics(
            bundle.get("trace", []),
            finding_count=report["finding_count"],
            escalation_count=report["escalation_count"],
            latency_ms=latency_ms,
        )
        store.put_metrics(inspection_id, metrics, upload_ts=str(int(started)))

        _log(
            f"done: status={report['status']} findings={report['finding_count']} "
            f"escalations={report['escalation_count']} trace={report['trace_length']} "
            f"latency_ms={latency_ms}"
        )
    return 0


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


# --- cloud-backed Store adapter over the DynamoDB single-table store --------- #


class _DynamoAgentStore:
    """Adapts the handlers' DynamoStore to the agent.store.Store interface.

    The agent loop/tools call put_inspection/append_trace/put_finding/
    enqueue_review/put_metrics/get_inspection_bundle. The handlers' DynamoStore
    provides the same operations against the single table; this thin adapter
    supplies the per-entity sequence numbers the finding/review writes need.
    """

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


if __name__ == "__main__":
    raise SystemExit(main())
