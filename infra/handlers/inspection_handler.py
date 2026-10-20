"""inspection_handler: start an inspection and launch the Fargate worker (12.4).

POST /inspections  { "object_key": str }
  -> 202 { "inspection_id", "status": "processing" }

Validates the object_key, creates the inspection META record, and launches the
ECS/Fargate Vision_Worker task (ecs:RunTask) to run the REAL OpenCV pipeline +
the Nova-backed agent loop off the Lambda path. The worker reads INSPECTION_ID /
OBJECT_KEY / EVIDENCE_BUCKET from its container env overrides and persists the
trace/findings/metrics to DynamoDB, which the dashboard then reads.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import uuid

import boto3

from dynamo_store import DynamoStore

_store = DynamoStore()


def _resp(status: int, body: dict) -> dict:
    return {
        "statusCode": status,
        "headers": {"content-type": "application/json", "access-control-allow-origin": "*"},
        "body": json.dumps(body),
    }


def _launch_worker(inspection_id: str, object_key: str) -> str | None:
    """Launch the Fargate Vision_Worker. Returns a failure reason or None on success."""
    cluster = os.environ.get("WORKER_CLUSTER_ARN", "")
    task_def = os.environ.get("WORKER_TASK_DEF_ARN", "")
    container = os.environ.get("WORKER_CONTAINER_NAME", "VisionWorker")
    subnets = [s for s in os.environ.get("WORKER_SUBNETS", "").split(",") if s]
    sg = os.environ.get("WORKER_SECURITY_GROUP", "")
    bucket = os.environ.get("EVIDENCE_BUCKET", "")
    if not (cluster and task_def and subnets and sg and bucket):
        return "worker wiring incomplete (missing cluster/task/subnets/sg/bucket env)"
    ecs = boto3.client("ecs")
    try:
        ecs.run_task(
            cluster=cluster,
            taskDefinition=task_def,
            launchType="FARGATE",
            count=1,
            networkConfiguration={
                "awsvpcConfiguration": {
                    "subnets": subnets,
                    "securityGroups": [sg],
                    "assignPublicIp": "ENABLED",
                }
            },
            overrides={
                "containerOverrides": [
                    {
                        "name": container,
                        "environment": [
                            {"name": "INSPECTION_ID", "value": inspection_id},
                            {"name": "OBJECT_KEY", "value": object_key},
                            {"name": "EVIDENCE_BUCKET", "value": bucket},
                        ],
                    }
                ]
            },
        )
        return None
    except Exception as exc:  # surface honestly; do not fake success
        return f"run_task failed: {exc}"


def handler(event, _context=None):
    try:
        body = json.loads(event.get("body") or "{}")
    except (ValueError, TypeError):
        return _resp(400, {"error": "invalid_json"})

    object_key = str(body.get("object_key", ""))
    if not object_key.startswith("uploads/"):
        return _resp(400, {"error": "invalid_object_key", "reason": "must reference an uploads/ key"})

    iid = uuid.uuid4().hex
    _store.put_inspection({
        "inspection_id": iid,
        "status": "processing",
        "filename": object_key.rsplit("/", 1)[-1],
        "duration_ms": "unavailable",
        "resolution": "unavailable",
        "fps": "unavailable",
        "upload_ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "progress": 0,
        "stop_reason": "unavailable",
        "object_key": object_key,
    })

    # Launch the Fargate worker to actually process the video.
    reason = _launch_worker(iid, object_key)
    if reason is not None:
        # Record an honest failure on the record so the dashboard shows it.
        _store.put_inspection({
            "inspection_id": iid, "status": "failed",
            "filename": object_key.rsplit("/", 1)[-1],
            "duration_ms": "unavailable", "resolution": "unavailable", "fps": "unavailable",
            "upload_ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "progress": 100, "stop_reason": reason, "object_key": object_key,
        })
        return _resp(202, {"inspection_id": iid, "status": "failed", "reason": reason})

    return _resp(202, {"inspection_id": iid, "status": "processing"})
