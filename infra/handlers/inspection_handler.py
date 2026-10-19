"""inspection_handler: start an inspection and report status (12.4).

POST /inspections  { "object_key": str }
  -> 202 { "inspection_id", "status": "processing" }

Validates that an object_key was provided, creates the inspection META record,
and (in a full deploy) launches the Fargate Vision_Worker task to run the real
OpenCV pipeline + agent. Here it records the record and returns the id; the
Fargate RunTask wiring is environment-specific and added at deploy time. Keeps
agent/video processing OFF the Lambda path (design: Lambda is short tasks only).
"""

from __future__ import annotations

import datetime as _dt
import json
import uuid

from dynamo_store import DynamoStore

_store = DynamoStore()


def _resp(status: int, body: dict) -> dict:
    return {
        "statusCode": status,
        "headers": {"content-type": "application/json", "access-control-allow-origin": "*"},
        "body": json.dumps(body),
    }


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
    # A full deploy launches the Fargate worker here (ecs:RunTask) to process the
    # video off the Lambda path. Left to deploy-time wiring (needs cluster/subnets).
    return _resp(202, {"inspection_id": iid, "status": "processing"})
