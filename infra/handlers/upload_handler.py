"""upload_handler: issue a presigned PUT and validate an uploaded video (12.3/12.4).

POST /uploads  { "filename": str, "content_type": str, "size": int }
  -> 200 { "upload_url", "object_key", "expires_in" }   (presigned PUT, 1.1, 18.4)
  -> 400 { "error", "reason" }                            (validation failure, 1.2/1.3)

The presigned URL is scoped to a SINGLE object key under the private bucket's
``uploads/`` prefix (18.4): the client can PUT exactly that key and nothing else.
Container/codec/duration/size are validated against configured ranges; on failure
the reason is recorded and NO inspection is started (1.3). Full codec/duration
probing happens in the Vision_Worker on decode; here we enforce the cheap,
pre-upload checks (extension/content-type/size).
"""

from __future__ import annotations

import json
import os
import re
import uuid

import boto3
from botocore.config import Config

_BUCKET = os.environ.get("EVIDENCE_BUCKET", "")
_MAX_BYTES = 500 * 1024 * 1024  # 500 MB cap (configured range, 1.2)
_ALLOWED_CT = {"video/mp4", "video/quicktime", "video/x-msvideo"}
_ALLOWED_EXT = {"mp4", "mov", "avi"}
_SAFE = re.compile(r"[^A-Za-z0-9._-]")


def _resp(status: int, body: dict) -> dict:
    return {
        "statusCode": status,
        "headers": {"content-type": "application/json", "access-control-allow-origin": "*"},
        "body": json.dumps(body),
    }


def _validate(filename: str, content_type: str, size: int) -> str | None:
    """Return a specific failure reason, or None when the request is acceptable (1.2)."""
    if not filename:
        return "filename is required"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in _ALLOWED_EXT:
        return f"unsupported extension {ext!r}; accepts {sorted(_ALLOWED_EXT)}"
    if content_type not in _ALLOWED_CT:
        return f"unsupported content_type {content_type!r}; accepts {sorted(_ALLOWED_CT)}"
    if not isinstance(size, int) or size <= 0:
        return f"size must be a positive integer, got {size!r}"
    if size > _MAX_BYTES:
        return f"size {size} exceeds the {_MAX_BYTES}-byte limit"
    return None


def handler(event, _context=None):
    try:
        body = json.loads(event.get("body") or "{}")
    except (ValueError, TypeError):
        return _resp(400, {"error": "invalid_json", "reason": "request body must be JSON"})

    filename = str(body.get("filename", ""))
    content_type = str(body.get("content_type", ""))
    size = body.get("size", 0)

    reason = _validate(filename, content_type, size)
    if reason is not None:
        # Record the rejection reason; do NOT start an inspection (1.3).
        return _resp(400, {"error": "validation_failed", "reason": reason})

    safe_name = _SAFE.sub("_", filename)
    object_key = f"uploads/{uuid.uuid4().hex}/{safe_name}"
    s3 = boto3.client("s3", config=Config(signature_version="s3v4"))
    expires_in = 900
    upload_url = s3.generate_presigned_url(
        "put_object",
        Params={"Bucket": _BUCKET, "Key": object_key, "ContentType": content_type},
        ExpiresIn=expires_in,
    )
    return _resp(200, {"upload_url": upload_url, "object_key": object_key, "expires_in": expires_in})
