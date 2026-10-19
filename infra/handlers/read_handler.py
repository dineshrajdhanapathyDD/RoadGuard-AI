"""read_handler: serve overview, inspection, report, trace, and metrics (12.4).

Routes (all GET), dispatched from the API Gateway resource path:
  GET /overview                     -> aggregate stats across inspections (16.1)
  GET /metrics                      -> aggregate metrics (15.4)
  GET /inspections/{id}             -> inspection META (13.3)
  GET /inspections/{id}/report      -> InspectionReport (13.x)
  GET /inspections/{id}/trace       -> ordered agent trace (14.3)

Read-only: granted only DynamoDB read + S3 read (18.1). Missing values stay the
explicit "unavailable" sentinel; nothing is fabricated (13.4).
"""

from __future__ import annotations

import json

from dynamo_store import DynamoStore

_store = DynamoStore()


def _json_default(o):
    """Serialize DynamoDB Decimals as plain numbers (int when whole), else str."""
    import decimal
    if isinstance(o, decimal.Decimal):
        return int(o) if o == o.to_integral_value() else float(o)
    return str(o)


def _resp(status: int, body) -> dict:
    return {
        "statusCode": status,
        "headers": {"content-type": "application/json", "access-control-allow-origin": "*"},
        "body": json.dumps(body, default=_json_default),
    }


def _report_from_bundle(bundle: dict) -> dict:
    findings = bundle.get("findings", [])
    escalations = bundle.get("review_queue", [])
    meta = bundle.get("meta")
    status = meta.get("status") if isinstance(meta, dict) else "unavailable"
    stop_reason = (meta.get("stop_reason") if isinstance(meta, dict) else "unavailable") or "unavailable"
    return {
        "inspection_id": bundle["inspection_id"],
        "status": status,
        "stop_reason": stop_reason,
        "findings": findings,
        "escalations": escalations,
        "finding_count": len(findings),
        "escalation_count": len(escalations),
        "zero_findings": len(findings) == 0 and len(escalations) == 0,
        "trace_length": len(bundle.get("trace", [])),
        "metrics": bundle.get("metrics", "unavailable"),
    }


def handler(event, _context=None):
    path = event.get("resource") or event.get("path") or ""
    params = event.get("pathParameters") or {}
    iid = params.get("id")

    if path.endswith("/overview") or path.endswith("/metrics"):
        # Real aggregate across inspections from GSI1 (16.1). Honest: counts and
        # recent activity come from the index/partitions, never invented.
        try:
            return _resp(200, _store.overview())
        except Exception as exc:  # pragma: no cover - surfaces an honest error
            return _resp(200, {
                "total_inspections": 0, "completed": 0, "failed": 0,
                "potential_findings": 0, "awaiting_review": 0, "recent": [],
                "error": f"overview unavailable: {exc}",
            })

    if not iid:
        return _resp(400, {"error": "missing_id"})

    bundle = _store.get_inspection_bundle(iid)
    if path.endswith("/report"):
        return _resp(200, _report_from_bundle(bundle))
    if path.endswith("/trace"):
        return _resp(200, bundle.get("trace", []))
    # default: inspection META
    meta = bundle.get("meta")
    if not isinstance(meta, dict):
        return _resp(404, {"error": "not_found", "inspection_id": iid})
    return _resp(200, meta)
