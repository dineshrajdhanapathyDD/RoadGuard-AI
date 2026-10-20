"""DynamoDB-backed Store for the RoadGuard backend (task 12.2).

Implements the same logical operations as ``agent.store.Store`` against the
single-table design (design "Single-Table Design"), mapping each entity to a
PK/SK and the GSIs. Trace entries use a zero-padded SK so lexical order equals
iteration order (14.3). All writes/reads go through a bounded-retry wrapper with
a timeout so a transient AWS error is retried a few times and then surfaced
honestly (18.7).

This module is deployed inside the Lambda asset (``handlers/``) and imported by
the handlers. It uses boto3 (available in the Lambda Python runtime); it is not
imported by the offline Mock_Runtime backbone, which keeps using
``agent.store.InMemoryStore``.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, TypeVar

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

_T = TypeVar("_T")

# Bounded retries with a timeout on every AWS call (18.7). boto3's standard mode
# adds adaptive retries; we also cap wall time with a small application-level loop.
_BOTO_CONFIG = Config(retries={"max_attempts": 3, "mode": "standard"}, connect_timeout=3, read_timeout=5)

_TABLE_NAME = os.environ.get("ROADGUARD_TABLE", "RoadGuard-dev")

#: Transient error codes worth a bounded retry; everything else surfaces at once.
_RETRYABLE = {
    "ProvisionedThroughputExceededException",
    "ThrottlingException",
    "RequestLimitExceeded",
    "InternalServerError",
    "ServiceUnavailable",
}


def _with_retries(op: Callable[[], _T], *, attempts: int = 3, base_delay: float = 0.1) -> _T:
    """Run ``op`` with bounded retries + backoff; re-raise after the cap (18.7)."""
    last: Exception | None = None
    for i in range(attempts):
        try:
            return op()
        except ClientError as exc:  # pragma: no cover - exercised via fault injection
            code = exc.response.get("Error", {}).get("Code", "")
            last = exc
            if code not in _RETRYABLE or i == attempts - 1:
                raise
            time.sleep(base_delay * (2 ** i))
    assert last is not None
    raise last


def _table():
    return boto3.resource("dynamodb", config=_BOTO_CONFIG).Table(_TABLE_NAME)


def _to_dynamo(value):
    """Recursively convert a value into DynamoDB-safe types.

    DynamoDB's resource API rejects Python float (requires Decimal). Trace
    entries carry nested AnalysisResult dicts with float confidences, so we
    walk the whole structure: float -> Decimal (via str), dict/list recursed,
    everything else passed through. Only the wire type changes.
    """
    import decimal
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return decimal.Decimal(str(value))
    if isinstance(value, dict):
        return {k: _to_dynamo(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_dynamo(v) for v in value]
    return value


class DynamoStore:
    """Single-table Store over DynamoDB (mirrors agent.store.Store semantics)."""

    def put_inspection(self, record: dict[str, Any]) -> None:
        iid = record["inspection_id"]
        item = {
            **record,
            # Derived keys come AFTER the spread so a stale GSI1PK carried in
            # ``record`` (e.g. re-saving a bundle read back from DynamoDB on
            # completion) cannot clobber the value computed from the current
            # status. This keeps the GSI1 status index in sync with META.
            "PK": f"INSPECTION#{iid}", "SK": "META",
            "GSI1PK": f"STATUS#{record.get('status', 'pending')}",
            "GSI1SK": str(record.get("upload_ts", "")),
        }
        _with_retries(lambda: _table().put_item(Item=_to_dynamo(item)))

    def append_trace(self, inspection_id: str, entry: dict[str, Any]) -> None:
        ts = int(entry.get("ts", 0))
        item = {
            "PK": f"INSPECTION#{inspection_id}",
            "SK": f"TRACE#{ts:08d}",  # zero-padded -> lexical == iteration order (14.3)
            **entry,
        }
        _with_retries(lambda: _table().put_item(Item=_to_dynamo(item)))

    def put_finding(self, inspection_id: str, finding: dict[str, Any], seq: int) -> None:
        item = {"PK": f"INSPECTION#{inspection_id}", "SK": f"FINDING#{seq:06d}", **finding}
        _with_retries(lambda: _table().put_item(Item=_to_dynamo(item)))

    def enqueue_review(self, inspection_id: str, item_data: dict[str, Any], seq: int) -> None:
        item = {
            **item_data,
            "PK": f"INSPECTION#{inspection_id}", "SK": f"REVIEW#{seq:06d}",
            "GSI2PK": f"REVIEW_STATUS#{item_data.get('status', 'open')}",
            "GSI2SK": f"{inspection_id}#{seq:06d}",
        }
        _with_retries(lambda: _table().put_item(Item=_to_dynamo(item)))

    def put_metrics(self, inspection_id: str, metrics: dict[str, Any], upload_ts: str = "") -> None:
        item = {
            **metrics,
            "PK": f"INSPECTION#{inspection_id}", "SK": "METRICS",
            "GSI3PK": "METRICS", "GSI3SK": upload_ts,
        }
        _with_retries(lambda: _table().put_item(Item=_to_dynamo(item)))

    def overview(self, limit: int = 25) -> dict[str, Any]:
        """Aggregate stats across inspections for the dashboard overview (16.1).

        Queries GSI1 (``GSI1PK = STATUS#<status>``) for each known status and
        assembles real counts + a recent-activity list. Honest: counts come from
        the index, never invented; findings are tallied from each recent
        inspection's partition only (bounded) so the call stays cheap.
        """
        from boto3.dynamodb.conditions import Key

        t = _table()
        statuses = ("pending", "processing", "completed", "failed")
        records: list[dict[str, Any]] = []
        for st in statuses:
            resp = _with_retries(
                lambda st=st: t.query(
                    IndexName="GSI1",
                    KeyConditionExpression=Key("GSI1PK").eq(f"STATUS#{st}"),
                    ScanIndexForward=False,
                )
            )
            records.extend(resp.get("Items", []))

        total = len(records)
        completed = sum(1 for r in records if r.get("status") == "completed")
        failed = sum(1 for r in records if r.get("status") == "failed")

        # recent activity: newest first by upload_ts, capped at `limit`.
        recent = sorted(records, key=lambda r: str(r.get("upload_ts", "")), reverse=True)[:limit]

        # potential findings + awaiting review: tally from each recent inspection
        # partition (bounded by `limit`), so the number is real, not invented.
        potential_findings = 0
        awaiting_review = 0
        for r in recent:
            iid = r.get("inspection_id")
            if not iid:
                continue
            part = _with_retries(
                lambda iid=iid: t.query(
                    KeyConditionExpression=Key("PK").eq(f"INSPECTION#{iid}")
                )
            ).get("Items", [])
            potential_findings += sum(1 for i in part if str(i["SK"]).startswith("FINDING#"))
            awaiting_review += sum(
                1 for i in part
                if str(i["SK"]).startswith("REVIEW#") and i.get("status") == "open"
            )

        return {
            "total_inspections": total,
            "completed": completed,
            "failed": failed,
            "potential_findings": potential_findings,
            "awaiting_review": awaiting_review,
            "recent": recent,
        }

    def get_inspection_bundle(self, inspection_id: str) -> dict[str, Any]:
        """Query the whole partition and assemble the bundle (13.3, 14.3)."""
        resp = _with_retries(
            lambda: _table().query(
                KeyConditionExpression=boto3.dynamodb.conditions.Key("PK").eq(
                    f"INSPECTION#{inspection_id}"
                )
            )
        )
        items = resp.get("Items", [])
        meta = next((i for i in items if i["SK"] == "META"), "unavailable")
        metrics = next((i for i in items if i["SK"] == "METRICS"), "unavailable")
        trace = sorted(
            [i for i in items if str(i["SK"]).startswith("TRACE#")],
            key=lambda i: i["SK"],
        )
        findings = [i for i in items if str(i["SK"]).startswith("FINDING#")]
        review = [i for i in items if str(i["SK"]).startswith("REVIEW#")]
        return {
            "inspection_id": inspection_id,
            "meta": meta,
            "findings": findings,
            "trace": trace,
            "review_queue": review,
            "metrics": metrics,
        }
