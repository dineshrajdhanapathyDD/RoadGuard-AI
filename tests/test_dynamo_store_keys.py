"""Regression tests for DynamoStore index-key derivation.

These tests pin the fix for the stale-GSI1 bug: when a META record is read back
from DynamoDB and re-saved on completion, it carries the *old* ``GSI1PK`` (e.g.
``STATUS#processing``). ``put_inspection`` must regenerate the derived keys from
the current ``status`` so the GSI1 status index stays in sync with META, instead
of letting the stale key from the spread win.

No DynamoDB / moto needed: we stub ``_table()`` to capture the written item.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

# Make the Lambda handler package importable the same way the deployed asset does.
_HANDLERS = Path(__file__).resolve().parents[1] / "infra" / "handlers"
if str(_HANDLERS) not in sys.path:
    sys.path.insert(0, str(_HANDLERS))

dynamo_store = importlib.import_module("dynamo_store")


class _FakeTable:
    """Captures the last item passed to put_item; records every write."""

    def __init__(self) -> None:
        self.items: list[dict] = []

    def put_item(self, Item):  # noqa: N803 (boto3 kwarg name)
        self.items.append(Item)
        return {}


@pytest.fixture()
def fake_table(monkeypatch):
    t = _FakeTable()
    monkeypatch.setattr(dynamo_store, "_table", lambda: t)
    return t


def test_put_inspection_regenerates_gsi1pk_from_current_status(fake_table):
    """A stale GSI1PK carried in the record must NOT survive the write."""
    store = dynamo_store.DynamoStore()
    # Simulate completion.py: a bundle read back from Dynamo still says processing.
    record = {
        "inspection_id": "insp-123",
        "status": "completed",            # current, authoritative status
        "upload_ts": "2026-01-01T00:00:00Z",
        "GSI1PK": "STATUS#processing",    # STALE key carried from the read-back
        "GSI1SK": "2026-01-01T00:00:00Z",
        "PK": "INSPECTION#insp-123",
        "SK": "META",
    }
    store.put_inspection(record)

    written = fake_table.items[-1]
    assert written["GSI1PK"] == "STATUS#completed", (
        "derived GSI1PK must reflect current status, not the stale carried value"
    )
    assert written["SK"] == "META"
    assert written["PK"] == "INSPECTION#insp-123"


def test_put_inspection_defaults_status_to_pending(fake_table):
    store = dynamo_store.DynamoStore()
    store.put_inspection({"inspection_id": "insp-x", "upload_ts": "t"})
    written = fake_table.items[-1]
    assert written["GSI1PK"] == "STATUS#pending"


def test_enqueue_review_regenerates_gsi2pk(fake_table):
    store = dynamo_store.DynamoStore()
    store.enqueue_review(
        "insp-9",
        {"status": "resolved", "GSI2PK": "REVIEW_STATUS#open"},  # stale GSI2PK
        seq=1,
    )
    written = fake_table.items[-1]
    assert written["GSI2PK"] == "REVIEW_STATUS#resolved"
    assert written["SK"] == "REVIEW#000001"


def test_put_metrics_keeps_fixed_gsi3pk(fake_table):
    store = dynamo_store.DynamoStore()
    store.put_metrics("insp-7", {"GSI3PK": "WRONG", "f1": "unavailable"}, upload_ts="123")
    written = fake_table.items[-1]
    assert written["GSI3PK"] == "METRICS"
    assert written["GSI3SK"] == "123"
    assert written["SK"] == "METRICS"
