"""Persistence interface for the RoadGuard agent layer.

This module defines the :class:`Store` ``Protocol`` -- the single seam through
which the agent loop, the seven tools, and the read APIs persist and retrieve
inspection state -- together with the record types it stores and a
credential-free in-memory implementation, :class:`InMemoryStore`.

See .kiro/specs/roadguard-ai/design.md:
* "Persistence Interface" for the :class:`Store` Protocol signatures.
* "Data Models" -> "Entities" and "Single-Table Design" for the record shapes
  (Inspection META, Finding, Trace entry, Review item, Metrics) and the
  full-bundle query pattern (``PK = INSPECTION#<id>`` returns meta + findings +
  trace + review + metrics).

Requirements: 14.1 (ordered agent trace), 19.4 (automated tests run against the
in-memory fake with no AWS credentials).

Design intent (drop-in DynamoDB later, task 12.2):
    Everything the agent touches goes through the :class:`Store` Protocol, so
    the DynamoDB-backed store added later is a drop-in replacement -- same
    method names, same record types, same :class:`InspectionBundle` return.
    The single-table query patterns in the design map one-to-one onto these
    methods:

        put_inspection        -> PutItem  SK=META
        append_trace          -> PutItem  SK=TRACE#<zero-padded ts>  (ordered)
        put_finding           -> PutItem  SK=FINDING#<seq>
        enqueue_review        -> PutItem  SK=REVIEW#<seq>
        put_metrics           -> PutItem  SK=METRICS
        get_inspection_bundle -> Query    PK=INSPECTION#<id>  (whole partition)

Trace ordering is the load-bearing invariant: the critical-criterion proof
depends on the persisted ``before -> after -> decision`` sequence staying in
append order. :class:`InMemoryStore` preserves append order and additionally
returns the trace sorted by :class:`~agent.trace.TraceEntry` ``ts`` so ordering
holds even if entries are appended out of order.

Any value that can be missing is the explicit
:data:`~vision.result_schema.UNAVAILABLE` sentinel rather than a fabricated
value (R13.4 style). This module performs no network or AWS work and is
importable with no AWS credentials.
"""

from __future__ import annotations

import copy
from typing import Literal, Protocol, TypedDict, runtime_checkable

from vision.result_schema import UNAVAILABLE
from agent.trace import TraceEntry

# --------------------------------------------------------------------------- #
# Closed vocabularies
# --------------------------------------------------------------------------- #

#: Lifecycle status of an inspection record (design META ``status``).
InspectionStatus = Literal["pending", "processing", "completed", "failed"]

#: Open/resolved status of a human-review queue item (design REVIEW ``status``).
ReviewStatus = Literal["open", "resolved"]


# --------------------------------------------------------------------------- #
# Record types (match design "Single-Table Design" attributes)
# --------------------------------------------------------------------------- #


class InspectionRecord(TypedDict):
    """The per-inspection META record -- one per uploaded video (1.4, 5, 13.3).

    Mirrors the design's Inspection row attributes. ``resolution`` /
    ``stop_reason`` and other values that may not yet be known are the explicit
    :data:`UNAVAILABLE` sentinel rather than fabricated (13.4). ``inspection_id``
    is carried on the record (it is the partition key in the DynamoDB design).
    """

    inspection_id: str
    status: InspectionStatus
    filename: str
    duration_ms: int | str  # int, or UNAVAILABLE if not yet probed
    resolution: str  # "WxH" in pixels, or UNAVAILABLE
    fps: float | str  # float, or UNAVAILABLE
    upload_ts: str  # ISO-8601 upload timestamp
    progress: int  # 0-100 (1.5)
    stop_reason: str  # loop stop reason, or UNAVAILABLE until completed


class InspectionFinding(TypedDict):
    """A confirmed hazard finding referencing real evidence (10.3, 13.1).

    Mirrors the design's Finding row. ``evidence_ref`` is the S3 key of the
    annotated frame (never fabricated); ``method`` records which detection path
    produced it (4.3). A finding is only ever created from a successful
    :class:`~vision.result_schema.AnalysisResult`, so this type carries no
    failure state.
    """

    hazard_type: str
    evidence_ref: str
    confidence: float
    source_timestamp_ms: int
    method: str


class HumanReviewItem(TypedDict):
    """An escalated observation enqueued for human review (11.2, 13.2).

    Mirrors the design's Review row. ``status`` starts ``"open"`` and becomes
    ``"resolved"`` when a reviewer dispositions it.
    """

    observation_ref: str
    evidence_ref: str
    reason: str
    status: ReviewStatus


class EvaluationMetrics(TypedDict):
    """Per-inspection evaluation metrics (15).

    Mirrors the design's Metrics row. Any metric not computed for a given
    inspection is the explicit :data:`UNAVAILABLE` sentinel rather than a
    fabricated zero (13.4) -- callers distinguish "measured 0.0" from
    "not measured".
    """

    precision: float | str
    recall: float | str
    f1: float | str
    iou: float | str
    action_accuracy: float | str
    completion: float | str
    escalation_rate: float | str
    latency_ms: int | str
    failure_rate: float | str
    aws_cost: float | str


class InspectionBundle(TypedDict):
    """The whole-inspection aggregate returned by :meth:`Store.get_inspection_bundle`.

    This is the single-query bundle from the design (``PK = INSPECTION#<id>``
    returns meta + findings + trace + review + metrics in one query), serving
    the report (13.3) and trace (14.3) read patterns.

    * ``meta`` -- the :class:`InspectionRecord`, or :data:`UNAVAILABLE` if no
      inspection has been stored under this id.
    * ``trace`` -- the ordered list of :class:`~agent.trace.TraceEntry` (14.1).
    * ``metrics`` -- the :class:`EvaluationMetrics`, or :data:`UNAVAILABLE` if
      none has been recorded yet.
    """

    inspection_id: str
    meta: InspectionRecord | str
    findings: list[InspectionFinding]
    trace: list[TraceEntry]
    review_queue: list[HumanReviewItem]
    metrics: EvaluationMetrics | str


# --------------------------------------------------------------------------- #
# Persistence Protocol (design "Persistence Interface")
# --------------------------------------------------------------------------- #


@runtime_checkable
class Store(Protocol):
    """Durable-store seam for the agent layer (design "Persistence Interface").

    The agent loop, the seven tools, and the read APIs depend only on this
    Protocol, so the in-memory fake (dev/test, credential-free) and the
    DynamoDB-backed store (task 12.2) are interchangeable. Implementations must
    preserve trace append order (14.1): entries read back from
    :meth:`get_inspection_bundle` are ordered by their :class:`TraceEntry`
    ``ts`` / append sequence.
    """

    def put_inspection(self, record: InspectionRecord) -> None:
        """Create or replace the inspection META record."""
        ...

    def append_trace(self, inspection_id: str, entry: TraceEntry) -> None:
        """Append one trace entry, preserving order (14.1)."""
        ...

    def put_finding(self, inspection_id: str, finding: InspectionFinding) -> None:
        """Append a confirmed finding to the inspection."""
        ...

    def enqueue_review(self, inspection_id: str, item: HumanReviewItem) -> None:
        """Enqueue an escalated observation for human review."""
        ...

    def put_metrics(self, inspection_id: str, metrics: EvaluationMetrics) -> None:
        """Create or replace the inspection's evaluation metrics."""
        ...

    def get_inspection_bundle(self, inspection_id: str) -> InspectionBundle:
        """Return the whole-inspection aggregate (meta + findings + trace + review + metrics)."""
        ...


# --------------------------------------------------------------------------- #
# In-memory fake (credential-free default, requirement 19.4)
# --------------------------------------------------------------------------- #


class InMemoryStore:
    """In-memory :class:`Store` for credential-free dev/test runs (19.4).

    Backs the full agent loop and the automated tests with no AWS. It stores
    inspections, appends **ordered** trace entries, findings, review items, and
    metrics, and returns an :class:`InspectionBundle` for an id.

    Isolation: records are deep-copied on write and on read, so a caller that
    later mutates the object it passed in (or the bundle it got back) cannot
    corrupt stored state -- the same defensive copy the DynamoDB marshalling
    boundary gives for free. Partitions are created lazily on first write.
    """

    def __init__(self) -> None:
        self._meta: dict[str, InspectionRecord] = {}
        self._findings: dict[str, list[InspectionFinding]] = {}
        self._trace: dict[str, list[TraceEntry]] = {}
        self._review: dict[str, list[HumanReviewItem]] = {}
        self._metrics: dict[str, EvaluationMetrics] = {}

    # -- writes ------------------------------------------------------------- #

    def put_inspection(self, record: InspectionRecord) -> None:
        """Create or replace the META record, keyed by ``record['inspection_id']``."""
        inspection_id = record["inspection_id"]
        if not inspection_id:
            raise ValueError("InspectionRecord requires a non-empty inspection_id")
        self._meta[inspection_id] = copy.deepcopy(record)

    def append_trace(self, inspection_id: str, entry: TraceEntry) -> None:
        """Append ``entry`` to the inspection's trace, preserving append order (14.1)."""
        self._trace.setdefault(inspection_id, []).append(copy.deepcopy(entry))

    def put_finding(self, inspection_id: str, finding: InspectionFinding) -> None:
        """Append a confirmed finding to the inspection."""
        self._findings.setdefault(inspection_id, []).append(copy.deepcopy(finding))

    def enqueue_review(self, inspection_id: str, item: HumanReviewItem) -> None:
        """Enqueue an escalated observation for human review."""
        self._review.setdefault(inspection_id, []).append(copy.deepcopy(item))

    def put_metrics(self, inspection_id: str, metrics: EvaluationMetrics) -> None:
        """Create or replace the inspection's evaluation metrics."""
        self._metrics[inspection_id] = copy.deepcopy(metrics)

    # -- reads -------------------------------------------------------------- #

    def get_inspection_bundle(self, inspection_id: str) -> InspectionBundle:
        """Return the whole-inspection aggregate for ``inspection_id`` (13.3, 14.3).

        The trace is returned sorted by :class:`TraceEntry` ``ts`` (then append
        order) so the ``before -> after -> decision`` sequence is always ordered
        even if entries were appended out of order (14.1). Missing meta/metrics
        are the explicit :data:`UNAVAILABLE` sentinel, not a fabricated record.
        All returned data is deep-copied, so mutating the bundle cannot corrupt
        stored state.
        """
        trace = sorted(self._trace.get(inspection_id, []), key=lambda e: e["ts"])
        meta = self._meta.get(inspection_id)
        metrics = self._metrics.get(inspection_id)
        return InspectionBundle(
            inspection_id=inspection_id,
            meta=copy.deepcopy(meta) if meta is not None else UNAVAILABLE,
            findings=copy.deepcopy(self._findings.get(inspection_id, [])),
            trace=copy.deepcopy(trace),
            review_queue=copy.deepcopy(self._review.get(inspection_id, [])),
            metrics=copy.deepcopy(metrics) if metrics is not None else UNAVAILABLE,
        )


# --------------------------------------------------------------------------- #
# Self-validation (import-time smoke check)
# --------------------------------------------------------------------------- #


def _self_check() -> None:
    """Exercise the store end-to-end and assert trace order is preserved (14.1, 19.4).

    Not the property-test suite: a lightweight guard that the fake satisfies the
    Protocol and the ordering invariant the critical-criterion proof depends on.
    """
    from vision.result_schema import pixel_measurement, success_result, detection
    from agent.trace import trace_entry

    store: Store = InMemoryStore()
    assert isinstance(store, Store)  # runtime_checkable Protocol conformance

    inspection_id = "insp-001"

    store.put_inspection(
        InspectionRecord(
            inspection_id=inspection_id,
            status="processing",
            filename="road.mp4",
            duration_ms=12000,
            resolution="1920x1080",
            fps=30.0,
            upload_ts="2026-01-01T00:00:00Z",
            progress=10,
            stop_reason=UNAVAILABLE,
        )
    )

    # Two ordered trace entries: a low-confidence before, then a corroborating after.
    m = pixel_measurement(10, 20)
    low = success_result(
        method="classical",
        frame_ref="frame_42",
        source_timestamp_ms=1000,
        detections=[detection("pothole", (1, 2, 3, 4), 0.42, m)],
    )
    high = success_result(
        method="model",
        frame_ref="frame_42",
        source_timestamp_ms=1000,
        detections=[detection("pothole", (1, 2, 3, 4), 0.88, m)],
    )
    store.append_trace(
        inspection_id,
        trace_entry(
            perception="frame_42",
            reasoning="low confidence near region R",
            tool="analyze_frame",
            after_output=low,
            decision="analyze_region",
            ts=0,
        ),
    )
    store.append_trace(
        inspection_id,
        trace_entry(
            perception="frame_42 region R",
            reasoning="region corroborates the hazard at higher confidence",
            tool="analyze_region",
            before_output=low,
            after_output=high,
            decision="save_inspection_finding",
            ts=1,
        ),
    )

    store.put_finding(
        inspection_id,
        InspectionFinding(
            hazard_type="pothole",
            evidence_ref="evidence/insp-001/frame_42.png",
            confidence=0.88,
            source_timestamp_ms=1000,
            method="model",
        ),
    )
    store.enqueue_review(
        inspection_id,
        HumanReviewItem(
            observation_ref="frame_99",
            evidence_ref="evidence/insp-001/frame_99.png",
            reason="poor_input_quality",
            status="open",
        ),
    )
    store.put_metrics(
        inspection_id,
        EvaluationMetrics(
            precision=1.0,
            recall=1.0,
            f1=1.0,
            iou=0.75,
            action_accuracy=1.0,
            completion=1.0,
            escalation_rate=0.0,
            latency_ms=4200,
            failure_rate=0.0,
            aws_cost=UNAVAILABLE,
        ),
    )

    bundle = store.get_inspection_bundle(inspection_id)
    assert bundle["inspection_id"] == inspection_id
    assert bundle["meta"] != UNAVAILABLE and bundle["meta"]["status"] == "processing"
    assert len(bundle["findings"]) == 1
    assert len(bundle["review_queue"]) == 1
    assert bundle["metrics"] != UNAVAILABLE and bundle["metrics"]["aws_cost"] == UNAVAILABLE

    # The load-bearing invariant: trace order is preserved, before->after->decision.
    assert [e["ts"] for e in bundle["trace"]] == [0, 1]
    assert bundle["trace"][0]["decision"] == "analyze_region"
    assert bundle["trace"][1]["before_output"]["detections"][0]["confidence"] == 0.42
    assert bundle["trace"][1]["after_output"]["detections"][0]["confidence"] == 0.88

    # Unknown id yields an empty, honest bundle (explicit UNAVAILABLE meta/metrics).
    empty = store.get_inspection_bundle("does-not-exist")
    assert empty["meta"] == UNAVAILABLE
    assert empty["metrics"] == UNAVAILABLE
    assert empty["trace"] == []


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    _self_check()
    print("[store] OK: Store protocol + InMemoryStore; trace order preserved.")
