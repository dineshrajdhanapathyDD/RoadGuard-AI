"""Operational metrics for RoadGuard inspections (task 16.2, req 15.2, 15.3).

Computes agent-operation metrics from the persisted trace/report and records
them into the store's EvaluationMetrics item:

  * action_accuracy - fraction of agent tool invocations that did NOT fail
    (a proxy for "correct next action" over a run; honest and bounded);
  * completion       - 1.0 if the inspection reached complete_inspection, else 0.0;
  * escalation_rate  - escalations / (findings + escalations), or 0.0 when none;
  * latency_ms       - measured per-inspection wall time (caller supplies it);
  * failure_rate     - failed trace steps / total trace steps;
  * aws_cost         - measured AWS cost when available, else the explicit
    UNAVAILABLE sentinel (never fabricated, 13.4).

Detection-quality metrics (precision/recall/F1/IoU) are computed SEPARATELY by
vision.evaluation against labeled ground truth and kept distinct from these
agent-operation metrics (design: "keep detection quality and agent decision
quality as separate measurements"). This module fills only the operational
fields; detection fields stay UNAVAILABLE unless a ground-truth evaluation
supplies them.

Pure and deterministic; no AWS. Persists via the Store.put_metrics seam.
"""

from __future__ import annotations

from typing import Any, Sequence

from vision.result_schema import UNAVAILABLE

# Tool names that, when present as a trace step's tool, represent an escalation
# or a terminal completion (used to derive completion/escalation signals).
_COMPLETE = "complete_inspection"
_ESCALATE = "request_human_review"


def compute_operational_metrics(
    trace: Sequence[dict[str, Any]],
    *,
    finding_count: int,
    escalation_count: int,
    latency_ms: int | str = UNAVAILABLE,
    aws_cost: float | str = UNAVAILABLE,
    precision: float | str = UNAVAILABLE,
    recall: float | str = UNAVAILABLE,
    f1: float | str = UNAVAILABLE,
    iou: float | str = UNAVAILABLE,
) -> dict[str, Any]:
    """Build an EvaluationMetrics-shaped dict from a run's trace + counts (15.2, 15.3).

    ``trace`` entries are the persisted TraceEntry dicts. A step is counted as a
    failure when its ``decision`` records a tool failure (the loop writes
    "<tool> failed (...)" into ``decision`` on a failed tool, see
    agent.loop._resolve_decision). ``latency_ms``/``aws_cost`` are supplied by
    the caller (measured) or left as the explicit UNAVAILABLE sentinel.
    Detection metrics pass through (UNAVAILABLE unless a GT evaluation supplies
    them), keeping detection vs. agent-operation quality separate.
    """
    total = len(trace)
    failed_steps = sum(1 for e in trace if "failed (" in str(e.get("decision", "")))
    completed = any(e.get("tool") == _COMPLETE for e in trace)

    action_accuracy = (total - failed_steps) / total if total > 0 else UNAVAILABLE
    failure_rate = failed_steps / total if total > 0 else UNAVAILABLE
    completion = 1.0 if completed else 0.0
    denom = finding_count + escalation_count
    escalation_rate = (escalation_count / denom) if denom > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "action_accuracy": action_accuracy,
        "completion": completion,
        "escalation_rate": escalation_rate,
        "latency_ms": latency_ms,
        "failure_rate": failure_rate,
        "aws_cost": aws_cost,
    }


def record_metrics(store, inspection_id: str, metrics: dict[str, Any], **put_kwargs) -> None:
    """Persist operational metrics via the Store seam (put_metrics)."""
    store.put_metrics(inspection_id, metrics, **put_kwargs)


def _self_check() -> None:
    """Assert operational-metric derivations are correct and bounded."""
    # A trace: 2 successful analyses, 1 failed tool, then complete.
    trace = [
        {"tool": "analyze_frame", "decision": "analyzed frame; evaluating detections"},
        {"tool": "analyze_region", "decision": "investigated region at higher fidelity"},
        {"tool": "save_inspection_finding", "decision": "save_inspection_finding failed (rejected: low confidence)"},
        {"tool": "complete_inspection", "decision": "completed the inspection (complete_inspection)"},
    ]
    m = compute_operational_metrics(trace, finding_count=0, escalation_count=1, latency_ms=1840)
    assert m["completion"] == 1.0
    assert abs(m["failure_rate"] - 0.25) < 1e-9  # 1 of 4 steps failed
    assert abs(m["action_accuracy"] - 0.75) < 1e-9
    assert m["escalation_rate"] == 1.0  # 1 escalation, 0 findings
    assert m["latency_ms"] == 1840
    assert m["aws_cost"] == UNAVAILABLE  # not measured offline -> explicit sentinel
    # detection fields stay separate/unavailable unless supplied
    assert m["precision"] == UNAVAILABLE and m["iou"] == UNAVAILABLE

    # empty trace -> action_accuracy/failure_rate unavailable (nothing measured)
    m0 = compute_operational_metrics([], finding_count=0, escalation_count=0)
    assert m0["action_accuracy"] == UNAVAILABLE and m0["failure_rate"] == UNAVAILABLE
    assert m0["completion"] == 0.0 and m0["escalation_rate"] == 0.0

    # findings present, no escalation -> escalation_rate 0.0 and bounded
    m1 = compute_operational_metrics(trace[:2], finding_count=2, escalation_count=0)
    assert m1["escalation_rate"] == 0.0
    for k in ("completion", "escalation_rate"):
        assert 0.0 <= m1[k] <= 1.0

    # In-memory store round-trip via the seam.
    from agent.store import InMemoryStore, InspectionRecord
    s = InMemoryStore()
    s.put_inspection(InspectionRecord(
        inspection_id="opm", status="completed", filename="x.mp4",
        duration_ms=UNAVAILABLE, resolution=UNAVAILABLE, fps=UNAVAILABLE,
        upload_ts="2026-01-01T00:00:00Z", progress=100, stop_reason="complete_inspection"))
    record_metrics(s, "opm", m)
    back = s.get_inspection_bundle("opm")["metrics"]
    assert back != UNAVAILABLE and back["completion"] == 1.0


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    _self_check()
    print("[agent.metrics] OK: operational metrics derived, bounded, detection kept separate, persisted.")
