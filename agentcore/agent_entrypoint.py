"""AgentCore Runtime entrypoint for RoadGuard (Milestone 9, Option A).

This is the SMALL agent-reasoning service hosted on Amazon Bedrock AgentCore
Runtime. It hosts the DECISION brain only -- it does NOT bundle OpenCV. The
heavy computer-vision work stays on the Fargate Vision_Worker (deployed in
Milestone 6); this entrypoint reasons over vision results and returns the next
agent decision. Keeping OpenCV out of this image keeps the AgentCore container
lightweight (the clean agent/vision separation of Option A).

Contract (AgentCore Runtime HTTP protocol): receives a JSON payload on
POST /invocations and returns JSON. The payload is a RoadGuard decision request:

    {
      "action": "decide",
      "state": { "iteration": int, "max_iterations": int, ... },
      "perception": { ...a real AnalysisResult from the Vision_Worker... }
    }

and the response is the chosen AgentDecision (tool_name + arguments + reasoning),
constrained to the seven RoadGuard tools. The agent reasons over the REAL
perception the Vision_Worker produced (OpenCV output), so the critical criterion
-- real vision output drives the next action -- holds across the managed runtime
exactly as it does offline.

Runtime selection: uses the deterministic MockDecider by default (so the service
is reproducible and credential-free in local `agentcore dev`), and the live
BedrockDecider when RUNTIME!=mock and credentials resolve -- the same
DeciderService seam the offline backbone uses. This module deliberately imports
only the agent-decision layer (agent.decider_service / agent.state), never the
vision pipeline, so the AgentCore image carries no OpenCV.
"""

from __future__ import annotations

import os
from typing import Any

from bedrock_agentcore.runtime import BedrockAgentCoreApp

from agent.decider_service import resolve_decider
from agent.state import AgentDecision, InspectionState, ToolName

app = BedrockAgentCoreApp()


def _state_from_payload(raw: dict[str, Any]) -> InspectionState:
    """Rebuild an InspectionState from the request (bounds preserved)."""
    s = raw or {}
    return InspectionState(
        max_iterations=int(s.get("max_iterations", 20)),
        max_retries_per_tool=int(s.get("max_retries_per_tool", 2)),
        iteration=int(s.get("iteration", 0)),
        retries_by_tool=dict(s.get("retries_by_tool", {}) or {}),
    )


@app.entrypoint
def invoke(payload: dict[str, Any]) -> dict[str, Any]:
    """AgentCore Runtime entrypoint: choose the next agent action (decision).

    Reasons over the REAL perception (an AnalysisResult the Vision_Worker
    produced from OpenCV) and returns the next AgentDecision, constrained to the
    seven RoadGuard tools. Honest failure: a malformed request returns an error
    object, never a fabricated decision.
    """
    action = (payload or {}).get("action", "decide")
    if action == "ping":
        return {"status": "ok", "service": "roadguard-agent", "runtime": _runtime_name()}

    if action != "decide":
        return {"error": "unsupported_action", "detail": f"unknown action {action!r}; expected 'decide'"}

    perception = (payload or {}).get("perception")
    if not isinstance(perception, dict) or "status" not in perception:
        return {"error": "invalid_perception",
                "detail": "payload.perception must be a real AnalysisResult object"}

    state = _state_from_payload((payload or {}).get("state", {}))
    decider = resolve_decider()  # Bedrock when creds+RUNTIME!=mock, else Mock
    decision: AgentDecision = decider.decide(state, perception)  # type: ignore[arg-type]

    # Return only a valid, closed-vocabulary decision (5.5). AgentDecision is
    # already constrained to ToolName, so this is honest by construction.
    assert decision.tool_name in set(ToolName)
    return {
        "decision": decision.to_dict(),
        "runtime": getattr(decider, "runtime_name", _runtime_name()),
    }


def _runtime_name() -> str:
    rt = os.environ.get("RUNTIME", "mock").strip().lower()
    return "mock" if rt in ("", "mock") else "bedrock"


if __name__ == "__main__":
    # Local dev server (agentcore dev) and the AgentCore Runtime both call .run().
    app.run()
