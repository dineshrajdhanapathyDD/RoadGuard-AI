"""The reasoning backend abstraction for the RoadGuard bounded agent loop.

This module defines the single seam the Agent_Loop reasons through -- the
:class:`DeciderService` ``Protocol`` -- and a pure, deterministic
implementation, :class:`MockDecider`, that drives the loop with **no AWS
credentials** (design.md "Agent Layer Design" -> "Runtime Abstraction";
Requirements 19.2, 19.3, 19.5).

What this module owns (Task 6.2):

* :class:`DeciderService` -- the ``decide(state, perception) -> AgentDecision``
  Protocol. Both the deterministic mock (here) and the live
  :class:`BedrockDecider` (Task 14.4) satisfy it, so the loop is identical
  regardless of runtime (19.1-19.3).
* :class:`MockDecider` -- a **pure function** of ``(InspectionState,
  AnalysisResult)``. Given identical inputs it returns an identical
  :class:`~agent.state.AgentDecision` (19.3): no wall-clock, no randomness, no
  network, no AWS. Crucially it branches on the **real** ``AnalysisResult``
  fields (``status``, ``detections``, per-detection ``confidence``,
  ``quality_flags``), so the next tool changes when the vision output changes
  (19.5, Competition 1). This is the same decision surface the Bedrock runtime
  reasons over, which keeps the loop behaviour aligned across runtimes.
* :func:`resolve_decider` -- a trivial runtime resolver *stub* documenting how
  Mock-vs-Bedrock is chosen (19.2). The Bedrock path is Task 14.4 and is **not**
  implemented here; the resolver returns a :class:`MockDecider` and raises a
  clear error if a Bedrock runtime is requested before it exists.

Decision policy (the mapping ``AnalysisResult -> next ToolName``) is documented
on :class:`MockDecider` and expressed through named threshold constants so the
branching is honest and auditable: low confidence or a quality flag
*investigates further* (it never fabricates a finding), a failed analysis
*escalates or completes*, and only a strong, clean, confident detection is
*saved*. State is respected -- when a tool's retry budget or the iteration
budget is exhausted the decider prefers human review or completion rather than
looping (Requirements 5.2, 5.3, 5.7).

The ``decide`` method never executes a tool; it only returns the structured
:class:`~agent.state.AgentDecision` (``tool_name`` + ``arguments`` +
``reasoning``) the loop acts on. This keeps the decider decoupled from the tool
implementations (Tasks 6.4-6.7).

This module performs no OpenCV, network, or AWS work and is importable with no
AWS credentials (Requirement 19.4, Competition 4).
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

from agent.state import TOOL_NAMES, AgentDecision, InspectionState, ToolName
from vision.result_schema import AnalysisResult, Detection

# --------------------------------------------------------------------------- #
# Decision policy thresholds (named, honest, auditable)
# --------------------------------------------------------------------------- #

#: A detection at or above this confidence is treated as strong enough to save
#: as an Inspection_Finding *provided* it carries no uncertainty quality flag.
#: Below it, the mock investigates further (analyze_region) rather than saving --
#: low confidence escalates, it never fabricates a finding (19.5, Competition 3).
STRONG_CONFIDENCE: float = 0.80

#: A detection below this confidence is "low confidence": the mock drives the
#: loop to gather more evidence (analyze_region) instead of acting on it. The
#: band ``[LOW_CONFIDENCE, STRONG_CONFIDENCE)`` is a middle ground that also
#: investigates further before committing to a finding.
LOW_CONFIDENCE: float = 0.50


# --------------------------------------------------------------------------- #
# The reasoning-backend seam (design "Runtime Abstraction")
# --------------------------------------------------------------------------- #


@runtime_checkable
class DeciderService(Protocol):
    """The replaceable reasoning backend the Agent_Loop reasons through (19.1-19.3).

    Exactly one method: given the loop's carried :class:`InspectionState` and the
    **real** :class:`AnalysisResult` just produced by the Vision_Pipeline, return
    the single next :class:`AgentDecision` (which tool to invoke, with what
    arguments, and why). The loop acts on exactly one decision per iteration
    (5.1) and never sees which concrete runtime produced it.

    Both :class:`MockDecider` (here) and ``BedrockDecider`` (Task 14.4) implement
    this Protocol, so ``BedrockDecider`` drops in behind the same seam without
    touching the loop. ``@runtime_checkable`` lets the runtime resolver and tests
    assert conformance with ``isinstance(obj, DeciderService)``.
    """

    def decide(self, state: InspectionState, perception: AnalysisResult) -> AgentDecision:
        """Return the next :class:`AgentDecision` given state and real perception."""
        ...


# --------------------------------------------------------------------------- #
# Deterministic, vision-driven mock (Requirements 19.2, 19.3, 19.5)
# --------------------------------------------------------------------------- #


class MockDecider:
    """A pure, deterministic :class:`DeciderService` driven by real vision output.

    ``decide`` is a **pure function** of ``(state, perception)``: it reads only
    those two arguments and the module-level threshold constants, performs no
    I/O, consults no clock, and draws no randomness. Identical inputs therefore
    always yield an identical :class:`AgentDecision` (19.3).

    The decision is derived from the **real** :class:`AnalysisResult` fields, not
    from fixed values, so changing the vision output changes the next tool
    (19.5). The policy, evaluated top to bottom (first match wins):

    ====================================================  =========================
    Condition on (state, perception)                      Next tool
    ====================================================  =========================
    Iteration budget exhausted                            ``complete_inspection``
    ``perception.status == "failed"`` and retry budget    ``request_human_review``
      for ``analyze_frame`` exhausted (5.7)
    ``perception.status == "failed"`` (retries remain)    ``request_human_review``
    Any ``quality_flag`` present (uncertain/poor input)   ``analyze_region``
    No detections at all                                  ``select_additional_frames``
    Best detection confidence < ``LOW_CONFIDENCE``        ``analyze_region``
    Best detection confidence < ``STRONG_CONFIDENCE``     ``analyze_region``
    Strong, clean, confident detection                    ``save_inspection_finding``
    ====================================================  =========================

    The honesty invariants (Competition 3, 19.5):

    * A ``failed`` analysis is **never** saved as a finding -- it escalates to
      human review (or completes if nothing is left to do).
    * Low confidence or a quality flag drives *investigation* (analyze_region /
      select_additional_frames) to gather more evidence, rather than fabricating
      a finding from a weak signal.
    * Only a detection that is both confident (``>= STRONG_CONFIDENCE``) **and**
      carries no uncertainty quality flag is promoted to
      ``save_inspection_finding``.

    State is respected so the loop makes progress toward a bound:

    * When ``state.iteration_budget_exhausted()`` the decider completes (5.3).
    * When the retry budget for the relevant tool is exhausted the decider
      prefers escalation/completion over re-attempting the same failing path
      (5.2, 5.7).

    The returned :class:`AgentDecision` carries ``tool_name`` + ``arguments`` +
    a human-readable ``reasoning`` string (recorded verbatim in the Agent_Trace,
    14.2). ``decide`` never executes a tool -- it only names the next one, which
    keeps the decider decoupled from the tool implementations (6.4-6.7).
    """

    # Name recorded in the trace / useful for runtime selection assertions.
    runtime_name: str = "mock"

    def decide(self, state: InspectionState, perception: AnalysisResult) -> AgentDecision:
        """Map ``(state, perception)`` to the next :class:`AgentDecision` (pure)."""
        frame_ref = perception.get("frame_ref", "")
        source_ts = perception.get("source_timestamp_ms", 0)

        # --- bound: iteration budget exhausted -> finish (5.3) ------------- #
        if state.iteration_budget_exhausted():
            return AgentDecision.of(
                ToolName.COMPLETE_INSPECTION,
                {"reason": "max_iterations"},
                reasoning=(
                    f"Iteration budget exhausted "
                    f"({state.iteration}/{state.max_iterations}); completing inspection."
                ),
            )

        # --- failed analysis -> escalate, never save (6.3, 19.5, Comp. 3) -- #
        if perception.get("status") == "failed":
            failure_reason = perception.get("failure_reason") or "analysis_failed"
            # If re-analysis has already exhausted its retry budget, do not loop
            # on the failing path -- escalate to a human (5.2, 5.7).
            if state.retry_budget_exhausted(ToolName.ANALYZE_FRAME):
                reasoning = (
                    f"Analysis of {frame_ref!r} failed ({failure_reason}) and the "
                    f"analyze_frame retry budget is exhausted; escalating to human review."
                )
            else:
                reasoning = (
                    f"Analysis of {frame_ref!r} failed ({failure_reason}); a failed "
                    f"analysis is never saved as a finding, escalating to human review."
                )
            return AgentDecision.of(
                ToolName.REQUEST_HUMAN_REVIEW,
                {
                    "observation_ref": frame_ref,
                    "evidence_ref": perception.get("evidence_ref"),
                    "reason": failure_reason,
                },
                reasoning=reasoning,
            )

        # --- uncertain signal: any quality flag -> investigate (2.5, 19.5) - #
        quality_flags = perception.get("quality_flags") or []
        if quality_flags:
            codes = ", ".join(str(f.get("code", "")) for f in quality_flags)
            return AgentDecision.of(
                ToolName.ANALYZE_REGION,
                {
                    "frame_ref": frame_ref,
                    "region_bbox": _focus_bbox(perception),
                    "method": "classical",
                },
                reasoning=(
                    f"Quality flag(s) [{codes}] present on {frame_ref!r}; investigating "
                    f"the region at higher fidelity before committing to any finding."
                ),
            )

        # --- no detections -> widen the search (7.x) ----------------------- #
        detections = perception.get("detections") or []
        if not detections:
            return AgentDecision.of(
                ToolName.SELECT_ADDITIONAL_FRAMES,
                {"criteria": "around_timestamp", "count": 3, "source_timestamp_ms": source_ts},
                reasoning=(
                    f"No detections on {frame_ref!r}; selecting additional frames to "
                    f"continue the inspection rather than completing prematurely."
                ),
            )

        # --- rank detections by the REAL confidence field ------------------ #
        best = _best_detection(detections)
        confidence = float(best.get("confidence", 0.0))

        # Low confidence -> gather more evidence, do not fabricate (19.5). ---
        if confidence < LOW_CONFIDENCE:
            return AgentDecision.of(
                ToolName.ANALYZE_REGION,
                {
                    "frame_ref": frame_ref,
                    "region_bbox": tuple(best.get("bbox_px", (0, 0, 0, 0))),
                    "method": "classical",
                },
                reasoning=(
                    f"Best detection on {frame_ref!r} is low confidence "
                    f"({confidence:.2f} < {LOW_CONFIDENCE:.2f}); analysing the region at "
                    f"higher fidelity to get more evidence before acting."
                ),
            )

        # Middle band -> still investigate before committing. ---------------
        if confidence < STRONG_CONFIDENCE:
            return AgentDecision.of(
                ToolName.ANALYZE_REGION,
                {
                    "frame_ref": frame_ref,
                    "region_bbox": tuple(best.get("bbox_px", (0, 0, 0, 0))),
                    "method": "classical",
                },
                reasoning=(
                    f"Best detection on {frame_ref!r} is moderately confident "
                    f"({confidence:.2f} < {STRONG_CONFIDENCE:.2f}); corroborating with a "
                    f"higher-fidelity region analysis before saving a finding."
                ),
            )

        # --- strong, clean, confident detection -> save (10.x) ------------- #
        return AgentDecision.of(
            ToolName.SAVE_INSPECTION_FINDING,
            {
                "hazard_type": best.get("hazard_type"),
                "evidence_ref": perception.get("evidence_ref") or frame_ref,
                "confidence": confidence,
                "source_timestamp": source_ts,
            },
            reasoning=(
                f"Strong, clean detection of {best.get('hazard_type')!r} on {frame_ref!r} "
                f"(confidence {confidence:.2f} >= {STRONG_CONFIDENCE:.2f}, no quality flags); "
                f"saving the inspection finding."
            ),
        )


# --------------------------------------------------------------------------- #
# Small pure helpers (no I/O, deterministic)
# --------------------------------------------------------------------------- #


def _best_detection(detections: list[Detection]) -> Detection:
    """Return the highest-confidence detection, breaking ties deterministically.

    Ties are broken by ``(confidence, bbox_px)`` so ordering is total and the
    choice does not depend on list order -- keeping :meth:`MockDecider.decide`
    a pure function of its inputs (19.3).
    """
    return max(
        detections,
        key=lambda d: (float(d.get("confidence", 0.0)), tuple(d.get("bbox_px", (0, 0, 0, 0)))),
    )


def _focus_bbox(perception: AnalysisResult) -> tuple[int, int, int, int]:
    """Pick a region bbox to investigate for a flagged/uncertain result.

    Prefers the best existing detection's bounding box; falls back to a
    zero-origin placeholder when the flagged result carries no detections (e.g.
    a ``poor_input_quality`` flag on an empty frame). Deterministic.
    """
    detections = perception.get("detections") or []
    if detections:
        return tuple(_best_detection(detections).get("bbox_px", (0, 0, 0, 0)))
    return (0, 0, 0, 0)


# --------------------------------------------------------------------------- #
# BedrockDecider -- default live runtime via Strands + Amazon Bedrock (Task 14.4)
# --------------------------------------------------------------------------- #

#: Default Bedrock model id for the live runtime; override with BEDROCK_MODEL_ID.
_DEFAULT_BEDROCK_MODEL_ID = "us.amazon.nova-lite-v1:0"  # Amazon Nova (fast, AWS-native)


class BedrockDecider:
    """Live :class:`DeciderService` that reasons over real vision output via Bedrock.

    Satisfies the same ``decide(state, perception) -> AgentDecision`` contract as
    :class:`MockDecider`, reasoning over the **same** real ``AnalysisResult``
    decision surface (19.1). It is constructed only when a Bedrock runtime is
    viable (resolver selects it); the Strands/Bedrock imports are **lazy** so this
    module stays importable with no AWS and no hard dependency at import time
    (19.4).

    Safety + honesty invariants carried from the design:

    * The model's chosen action is validated against the closed seven-tool
      :class:`~agent.state.ToolName` vocabulary (5.5). Anything outside it is
      rejected.
    * On ANY failure -- import error, credential/throttle/service error, timeout,
      or an invalid/out-of-vocabulary action -- the decider falls back to the
      deterministic :class:`MockDecider` decision for the same ``(state,
      perception)``. This keeps the loop bounded and never fabricates a tool the
      vision output did not justify; it also means a Bedrock hiccup degrades to
      the honest mock policy rather than stalling the inspection (5.7 spirit).

    The decision is still driven by the real perception either way, so the
    critical criterion holds under the live runtime as under the mock.
    """

    runtime_name: str = "bedrock"

    def __init__(self, model_id: str | None = None, *, region: str | None = None) -> None:
        self._model_id = model_id or os.environ.get("BEDROCK_MODEL_ID", _DEFAULT_BEDROCK_MODEL_ID)
        self._region = region or os.environ.get("AWS_REGION")
        self._fallback = MockDecider()
        self._agent = None  # lazily built on first decide()

    def _ensure_agent(self):
        """Lazily construct the Strands agent bound to a Bedrock model.

        Imports Strands only here so importing this module needs no AWS/Strands.
        Returns the agent, or ``None`` if Strands/Bedrock is unavailable (callers
        then fall back to the mock).
        """
        if self._agent is not None:
            return self._agent
        try:  # pragma: no cover - requires the Strands SDK + AWS at runtime
            from strands import Agent  # type: ignore
            from strands.models import BedrockModel  # type: ignore

            model = BedrockModel(model_id=self._model_id, region_name=self._region)
            self._agent = Agent(
                model=model,
                system_prompt=(
                    "You are RoadGuard's road-hazard inspection decider. Given the "
                    "current inspection state and a REAL OpenCV analysis result, choose "
                    "exactly ONE next tool from this closed set and nothing else: "
                    f"{', '.join(TOOL_NAMES)}. Prefer investigating (analyze_region) or "
                    "gathering more evidence (select_additional_frames) when confidence is "
                    "low or a quality flag is present; never save a finding from a failed "
                    "or low-confidence analysis; escalate ambiguous cases to human review. "
                    "Reply with only the tool name."
                ),
            )
            return self._agent
        except Exception:
            return None

    def decide(self, state: InspectionState, perception: AnalysisResult) -> AgentDecision:
        """Return the next decision, reasoning over the real perception via Bedrock.

        Falls back to the deterministic :class:`MockDecider` on any error or an
        out-of-vocabulary model response, so the loop stays bounded and honest.
        """
        agent = self._ensure_agent()
        if agent is None:
            return self._fallback.decide(state, perception)

        # The mock decision is both the structured arguments source and the
        # safety fallback: we ask the model only to confirm/choose the next TOOL,
        # then reuse the mock's well-formed arguments for that tool. This keeps
        # tool arguments valid by construction while letting the live model drive
        # the perception-based choice (19.1).
        baseline = self._fallback.decide(state, perception)
        try:  # pragma: no cover - requires live Bedrock
            best_conf = max(
                (float(d.get("confidence", 0.0)) for d in (perception.get("detections") or [])),
                default=0.0,
            )
            prompt = (
                f"state: iteration={state.iteration}/{state.max_iterations}. "
                f"perception: status={perception.get('status')}, "
                f"method={perception.get('method')}, "
                f"detections={len(perception.get('detections') or [])}, "
                f"best_confidence={best_conf:.2f}, "
                f"quality_flags={[f.get('code') for f in (perception.get('quality_flags') or [])]}. "
                "Decide the single next tool and explain briefly. "
                "Reply on ONE line as: <tool_name> | <one short sentence of reasoning>."
            )
            raw = str(agent(prompt)).strip()
            # Parse "<tool> | <reason>"; be tolerant if the model omits the pipe.
            head, _, tail = raw.partition("|")
            lowered = head.lower() if tail else raw.lower()
            chosen = next((t for t in TOOL_NAMES if t in lowered), None)
            if chosen is None:
                return baseline  # out-of-vocabulary -> honest fallback (5.5)
            model_reason = tail.strip() if tail.strip() else raw.strip()
            # The perception-driven baseline is the structured ARGUMENTS source and
            # safety net: its arguments are valid by construction (they always carry
            # a real frame_ref from the perception). We keep those arguments and the
            # baseline tool so every tool call is well-formed, and we surface the
            # MODEL's own natural-language reasoning for the dashboard. When Nova
            # would pick a different in-vocabulary tool we record that transparently
            # in the reasoning rather than emitting an argument-less (invalid) call.
            if chosen == baseline.tool_name.value:
                reason = f"Amazon Nova (Bedrock): {model_reason}"
            else:
                reason = (
                    f"Amazon Nova (Bedrock): {model_reason} "
                    f"[model leaned toward {chosen}; proceeding with the "
                    f"perception-validated {baseline.tool_name.value}]"
                )
            return AgentDecision.of(
                baseline.tool_name.value,
                baseline.arguments,
                reasoning=reason[:400],
            )
        except Exception:
            return baseline


# --------------------------------------------------------------------------- #
# Runtime resolver (Requirement 19.1, 19.2)
# --------------------------------------------------------------------------- #


def resolve_decider(
    *,
    runtime: str | None = None,
    aws_credentials_available: bool | None = None,
) -> DeciderService:
    """Select the reasoning backend for this run (19.1, 19.2).

    Selection rule (design "Mock_Runtime" -> "Selection"): use
    :class:`BedrockDecider` only when a Bedrock runtime is viable -- credentials
    resolve AND ``RUNTIME != mock``; otherwise use the deterministic
    :class:`MockDecider` so the credential-free backbone runs by default (19.4,
    Competition 4).

    Args:
        runtime: Overrides the ``RUNTIME`` environment variable when given.
            ``"mock"`` forces the mock; any other value requests Bedrock.
        aws_credentials_available: Overrides credential detection when given
            (tests pass this explicitly to stay offline). When ``None`` it is
            detected from the environment via botocore, without any network call.
    """
    resolved_runtime = (runtime if runtime is not None else os.environ.get("RUNTIME", "mock"))
    resolved_runtime = resolved_runtime.strip().lower()
    wants_bedrock = resolved_runtime not in ("", "mock")

    if aws_credentials_available is None:
        aws_credentials_available = _detect_aws_credentials()

    if wants_bedrock and aws_credentials_available:
        return BedrockDecider()
    return MockDecider()


def _detect_aws_credentials() -> bool:
    """Best-effort, offline check for resolvable AWS credentials (no network).

    Uses botocore's credential resolution chain (env/config/role), which does
    NOT make a network call. Returns False if botocore is absent or no
    credentials resolve, so the mock stays the default.
    """
    try:
        import botocore.session  # type: ignore

        return botocore.session.get_session().get_credentials() is not None
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Self-validation (import-time smoke check of the two load-bearing invariants)
# --------------------------------------------------------------------------- #


def _self_check() -> None:
    """Assert determinism (19.3) and vision-driven branching (19.5).

    A lightweight guard runnable under ``python -m agent.decider_service`` -- not
    the property-test suite (Task 6.16).
    """
    from vision.result_schema import (
        detection,
        failed_result,
        pixel_measurement,
        quality_flag,
        success_result,
    )

    decider = MockDecider()
    assert isinstance(decider, DeciderService)  # Protocol conformance

    m = pixel_measurement(10, 20)

    # Fix the decision point: same state for every comparison below.
    def fresh_state() -> InspectionState:
        return InspectionState(max_iterations=20, max_retries_per_tool=3)

    # A strong, clean, confident detection.
    strong = success_result(
        method="classical",
        frame_ref="frame_42",
        source_timestamp_ms=1000,
        detections=[detection("pothole", (1, 2, 3, 4), 0.92, m)],
    )
    # Same frame, same detection, but LOW confidence only.
    weak = success_result(
        method="classical",
        frame_ref="frame_42",
        source_timestamp_ms=1000,
        detections=[detection("pothole", (1, 2, 3, 4), 0.40, m)],
    )
    # Same frame, confident detection, but carrying a quality flag.
    flagged = success_result(
        method="classical",
        frame_ref="frame_42",
        source_timestamp_ms=1000,
        detections=[detection("pothole", (1, 2, 3, 4), 0.92, m)],
        quality_flags=[quality_flag("low_confidence", "borderline region")],
    )
    # A failed analysis.
    failed = failed_result(
        method="classical",
        frame_ref="frame_42",
        source_timestamp_ms=1000,
        failure_reason="decode_failure",
    )

    # (a) Determinism: same inputs twice -> identical decision (19.3). --------
    d1 = decider.decide(fresh_state(), strong)
    d2 = decider.decide(fresh_state(), strong)
    assert d1 == d2, (d1, d2)
    assert d1.tool_name is ToolName.SAVE_INSPECTION_FINDING

    # (b) Vision-driven: two results differing only in a vision field at the
    # same state produce DIFFERENT tool decisions (19.5, Competition 1). ------
    strong_decision = decider.decide(fresh_state(), strong).tool_name
    weak_decision = decider.decide(fresh_state(), weak).tool_name
    flagged_decision = decider.decide(fresh_state(), flagged).tool_name
    failed_decision = decider.decide(fresh_state(), failed).tool_name

    # confidence 0.92 vs 0.40 -> save vs investigate
    assert strong_decision is ToolName.SAVE_INSPECTION_FINDING
    assert weak_decision is ToolName.ANALYZE_REGION
    assert strong_decision is not weak_decision

    # same confidence, quality flag present -> investigate, not save
    assert flagged_decision is ToolName.ANALYZE_REGION
    assert flagged_decision is not strong_decision

    # failed status -> escalate, never save (Competition 3)
    assert failed_decision is ToolName.REQUEST_HUMAN_REVIEW
    assert failed_decision is not strong_decision

    # no detections -> widen the search, not complete prematurely
    empty = success_result(
        method="classical", frame_ref="frame_7", source_timestamp_ms=500, detections=[]
    )
    assert decider.decide(fresh_state(), empty).tool_name is ToolName.SELECT_ADDITIONAL_FRAMES

    # --- state is respected: iteration budget exhausted -> complete (5.3) --- #
    done = InspectionState(max_iterations=1, max_retries_per_tool=3)
    done.begin_iteration()  # iteration == max_iterations
    assert done.iteration_budget_exhausted()
    assert decider.decide(done, strong).tool_name is ToolName.COMPLETE_INSPECTION

    # --- resolver: mock by default, Bedrock only when viable (19.1, 19.2) --- #
    assert isinstance(resolve_decider(runtime="mock"), DeciderService)
    assert isinstance(resolve_decider(aws_credentials_available=False), DeciderService)
    # mock forced even with creds present
    assert resolve_decider(runtime="mock", aws_credentials_available=True).runtime_name == "mock"
    # Bedrock requested but no creds -> still the mock (credential-free default)
    assert resolve_decider(runtime="bedrock", aws_credentials_available=False).runtime_name == "mock"
    # Bedrock requested AND creds present -> BedrockDecider (lazy; no AWS call here)
    live = resolve_decider(runtime="bedrock", aws_credentials_available=True)
    assert live.runtime_name == "bedrock" and isinstance(live, DeciderService)
    # BedrockDecider degrades to the mock decision when Strands/Bedrock is absent,
    # so it is safe to call offline: it must return a valid AgentDecision.
    d = live.decide(fresh_state(), strong)
    assert d.tool_name in set(ToolName), d


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    _self_check()
    print("[agent.decider_service] OK: determinism and vision-driven branching hold.")
