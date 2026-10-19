"""Agent state and decision types for the RoadGuard bounded agent loop.

This module owns the small, stable data structures the bounded
perception-decision-action loop is built on (design.md "Agent Layer Design",
subsections "Runtime Abstraction" and "Bounded Decision Loop"):

* :class:`ToolName` -- the **closed** vocabulary of the seven Agent_Tools the
  agent may invoke, and nothing else (Requirement 5.5).
* :class:`StopReason` -- the closed vocabulary of why the loop terminated
  (Requirements 5.3, 5.4).
* :class:`AgentDecision` -- the structured decision a :class:`DeciderService`
  returns: the next tool to invoke, its arguments, and optional reasoning
  (Requirement 5.1, design "Reason --> Act: AgentDecision (next tool)").
* :class:`InspectionState` -- the loop's carried state
  (``iteration``, ``max_iterations``, ``retries_by_tool``, ``findings``,
  ``review_queue``, ``stop_reason``) plus the small helpers that **enforce the
  bounds** (Requirements 5.2, 5.3, 5.4).

Scope (Task 6.1): only the state and decision types and their minimal
bound-enforcement helpers live here. The decider (6.2), the trace schema (6.3),
the seven tools (6.4-6.7), and the loop itself (6.8) are implemented elsewhere
and consume these types.

The module performs no OpenCV, network, or AWS work and is importable with no
AWS credentials (Requirement 19.4, Competition 4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# --------------------------------------------------------------------------- #
# Closed tool vocabulary (Requirement 5.5)
# --------------------------------------------------------------------------- #


class ToolName(str, Enum):
    """The seven -- and only seven -- Agent_Tools the agent may invoke (5.5).

    Subclassing ``str`` keeps members JSON-serializable (``tool.value`` is the
    wire string) for the trace/persistence layer while remaining a closed
    vocabulary: a decision whose ``tool_name`` is not one of these cannot be
    constructed, so the loop can never invoke an undefined operation.

    The members match the design "The Seven Agent Tools" table exactly.
    """

    ANALYZE_FRAME = "analyze_frame"
    SELECT_ADDITIONAL_FRAMES = "select_additional_frames"
    ANALYZE_REGION = "analyze_region"
    COMPARE_VISUAL_EVIDENCE = "compare_visual_evidence"
    SAVE_INSPECTION_FINDING = "save_inspection_finding"
    REQUEST_HUMAN_REVIEW = "request_human_review"
    COMPLETE_INSPECTION = "complete_inspection"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


#: Tuple view of the closed tool vocabulary, for runtime membership checks.
TOOL_NAMES: tuple[str, ...] = tuple(t.value for t in ToolName)


# --------------------------------------------------------------------------- #
# Stop-reason vocabulary (Requirements 5.3, 5.4)
# --------------------------------------------------------------------------- #


class StopReason(str, Enum):
    """Why the Agent_Loop terminated (5.4).

    The loop stops when the agent invokes ``complete_inspection``, when the
    configured maximum iteration count is reached, or when an unrecoverable
    failure is detected -- exactly the three members below.
    """

    #: The agent chose to finish via the complete_inspection tool (5.4).
    COMPLETE_INSPECTION = "complete_inspection"
    #: The configured maximum iteration count was reached (5.3, 5.4).
    MAX_ITERATIONS = "max_iterations"
    #: An unrecoverable failure was detected (5.4, 5.7).
    UNRECOVERABLE_FAILURE = "unrecoverable_failure"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


# --------------------------------------------------------------------------- #
# Agent decision (Requirement 5.1; design "Runtime Abstraction")
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AgentDecision:
    """The structured decision a :class:`DeciderService` returns each iteration.

    Describes the single next tool to invoke, its arguments, and an optional
    human-readable ``reasoning`` string recorded in the Agent_Trace. The loop
    acts on exactly one :class:`AgentDecision` per iteration (5.1).

    ``tool_name`` is a :class:`ToolName`, so a decision can only name one of the
    seven defined tools (5.5). ``arguments`` is a free-form mapping validated by
    the individual tool (6.4-6.7); keeping it a plain ``dict`` here lets each
    tool own its own input schema without this module depending on them.

    Frozen (immutable): a decision is a value recorded verbatim in the trace.
    """

    tool_name: ToolName
    arguments: dict[str, Any] = field(default_factory=dict)
    reasoning: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.tool_name, ToolName):
            raise ValueError(
                f"tool_name must be a ToolName (one of {TOOL_NAMES}), "
                f"got {self.tool_name!r}"
            )
        if not isinstance(self.arguments, dict):
            raise ValueError(f"arguments must be a dict, got {type(self.arguments).__name__}")

    @classmethod
    def of(
        cls,
        tool: ToolName | str,
        arguments: dict[str, Any] | None = None,
        reasoning: str | None = None,
    ) -> AgentDecision:
        """Build a decision, coercing a tool string into the closed vocabulary.

        Accepts either a :class:`ToolName` or its wire string (e.g.
        ``"analyze_frame"``). An unknown string raises ``ValueError`` -- there is
        no way to produce a decision for an undefined tool (5.5).
        """
        resolved = tool if isinstance(tool, ToolName) else ToolName(tool)
        return cls(tool_name=resolved, arguments=dict(arguments or {}), reasoning=reasoning)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view for the trace/persistence layer."""
        return {
            "tool_name": self.tool_name.value,
            "arguments": dict(self.arguments),
            "reasoning": self.reasoning,
        }


# --------------------------------------------------------------------------- #
# Inspection state with bound-enforcement helpers (Requirements 5.2, 5.3, 5.4)
# --------------------------------------------------------------------------- #


DEFAULT_MAX_ITERATIONS = 20
DEFAULT_MAX_RETRIES_PER_TOOL = 3


@dataclass
class InspectionState:
    """The state the bounded Agent_Loop carries across iterations.

    Fields match design "Bounded Decision Loop" ("An ``InspectionState`` carries
    ``iteration``, ``max_iterations``, ``retries_by_tool``, ``findings``,
    ``review_queue``, and ``stop_reason``").

    This is a clean data structure with small helper methods that enforce the
    bounds (5.2) and detect the stopping conditions (5.3, 5.4). The loop (task
    6.8) drives these helpers; it does not reach inside the fields to mutate
    counters directly.

    ``findings`` and ``review_queue`` hold the records produced by the
    ``save_inspection_finding`` (6.6) and ``request_human_review`` (6.6) tools.
    They are typed loosely as ``list[dict]`` here so those later tasks can refine
    the record shape without breaking this interface.
    """

    max_iterations: int = DEFAULT_MAX_ITERATIONS
    max_retries_per_tool: int = DEFAULT_MAX_RETRIES_PER_TOOL

    iteration: int = 0
    retries_by_tool: dict[str, int] = field(default_factory=dict)
    findings: list[dict[str, Any]] = field(default_factory=list)
    review_queue: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: StopReason | None = None

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {self.max_iterations}")
        if self.max_retries_per_tool < 0:
            raise ValueError(
                f"max_retries_per_tool must be >= 0, got {self.max_retries_per_tool}"
            )
        if self.iteration < 0:
            raise ValueError(f"iteration must be >= 0, got {self.iteration}")

    # -- iteration bound (5.2, 5.3) ----------------------------------------- #

    def begin_iteration(self) -> int:
        """Increment and return the iteration counter for the next loop step.

        Raises ``RuntimeError`` if the loop is already stopped -- the loop must
        check :meth:`should_stop` before starting another iteration.
        """
        if self.stop_reason is not None:
            raise RuntimeError(
                f"cannot begin another iteration after the loop stopped ({self.stop_reason})"
            )
        self.iteration += 1
        return self.iteration

    def iteration_budget_exhausted(self) -> bool:
        """True once the configured maximum iteration count is reached (5.3)."""
        return self.iteration >= self.max_iterations

    # -- per-tool retry bound (5.2, 5.7) ------------------------------------ #

    def retries_for(self, tool: ToolName | str) -> int:
        """Return the recorded retry count for ``tool`` (0 if never retried)."""
        return self.retries_by_tool.get(_tool_key(tool), 0)

    def record_retry(self, tool: ToolName | str) -> int:
        """Increment and return the per-tool retry count after a failed attempt."""
        key = _tool_key(tool)
        self.retries_by_tool[key] = self.retries_by_tool.get(key, 0) + 1
        return self.retries_by_tool[key]

    def retry_budget_exhausted(self, tool: ToolName | str) -> bool:
        """True once ``tool`` has reached its configured maximum retry count (5.2).

        At this point the loop records the failure and must escalate via
        ``request_human_review`` or invoke ``complete_inspection`` (5.7).
        """
        return self.retries_for(tool) >= self.max_retries_per_tool

    # -- stopping conditions (5.3, 5.4) ------------------------------------- #

    def mark_stopped(self, reason: StopReason) -> StopReason:
        """Record why the loop stopped; the first reason recorded wins.

        Idempotent-ish: once a stop reason is set it is not overwritten, so the
        terminal reason reflects the first condition that fired.
        """
        if not isinstance(reason, StopReason):
            raise ValueError(f"reason must be a StopReason, got {reason!r}")
        if self.stop_reason is None:
            self.stop_reason = reason
        return self.stop_reason

    def should_stop(self) -> bool:
        """True when the loop must terminate (5.4).

        The loop stops when a stop reason has already been recorded (e.g.
        ``complete_inspection`` was invoked, or an unrecoverable failure was
        detected) or when the iteration budget is exhausted. When the budget is
        exhausted and no reason is set yet, this flips the stop reason to
        :attr:`StopReason.MAX_ITERATIONS` so the terminal reason is always
        explicit (5.3).
        """
        if self.stop_reason is not None:
            return True
        if self.iteration_budget_exhausted():
            self.mark_stopped(StopReason.MAX_ITERATIONS)
            return True
        return False

    @property
    def stopped(self) -> bool:
        """Whether a terminal stop reason has been recorded."""
        return self.stop_reason is not None

    # -- finding / review accumulation -------------------------------------- #

    def add_finding(self, finding: dict[str, Any]) -> None:
        """Append a saved Inspection_Finding record (populated by tool 6.6)."""
        self.findings.append(finding)

    def enqueue_review(self, item: dict[str, Any]) -> None:
        """Append an escalated Human_Review_Queue item (populated by tool 6.6)."""
        self.review_queue.append(item)


def _tool_key(tool: ToolName | str) -> str:
    """Normalize a tool reference to its wire string, validating the vocabulary."""
    if isinstance(tool, ToolName):
        return tool.value
    # Raises ValueError for an unknown string -> only the seven tools are keys.
    return ToolName(tool).value


# --------------------------------------------------------------------------- #
# Self-validation (import-time smoke check of the bound helpers)
# --------------------------------------------------------------------------- #


def _self_check() -> None:
    """Assert the bound/stop helpers behave. Runs under ``python -m``.

    This is a lightweight guard, not the property-test suite (tasks 6.9/6.10).
    """
    # --- closed tool vocabulary -------------------------------------------- #
    assert len(TOOL_NAMES) == 7, TOOL_NAMES
    assert ToolName("analyze_frame") is ToolName.ANALYZE_FRAME
    try:
        ToolName("rm -rf /")  # not one of the seven (5.5)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("ToolName accepted an undefined tool")

    # --- AgentDecision ----------------------------------------------------- #
    d = AgentDecision.of("analyze_frame", {"frame_ref": "f0", "method": "classical"}, "start")
    assert d.tool_name is ToolName.ANALYZE_FRAME
    assert d.to_dict()["tool_name"] == "analyze_frame"
    try:
        AgentDecision(tool_name="analyze_frame")  # type: ignore[arg-type]
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("AgentDecision accepted a raw string tool_name")

    # --- iteration bound flips the stopping condition (5.3) ---------------- #
    st = InspectionState(max_iterations=2, max_retries_per_tool=2)
    assert not st.should_stop()
    assert st.begin_iteration() == 1
    assert not st.should_stop()
    assert st.begin_iteration() == 2
    # iteration now == max_iterations -> budget exhausted -> MAX_ITERATIONS
    assert st.iteration_budget_exhausted()
    assert st.should_stop()
    assert st.stop_reason is StopReason.MAX_ITERATIONS
    assert st.stopped

    # beginning another iteration after stopping is refused
    try:
        st.begin_iteration()
    except RuntimeError:
        pass
    else:  # pragma: no cover
        raise AssertionError("begin_iteration ran after the loop stopped")

    # --- per-tool retry counting (5.2) ------------------------------------- #
    st2 = InspectionState(max_iterations=5, max_retries_per_tool=2)
    tool = ToolName.ANALYZE_REGION
    assert st2.retries_for(tool) == 0
    assert not st2.retry_budget_exhausted(tool)
    assert st2.record_retry(tool) == 1
    assert not st2.retry_budget_exhausted(tool)
    assert st2.record_retry(tool) == 2
    assert st2.retry_budget_exhausted(tool)  # reached the cap
    # retries are tracked independently per tool
    assert st2.retries_for(ToolName.ANALYZE_FRAME) == 0

    # --- first stop reason wins (5.4) -------------------------------------- #
    st3 = InspectionState()
    st3.mark_stopped(StopReason.COMPLETE_INSPECTION)
    st3.mark_stopped(StopReason.UNRECOVERABLE_FAILURE)
    assert st3.stop_reason is StopReason.COMPLETE_INSPECTION
    assert st3.should_stop()

    # --- accumulation ------------------------------------------------------ #
    st4 = InspectionState()
    st4.add_finding({"hazard_type": "pothole"})
    st4.enqueue_review({"reason": "low_confidence"})
    assert len(st4.findings) == 1 and len(st4.review_queue) == 1


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    _self_check()
    print("[agent.state] OK: bound-enforcement and decision invariants hold.")
