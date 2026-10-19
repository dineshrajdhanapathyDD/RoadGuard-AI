"""Agent-trace entry schema for the RoadGuard agent loop.

This module defines :class:`TraceEntry` -- the ordered record the agent writes
once per decision step. It is the artifact that *proves* real vision output
drove each decision: every entry pairs the vision output the agent perceived
(``before_output``) with the reasoning over it, the tool it invoked, the
subsequent vision output (``after_output``), and the final decision. Persisting
this ``before -> reasoning -> tool -> after -> decision`` sequence in order is
the evidence that satisfies the critical agentic criterion and refutes the
"fixed vision + narrative" anti-pattern.

See .kiro/specs/roadguard-ai/design.md:
* "Agent Layer Design" -> "Vision Output Driving the Next Action (the critical
  criterion)" for the before/decision/after triple.
* "Data Models" -> "Single-Table Design" Trace-entry row:
  ``perception, reasoning, tool, before_output, after_output, decision, ts``.

Requirements: 14.1 (append perception, reasoning, tool, result, decision in
order), 14.2 (present initial output, reasoning, tool, subsequent output, final
decision), 19.4 (importable and runnable with no AWS credentials).

The shape is a ``TypedDict`` (matching ``result_schema.py``), so entries are
trivially JSON-serializable for the DynamoDB trace row added in a later task
(8.1 / 12.2). ``before_output`` / ``after_output`` hold the real
:class:`~vision.result_schema.AnalysisResult` (itself a ``TypedDict``) or, where
a step has no vision output on one side, the explicit :data:`UNAVAILABLE`
sentinel rather than a fabricated value (R13.4 style).

This module performs no OpenCV, network, or AWS work and is importable with no
AWS credentials.
"""

from __future__ import annotations

from typing import TypedDict, Union

from vision.result_schema import UNAVAILABLE, AnalysisResult, ComparisonResult

# --------------------------------------------------------------------------- #
# Trace output vocabulary
# --------------------------------------------------------------------------- #

#: A perceived or produced vision output recorded on one side of a trace step.
#: It is either a real :class:`AnalysisResult`, a :class:`ComparisonResult`
#: (for the ``compare_visual_evidence`` step), or the explicit
#: :data:`UNAVAILABLE` sentinel string when a step has no vision output on that
#: side (e.g. the first perception has no ``before_output``). It is never a
#: fabricated value (3.3, 13.4).
TraceOutput = Union[AnalysisResult, ComparisonResult, str]


class TraceEntry(TypedDict):
    """One ordered agent-trace entry, written once per decision step (14.1, 14.2).

    Fields match the design's Trace-entry row exactly:

    * ``perception`` -- the perception input the agent acted on this step
      (e.g. the ``frame_ref`` analyzed, or the tool input summary). This is the
      *input* side described by 14.1.
    * ``reasoning`` -- the reasoning step over the real vision output, in words.
    * ``tool`` -- the single Agent_Tool invoked this iteration (5.1).
    * ``before_output`` -- the vision output the decision was made *from* (the
      "initial OpenCV output" of 14.2). Holds a real :class:`AnalysisResult` /
      :class:`ComparisonResult`, or :data:`UNAVAILABLE` when there is none.
    * ``after_output`` -- the vision output produced *by* the invoked tool (the
      "subsequent output" of 14.2), or :data:`UNAVAILABLE` for tools that
      produce no vision output (e.g. ``save_inspection_finding``).
    * ``decision`` -- the final decision the agent reached this step.
    * ``ts`` -- the ordering key for the entry. The trace is an **ordered**
      sequence (14.1); ``ts`` is the monotonic iteration/sequence number the
      store uses to preserve order (zero-padded into the DynamoDB SK later).

    ``before_output`` and ``after_output`` are deliberately able to hold the
    real :class:`AnalysisResult`: that is what lets a test feed two materially
    different vision outputs at the same decision point and show the persisted
    trace captured both the differing inputs and the differing decisions.
    """

    perception: str
    reasoning: str
    tool: str
    before_output: TraceOutput
    after_output: TraceOutput
    decision: str
    ts: int


# --------------------------------------------------------------------------- #
# Factory helper
# --------------------------------------------------------------------------- #


def trace_entry(
    *,
    perception: str,
    reasoning: str,
    tool: str,
    decision: str,
    ts: int,
    before_output: TraceOutput | None = None,
    after_output: TraceOutput | None = None,
) -> TraceEntry:
    """Build a :class:`TraceEntry`, defaulting absent outputs to :data:`UNAVAILABLE`.

    ``before_output`` / ``after_output`` default to the explicit
    :data:`UNAVAILABLE` sentinel when not supplied, so a step that has no vision
    output on one side records the honest sentinel rather than ``None`` or a
    fabricated value (3.3, 13.4). ``ts`` must be non-negative; it is the
    ordering key the :class:`~agent.store.Store` preserves.
    """
    if ts < 0:
        raise ValueError(f"trace entry ts must be non-negative, got {ts!r}")
    return TraceEntry(
        perception=perception,
        reasoning=reasoning,
        tool=tool,
        before_output=UNAVAILABLE if before_output is None else before_output,
        after_output=UNAVAILABLE if after_output is None else after_output,
        decision=decision,
        ts=int(ts),
    )
