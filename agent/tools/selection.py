"""The ``select_additional_frames`` and ``compare_visual_evidence`` agent tools.

Task 6.5. Two of the seven constrained Agent_Tools (design.md "The Seven Agent
Tools"), both thin, honest adapters over the already-built, independently tested
:class:`~vision.pipeline.VisionPipeline` façade. Neither reimplements any vision
logic; each maps a tool-level input to the façade call, then wraps the façade's
structured ``result_schema`` output in a small tool-level result carrying an
explicit status.

The façade is the single seam the agent tools call
(design "Vision Pipeline Interface"):

    def select_frames(self, criteria: SelectionCriteria, count: int) -> list[FrameRef]: ...
    def compare_evidence(self, evidence_refs: list[str]) -> ComparisonResult: ...

These two tools sit directly on top of those two methods.

``select_additional_frames`` (Requirement 7)
---------------------------------------------

* **7.1 - accepts selection criteria and the number of frames requested.** The
  input is a :class:`SelectAdditionalFramesInput` describing the selection
  strategy (evenly-spaced over the whole timeline, or a ``[start_ms, end_ms]``
  time range) and ``count``. It maps 1:1 to the façade's
  :class:`~vision.pipeline.SelectionCriteria` + ``count``.
* **7.2 - returns an ordered list of frame references with source timestamps.**
  The façade already orders by ``source_timestamp_ms`` ascending and returns
  :class:`~vision.result_schema.FrameRef` records (``frame_ref`` +
  ``source_timestamp_ms``); the tool passes that ordering through unchanged.
* **7.3 - no frames match -> empty list + a "no matching frames" status.** Because
  the façade returns ``list[FrameRef]`` (no status channel), this tool wraps the
  list in a :class:`SelectAdditionalFramesResult` carrying an explicit status:
  ``"ok"`` when at least one frame was selected, ``"no_matching_frames"`` when
  the list is empty (including a ``count <= 0`` request or a time range that
  matches nothing). The frames list is empty exactly when the status is
  ``no_matching_frames`` -- the two are kept consistent by construction.

``compare_visual_evidence`` (Requirement 9)
-------------------------------------------

* **9.1 - accepts input referencing two or more Evidence items.** The input is a
  :class:`CompareVisualEvidenceInput` carrying ``evidence_refs`` (the evidence
  keys produced by ``analyze_frame`` / ``analyze_region``).
* **9.2 - returns a comparison result describing agreement/disagreement.** The
  tool returns the façade's :class:`~vision.result_schema.ComparisonResult`
  verbatim; on success its ``agreement`` field is ``"agreement"`` or
  ``"disagreement"``, derived from real re-analysis (never fabricated).
* **9.3 - fewer than two valid Evidence items -> validation failure.** The façade
  already returns a :func:`~vision.result_schema.comparison_failure`
  (``status="failed"``, ``agreement="unavailable"``) when fewer than two evidence
  items resolve and re-analyze successfully. This tool validates the *input*
  honestly before delegating: fewer than two distinct, non-empty refs supplied is
  itself a validation failure (we do not fabricate a second item), and anything
  with >= 2 refs is handed to the façade, whose re-analysis has the final say on
  how many are actually *valid*.

Tool boundary + honest validation
----------------------------------

Both tools follow the package convention (see ``analysis.py``): the public
function takes ``(pipeline, arguments)`` where ``arguments`` is the decider's
free-form :attr:`~agent.state.AgentDecision.arguments` mapping, parses it into a
typed input via ``from_arguments`` (raising
:class:`~agent.tools.analysis.ToolInputError` on bad input), and converts any
validation error into an honest failed result **at the boundary** -- it is never
raised into the agent loop, so the loop is driven by result status (6.3).

Neither tool fabricates data to satisfy a request: a bad ``count`` or an
unconstructable time range surfaces as a ``no_matching_frames`` result rather
than a guessed selection, and an under-referenced comparison surfaces as a
validation failure rather than an invented second evidence item (3.3).

Offline, no AWS
---------------

The tools are pure adapters over the façade, which runs real OpenCV 5 against a
local clip with a local evidence store and no AWS credentials (Requirement 19.4,
Competition 4). This module performs no network or AWS work.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from agent.tools.analysis import ToolInputError
from vision.pipeline import (
    SelectionCriteria,
    SelectionStrategy,
    VisionPipelineProtocol,
)
from vision.result_schema import ComparisonResult, FrameRef


# --------------------------------------------------------------------------- #
# select_additional_frames (Requirement 7)
# --------------------------------------------------------------------------- #

#: Status channel the façade's ``list[FrameRef]`` lacks (7.3).
SelectStatus = Literal["ok", "no_matching_frames"]


@dataclass(frozen=True)
class SelectAdditionalFramesInput:
    """Input schema for ``select_additional_frames`` (7.1).

    Describes *which* additional frames to select and *how many*, mapping 1:1 to
    the façade's :class:`~vision.pipeline.SelectionCriteria` + ``count``.

    Attributes:
        count: Number of frames requested. ``<= 0`` is accepted but yields a
            ``no_matching_frames`` result rather than an error (honest: a request
            for zero frames matches zero frames).
        strategy: ``"evenly_spaced"`` (default) spreads ``count`` frames evenly
            across the whole timeline; ``"time_range"`` restricts to frames whose
            ``source_timestamp_ms`` is within ``[start_ms, end_ms]`` inclusive.
        start_ms / end_ms: Inclusive bounds required for the ``time_range``
            strategy; ignored for ``evenly_spaced``.
    """

    count: int
    strategy: Literal["evenly_spaced", "time_range"] = "evenly_spaced"
    start_ms: int | None = None
    end_ms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view for the trace/decision-argument layer."""
        return {
            "count": self.count,
            "strategy": self.strategy,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
        }

    @classmethod
    def from_arguments(
        cls, arguments: Mapping[str, Any]
    ) -> "SelectAdditionalFramesInput":
        """Build an input from a loosely-typed :class:`AgentDecision.arguments` map.

        Raises :class:`~agent.tools.analysis.ToolInputError` if ``count`` is
        missing or not an integer, or if the strategy string is outside the
        closed vocabulary -- the tool validates its input rather than guessing
        (3.3). The error is caught at the tool boundary
        (:func:`select_additional_frames`) and surfaced as an honest
        ``no_matching_frames`` result, never raised into the agent loop.
        """
        if "count" not in arguments:
            raise ToolInputError("select_additional_frames input requires 'count'")
        count = arguments["count"]
        if isinstance(count, bool) or not isinstance(count, int):
            raise ToolInputError(f"'count' must be an int, got {count!r}")
        strategy = arguments.get("strategy", "evenly_spaced")
        if strategy not in ("evenly_spaced", "time_range"):
            raise ToolInputError(
                f"'strategy' must be 'evenly_spaced' or 'time_range', got {strategy!r}"
            )
        start_ms = arguments.get("start_ms")
        end_ms = arguments.get("end_ms")
        if start_ms is not None and (isinstance(start_ms, bool) or not isinstance(start_ms, int)):
            raise ToolInputError(f"'start_ms' must be an int or None, got {start_ms!r}")
        if end_ms is not None and (isinstance(end_ms, bool) or not isinstance(end_ms, int)):
            raise ToolInputError(f"'end_ms' must be an int or None, got {end_ms!r}")
        return cls(
            count=count,
            strategy=strategy,
            start_ms=start_ms,
            end_ms=end_ms,
        )


@dataclass(frozen=True)
class SelectAdditionalFramesResult:
    """Output schema for ``select_additional_frames`` (7.2, 7.3).

    Wraps the façade's ordered ``list[FrameRef]`` with an explicit status the
    bare list cannot express. The invariant ``frames == [] iff status ==
    "no_matching_frames"`` is enforced in :meth:`__post_init__`.

    Attributes:
        status: ``"ok"`` when at least one frame was selected, else
            ``"no_matching_frames"`` (7.3).
        frames: Ordered (ascending ``source_timestamp_ms``) frame references with
            their source timestamps; empty exactly when ``status`` is
            ``no_matching_frames`` (7.2).
        requested_count: The ``count`` that was asked for (for the trace).
    """

    status: SelectStatus
    frames: list[FrameRef] = field(default_factory=list)
    requested_count: int = 0

    def __post_init__(self) -> None:
        empty = len(self.frames) == 0
        if empty and self.status != "no_matching_frames":
            raise ValueError("empty frames must carry status 'no_matching_frames' (7.3)")
        if not empty and self.status != "ok":
            raise ValueError("non-empty frames must carry status 'ok'")

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view for the trace/persistence layer."""
        return {
            "status": self.status,
            "requested_count": self.requested_count,
            "frames": [dict(f) for f in self.frames],
        }


def select_additional_frames(
    pipeline: VisionPipelineProtocol,
    arguments: Mapping[str, Any],
) -> SelectAdditionalFramesResult:
    """``select_additional_frames`` tool: select additional frames (Requirement 7).

    Validates ``arguments`` into a :class:`SelectAdditionalFramesInput`
    (selection criteria + ``count``, 7.1), maps it to the façade's
    :class:`~vision.pipeline.SelectionCriteria` + ``count``, delegates to
    :meth:`~vision.pipeline.VisionPipeline.select_frames`, then wraps the ordered
    result in a :class:`SelectAdditionalFramesResult` with an explicit status.

    Consistent with the package convention (see ``analysis.py``), invalid
    arguments do not reach the façade and are not raised into the loop: a bad
    ``count`` / ``strategy`` / bounds is an unconstructable (not guessed)
    selection and is returned as an honest ``no_matching_frames`` result (7.3,
    3.3), so the agent reasons over result status rather than catching
    exceptions.

    Args:
        pipeline: The injected vision façade (any
            :class:`~vision.pipeline.VisionPipelineProtocol`).
        arguments: The decider's :attr:`~agent.state.AgentDecision.arguments`.

    Returns:
        A :class:`SelectAdditionalFramesResult`. ``status="ok"`` with an ordered,
        timestamped, non-empty ``frames`` list when frames match (7.2), or
        ``status="no_matching_frames"`` with an empty list when nothing matches --
        including invalid input, ``count <= 0``, or a time range that selects
        nothing (7.3).
    """
    try:
        tool_input = SelectAdditionalFramesInput.from_arguments(arguments)
    except ToolInputError:
        # Invalid input is not a guessed selection; it matches zero frames.
        requested = arguments.get("count") if isinstance(arguments, Mapping) else None
        return SelectAdditionalFramesResult(
            status="no_matching_frames",
            frames=[],
            requested_count=requested if isinstance(requested, int) else 0,
        )

    # count <= 0 matches zero frames by definition; surface it honestly rather
    # than calling the façade (which also returns [] for count <= 0). (7.3)
    if tool_input.count <= 0:
        return SelectAdditionalFramesResult(
            status="no_matching_frames",
            frames=[],
            requested_count=tool_input.count,
        )

    # Build the façade criteria. A malformed time range (missing/inverted bounds)
    # is an unconstructable selection, not a guessed one -> no matching frames.
    try:
        if tool_input.strategy == "time_range":
            criteria = SelectionCriteria(
                strategy=SelectionStrategy.TIME_RANGE,
                start_ms=tool_input.start_ms,
                end_ms=tool_input.end_ms,
            )
        else:
            criteria = SelectionCriteria(strategy=SelectionStrategy.EVENLY_SPACED)
    except ValueError:
        return SelectAdditionalFramesResult(
            status="no_matching_frames",
            frames=[],
            requested_count=tool_input.count,
        )

    frames = pipeline.select_frames(criteria, tool_input.count)
    if not frames:
        return SelectAdditionalFramesResult(
            status="no_matching_frames",
            frames=[],
            requested_count=tool_input.count,
        )
    return SelectAdditionalFramesResult(
        status="ok",
        frames=list(frames),
        requested_count=tool_input.count,
    )


# --------------------------------------------------------------------------- #
# compare_visual_evidence (Requirement 9)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CompareVisualEvidenceInput:
    """Input schema for ``compare_visual_evidence`` (9.1).

    References two or more Evidence items by the evidence keys that
    ``analyze_frame`` / ``analyze_region`` produced.

    Attributes:
        evidence_refs: The evidence keys to compare. The tool requires at least
            two distinct, non-empty refs (9.3); the façade's re-analysis then has
            the final say on how many are actually valid.
    """

    evidence_refs: list[str]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view for the trace/decision-argument layer."""
        return {"evidence_refs": list(self.evidence_refs)}

    @classmethod
    def from_arguments(
        cls, arguments: Mapping[str, Any]
    ) -> "CompareVisualEvidenceInput":
        """Build an input from a loosely-typed :class:`AgentDecision.arguments` map.

        Raises :class:`~agent.tools.analysis.ToolInputError` if ``evidence_refs``
        is missing or not a list of strings -- the tool validates its input
        rather than guessing (3.3). The error is caught at the tool boundary
        (:func:`compare_visual_evidence`) and surfaced as an honest comparison
        validation failure, never raised into the agent loop.
        """
        if "evidence_refs" not in arguments:
            raise ToolInputError("compare_visual_evidence input requires 'evidence_refs'")
        refs = arguments["evidence_refs"]
        if not isinstance(refs, (list, tuple)):
            raise ToolInputError(
                f"'evidence_refs' must be a list, got {type(refs).__name__}"
            )
        if not all(isinstance(r, str) for r in refs):
            raise ToolInputError("'evidence_refs' must contain only strings")
        return cls(evidence_refs=list(refs))


def compare_visual_evidence(
    pipeline: VisionPipelineProtocol,
    arguments: Mapping[str, Any],
) -> ComparisonResult:
    """``compare_visual_evidence`` tool: compare two or more Evidence items (Requirement 9).

    Validates ``arguments`` into a :class:`CompareVisualEvidenceInput`
    (``evidence_refs``, 9.1), then delegates to
    :meth:`~vision.pipeline.VisionPipeline.compare_evidence`, whose verdict is
    derived from real re-analysis of the referenced evidence (never fabricated).

    Input validation (9.3): invalid arguments, or fewer than two **distinct,
    non-empty** refs supplied, is itself a validation failure -- the tool never
    invents a second evidence item to reach the minimum, and never raises into
    the loop. When at least two such refs are supplied they are handed to the
    façade, which re-analyzes each and itself returns a validation failure if
    fewer than two turn out to be *valid* (resolvable + re-analyzable).

    Args:
        pipeline: The injected vision façade to delegate to.
        arguments: The decider's :attr:`~agent.state.AgentDecision.arguments`.

    Returns:
        A :class:`~vision.result_schema.ComparisonResult`: ``status="success"``
        with ``agreement`` in ``{"agreement", "disagreement"}`` when the
        comparison is meaningful (9.2), or ``status="failed"`` with
        ``agreement="unavailable"`` and a ``failure_reason`` when the input is
        invalid or fewer than two valid evidence items are referenced (9.3).
    """
    from vision.result_schema import comparison_failure

    try:
        tool_input = CompareVisualEvidenceInput.from_arguments(arguments)
    except ToolInputError as exc:
        supplied = arguments.get("evidence_refs") if isinstance(arguments, Mapping) else None
        supplied_list = list(supplied) if isinstance(supplied, (list, tuple)) else []
        return comparison_failure(
            [r for r in supplied_list if isinstance(r, str)],
            failure_reason=f"compare_visual_evidence input invalid: {exc} (9.3)",
        )

    # De-duplicate while preserving order; a ref compared against itself is not
    # two independent items, and blanks are not references.
    distinct: list[str] = []
    for ref in tool_input.evidence_refs:
        if ref and ref not in distinct:
            distinct.append(ref)

    if len(distinct) < 2:
        # Honest input-level validation failure: we do not fabricate a second
        # item to satisfy the request (9.3, 3.3).
        return comparison_failure(
            list(tool_input.evidence_refs),
            failure_reason=(
                "compare_visual_evidence requires at least two distinct, non-empty "
                f"evidence references; got {len(distinct)} distinct of "
                f"{len(tool_input.evidence_refs)} supplied (9.3)"
            ),
        )

    # Delegate to the façade; its real re-analysis decides how many are valid and
    # whether the evidence agrees or disagrees (9.2, 9.3).
    return pipeline.compare_evidence(distinct)


# --------------------------------------------------------------------------- #
# Self-validation (import-time smoke check: real façade over a sample clip)
# --------------------------------------------------------------------------- #


def _self_check() -> None:
    """Exercise both tools against a real :class:`VisionPipeline`. Runs under ``python -m``.

    Minimal real-OpenCV guard (no mocks), not the optional property/unit suite
    (tasks 6.15 / 6.17). Decodes a generated clip and asserts:

    * ``select_additional_frames`` returns ordered, timestamped frames with
      ``status="ok"`` (7.2); ``count=0`` and an empty time range yield
      ``status="no_matching_frames"`` with an empty list (7.3).
    * ``compare_visual_evidence`` with < 2 refs is a validation failure (9.3) and
      with >= 2 valid evidence refs returns a ``ComparisonResult`` (9.2).
    """
    import tempfile
    from pathlib import Path

    from vision.annotation import LocalEvidenceStore
    from vision.frame_sampler import SamplingPolicy
    from vision.pipeline import VisionPipeline
    from vision.samples.make_samples import make_samples

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        clips = make_samples(tmp_path)
        pothole_clip = next(c for c in clips if "pothole" in c.name)

        pipeline = VisionPipeline(
            str(pothole_clip),
            inspection_id="insp-tools-selfcheck",
            policy=SamplingPolicy.every_n_frames(5),
            store=LocalEvidenceStore(tmp_path / "artifacts"),
        )

        # -- select_additional_frames (7.1, 7.2, 7.3) ---------------------- #
        ok = select_additional_frames(pipeline, {"count": 3})
        assert ok.status == "ok", ok
        assert ok.frames, "expected frames from a real clip"
        assert len(ok.frames) <= 3
        ts = [f["source_timestamp_ms"] for f in ok.frames]
        assert ts == sorted(ts), f"frames not timestamp-ordered: {ts}"
        assert ok.requested_count == 3
        # Deterministic: identical input -> identical ordered selection.
        again = select_additional_frames(pipeline, {"count": 3})
        assert again.frames == ok.frames

        # count <= 0 -> no matching frames, empty list (7.3).
        zero = select_additional_frames(pipeline, {"count": 0})
        assert zero.status == "no_matching_frames"
        assert zero.frames == []

        # Invalid input (missing count) -> no matching frames, never raised (7.3, 3.3).
        bad = select_additional_frames(pipeline, {})
        assert bad.status == "no_matching_frames"
        assert bad.frames == []

        # A time range that matches nothing (far future) -> no matching frames.
        empty_range = select_additional_frames(
            pipeline,
            {
                "count": 5,
                "strategy": "time_range",
                "start_ms": 10_000_000,
                "end_ms": 10_000_001,
            },
        )
        assert empty_range.status == "no_matching_frames"
        assert empty_range.frames == []

        # A valid time range covering the whole clip -> ok, ordered.
        wide_range = select_additional_frames(
            pipeline,
            {"count": 2, "strategy": "time_range", "start_ms": 0, "end_ms": 10_000_000},
        )
        assert wide_range.status == "ok"
        assert wide_range.frames

        # from_arguments parses a loosely-typed decision argument map (7.1).
        parsed = SelectAdditionalFramesInput.from_arguments(
            {"count": 2, "strategy": "evenly_spaced"}
        )
        assert parsed.count == 2 and parsed.strategy == "evenly_spaced"

        # -- compare_visual_evidence (9.1, 9.2, 9.3) ----------------------- #
        first_ref = ok.frames[0]["frame_ref"]
        second_ref = ok.frames[1]["frame_ref"]
        res1 = pipeline.analyze_frame(first_ref, "classical")
        res2 = pipeline.analyze_frame(second_ref, "classical")
        assert res1["evidence_ref"] and res2["evidence_ref"]

        # < 2 distinct refs -> validation failure (9.3).
        one = compare_visual_evidence(
            pipeline, {"evidence_refs": [res1["evidence_ref"]]}
        )
        assert one["status"] == "failed"
        assert one["agreement"] == "unavailable"
        assert one["failure_reason"]

        # A single ref duplicated is still < 2 distinct items -> failure.
        dup = compare_visual_evidence(
            pipeline,
            {"evidence_refs": [res1["evidence_ref"], res1["evidence_ref"]]},
        )
        assert dup["status"] == "failed"

        # none -> validation failure.
        none = compare_visual_evidence(pipeline, {"evidence_refs": []})
        assert none["status"] == "failed"

        # Invalid input (missing key) -> validation failure, never raised (9.3, 3.3).
        invalid = compare_visual_evidence(pipeline, {})
        assert invalid["status"] == "failed"
        assert invalid["failure_reason"]

        # >= 2 valid evidence refs -> an honest comparison verdict (9.2).
        cmp = compare_visual_evidence(
            pipeline,
            {"evidence_refs": [res1["evidence_ref"], res2["evidence_ref"]]},
        )
        assert cmp["status"] == "success"
        assert cmp["agreement"] in ("agreement", "disagreement")
        assert len(cmp["evidence_refs"]) == 2

        # from_arguments parses a loosely-typed decision argument map (9.1).
        parsed_cmp = CompareVisualEvidenceInput.from_arguments(
            {"evidence_refs": [res1["evidence_ref"], res2["evidence_ref"]]}
        )
        assert parsed_cmp.evidence_refs == [res1["evidence_ref"], res2["evidence_ref"]]


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    _self_check()
    print(
        "[agent.tools.selection] OK: select_additional_frames returns ordered, "
        "timestamped frames (7.2) with a no_matching_frames status (7.3); "
        "compare_visual_evidence delegates to the façade, failing validation for "
        "fewer than two valid evidence items (9.3) and returning an honest "
        "agreement/disagreement verdict otherwise (9.2)."
    )
