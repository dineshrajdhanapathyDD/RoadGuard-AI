"""The ``save_inspection_finding`` and ``request_human_review`` agent tools.

Task 6.6. Two of the seven constrained Agent_Tools (design.md "The Seven Agent
Tools"). Unlike the *perception* tools (``analyze_frame`` / ``analyze_region``
in ``analysis.py``) and the *selection* tools (``select_additional_frames`` /
``compare_visual_evidence`` in ``selection.py``), these two are **recording**
tools: they do not call the :class:`~vision.pipeline.VisionPipeline` façade at
all. They take the agent's durable state (the :class:`~agent.store.Store` and/or
the :class:`~agent.state.InspectionState`) and the **real perception** the
decider reasoned over, and persist an :class:`~agent.store.InspectionFinding` or
enqueue a :class:`~agent.store.HumanReviewItem`.

Design mapping ("The Seven Agent Tools" table)
----------------------------------------------

| Tool                     | Input                                                     | Output                    | Failure behavior                                        |
| ``save_inspection_finding``| ``hazard_type``, ``evidence_ref``, ``confidence``, ``source_timestamp`` | persisted ``InspectionFinding`` | failed/flag-rejected evidence -> reject, no finding (10.2) |
| ``request_human_review``   | ``observation_ref``, ``evidence_ref``, ``reason``             | ``HumanReviewItem`` enqueued  | validation failure if refs missing                      |

Requirements wired here
-----------------------

``save_inspection_finding`` (Requirement 10)

* **10.1 - input schema.** The input (:class:`SaveInspectionFindingInput`)
  carries the hazard type, the referenced Evidence (``evidence_ref``), the
  confidence value, and the source timestamp. These are exactly the arguments
  :class:`~agent.decider_service.MockDecider` emits for
  :attr:`~agent.state.ToolName.SAVE_INSPECTION_FINDING`.
* **10.2 - the key safety property (ties to Competition 3).** IF the referenced
  Evidence is a *failed* or *Quality_Flag-rejected* Vision_Pipeline result, the
  save is rejected and **no** :class:`~agent.store.InspectionFinding` is
  persisted. The tool cannot enforce this from an opaque ``evidence_ref`` alone,
  so its contract takes the real :class:`~vision.result_schema.AnalysisResult`
  the decision was based on (``perception``) -- the same object
  ``decide(state, perception)`` reasoned over -- and checks it:

    - ``perception["status"] == "failed"`` -> reject (2.6, 6.3, Competition 3);
    - ``perception`` carries a **rejecting** :class:`~vision.result_schema.QualityFlag`
      (see :data:`REJECTING_QUALITY_CODES`) -> reject;
    - the referenced ``evidence_ref`` must actually belong to that successful
      perception (we never attach a finding to evidence the perception did not
      produce).

* **10.3 - persist with references to the real evidence.** On success the tool
  builds an :class:`~agent.store.InspectionFinding` whose ``evidence_ref``,
  ``hazard_type``, ``confidence``, ``source_timestamp_ms``, and ``method`` all
  come from the **real** perception (never fabricated), and persists it via
  :meth:`~agent.store.Store.put_finding` and mirrors it into
  :meth:`~agent.state.InspectionState.add_finding` so the loop's carried state
  and the durable store agree.

``request_human_review`` (Requirement 11)

* **11.1 - input schema.** The input (:class:`RequestHumanReviewInput`) carries
  the observation reference, the referenced Evidence (``evidence_ref``), and the
  escalation reason -- exactly the arguments the decider emits for
  :attr:`~agent.state.ToolName.REQUEST_HUMAN_REVIEW` (including on the
  failed-analysis escalation path).
* **11.2 - enqueue with Evidence + reason.** On success the tool enqueues a
  :class:`~agent.store.HumanReviewItem` (``status="open"``) carrying the
  observation ref, the evidence ref, and the reason, via
  :meth:`~agent.store.Store.enqueue_review` and
  :meth:`~agent.state.InspectionState.enqueue_review`.
* **11.3 - preference of escalation over save** for an unresolved flagged
  observation is a *decider/loop* policy (the :class:`MockDecider` already routes
  a quality-flagged or failed perception to ``request_human_review`` rather than
  ``save_inspection_finding``). This tool does not re-decide; it performs the
  enqueue honestly. Critically, escalation does **not** require successful
  evidence -- escalating a *failed* or *flagged* observation for a human to
  adjudicate is exactly the point, so ``request_human_review`` accepts any
  (well-formed) refs and never applies the 10.2 rejection rule.

The rejecting-quality-code policy (10.2)
----------------------------------------

A :class:`~vision.result_schema.QualityFlag` code is "rejecting" if a finding
backed by evidence carrying it would be dishonest:

* ``decode_failure`` -- the frame could not be decoded; any "detection" on it is
  not real. **Clearly rejecting.**
* ``low_confidence`` -- the analysis itself flagged the result as not
  trustworthy enough. The decider already routes low confidence *away* from
  save, but the tool still defends the invariant: a result the pipeline flagged
  as low-confidence must not be promoted to a finding. **Rejecting.**
* ``poor_input_quality`` -- the input was too poor to trust the output.
  **Rejecting.**

All three defined :data:`~vision.result_schema.QUALITY_CODES` are therefore
rejecting: a quality flag is, by definition, the pipeline telling us the
observation is uncertain or failed (2.5), and a finding is a *confirmed* hazard
(10.3). We treat the presence of **any** quality flag as rejecting rather than
cherry-picking codes, which keeps the invariant simple, conservative, and
testable. :data:`REJECTING_QUALITY_CODES` names the set explicitly so the policy
is auditable and so a future non-rejecting code (if one is ever added to the
vocabulary) must be opted out deliberately.

The tool contract (so the loop 6.8 can dispatch uniformly)
----------------------------------------------------------

The recording tools need the durable state, not the vision façade, so their
signature shape differs from the perception/selection tools (which take
``(pipeline, arguments)``). This is expected and documented. Both recording
tools have the uniform signature::

    tool(store, inspection_id, state, perception, arguments) -> <Result>

* ``store`` -- the :class:`~agent.store.Store` to persist into.
* ``inspection_id`` -- the current inspection's id (the store partition key).
* ``state`` -- the loop's :class:`~agent.state.InspectionState`; the tool
  mirrors the saved finding / enqueued review into it so carried state and the
  store agree.
* ``perception`` -- the **real** :class:`~vision.result_schema.AnalysisResult`
  the decider reasoned over this iteration. ``save_inspection_finding`` checks it
  to enforce 10.2; ``request_human_review`` records the observation it escalates.
* ``arguments`` -- the decider's :attr:`~agent.state.AgentDecision.arguments`.

The loop (6.8) dispatches by :class:`~agent.state.ToolName`: it calls the vision
tools with ``(pipeline, decision.arguments)`` and the recording tools with
``(store, inspection_id, state, perception, decision.arguments)``. Keeping the
two shapes explicit lets the loop route each group without this module depending
on the façade.

Honest validation at the boundary
----------------------------------

As with the other tool modules, bad input is parsed via ``from_arguments``
(raising :class:`~agent.tools.analysis.ToolInputError`) and converted into an
honest *rejected* / *failed* result **at the boundary** -- never raised into the
agent loop (6.3 spirit). Neither tool fabricates data: a save with missing /
invalid fields, or backed by failed/flagged evidence, is rejected and persists
nothing; an escalation with missing refs is a validation failure that enqueues
nothing.

This module performs no OpenCV, network, or AWS work on import and is importable
with no AWS credentials (Requirement 19.4, Competition 4).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from agent.state import InspectionState
from agent.store import HumanReviewItem, InspectionFinding, Store
from agent.tools.analysis import ToolInputError
from vision.result_schema import (
    METHODS,
    QUALITY_CODES,
    AnalysisResult,
    Method,
    QualityCode,
)

# --------------------------------------------------------------------------- #
# Rejecting-quality-code policy (10.2)
# --------------------------------------------------------------------------- #

#: Quality-flag codes whose presence on the backing evidence rejects a save
#: (10.2). Every defined quality code is rejecting: a quality flag is the
#: pipeline declaring the observation uncertain or failed (2.5), and a finding
#: is a *confirmed* hazard (10.3) -- the two are mutually exclusive. Named
#: explicitly (rather than "any flag") so the policy is auditable and a future
#: non-rejecting code must be removed from this set deliberately.
REJECTING_QUALITY_CODES: frozenset[QualityCode] = frozenset(QUALITY_CODES)  # type: ignore[arg-type]


def rejecting_quality_flags(perception: AnalysisResult) -> list[QualityCode]:
    """Return the rejecting quality-flag codes present on ``perception`` (10.2).

    A finding backed by evidence carrying any of these codes would be dishonest,
    so :func:`save_inspection_finding` refuses the save when this is non-empty.
    """
    flags = perception.get("quality_flags") or []
    return [
        f["code"]
        for f in flags
        if f.get("code") in REJECTING_QUALITY_CODES
    ]


# --------------------------------------------------------------------------- #
# save_inspection_finding (Requirement 10)
# --------------------------------------------------------------------------- #

#: Status channel for the save tool's result (the bare store write has none).
SaveStatus = Literal["saved", "rejected"]


@dataclass(frozen=True)
class SaveInspectionFindingInput:
    """Input schema for ``save_inspection_finding`` (10.1).

    Carries the hazard type, the referenced Evidence, the confidence, and the
    source timestamp -- exactly the arguments the decider emits. Note that the
    *rejection* decision (10.2) is driven by the backing ``perception``
    :class:`~vision.result_schema.AnalysisResult`, not by this input; this input
    only describes *what* the agent proposes to record.

    Attributes:
        hazard_type: The proposed hazard category (validated against the
            perception's real detections at save time, never trusted blindly).
        evidence_ref: The S3/evidence key of the annotated frame that supports
            the finding (10.3). Must be a non-empty string.
        confidence: The real detection confidence in ``[0.0, 1.0]``.
        source_timestamp: The source-frame timestamp (ms) linking the finding to
            its position in the video (preserved, never fabricated).
    """

    hazard_type: str
    evidence_ref: str
    confidence: float
    source_timestamp: int

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view for the trace/decision-argument layer."""
        return {
            "hazard_type": self.hazard_type,
            "evidence_ref": self.evidence_ref,
            "confidence": self.confidence,
            "source_timestamp": self.source_timestamp,
        }

    @classmethod
    def from_arguments(cls, arguments: Mapping[str, Any]) -> "SaveInspectionFindingInput":
        """Build an input from a loosely-typed :class:`AgentDecision.arguments` map.

        Raises :class:`~agent.tools.analysis.ToolInputError` on any missing or
        malformed field -- the tool validates its input rather than guessing
        (3.3). The error is caught at the tool boundary
        (:func:`save_inspection_finding`) and surfaced as an honest ``rejected``
        result, never raised into the agent loop.
        """
        if not isinstance(arguments, Mapping):
            raise ToolInputError(
                f"arguments must be a mapping, got {type(arguments).__name__}"
            )

        hazard_type = arguments.get("hazard_type")
        if not isinstance(hazard_type, str) or not hazard_type:
            raise ToolInputError(
                f"'hazard_type' must be a non-empty string, got {hazard_type!r}"
            )

        evidence_ref = arguments.get("evidence_ref")
        if not isinstance(evidence_ref, str) or not evidence_ref:
            raise ToolInputError(
                f"'evidence_ref' must be a non-empty string, got {evidence_ref!r}"
            )

        confidence = arguments.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ToolInputError(
                f"'confidence' must be a real number, got {confidence!r}"
            )
        confidence = float(confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ToolInputError(
                f"'confidence' must be in [0.0, 1.0], got {confidence!r}"
            )

        source_timestamp = arguments.get("source_timestamp")
        if source_timestamp is None:
            # Accept the schema's wire name too, for robustness.
            source_timestamp = arguments.get("source_timestamp_ms")
        if isinstance(source_timestamp, bool) or not isinstance(source_timestamp, int):
            raise ToolInputError(
                f"'source_timestamp' must be an int (ms), got {source_timestamp!r}"
            )
        if source_timestamp < 0:
            raise ToolInputError(
                f"'source_timestamp' must be >= 0, got {source_timestamp!r}"
            )

        return cls(
            hazard_type=hazard_type,
            evidence_ref=evidence_ref,
            confidence=confidence,
            source_timestamp=source_timestamp,
        )


@dataclass(frozen=True)
class SaveInspectionFindingResult:
    """Output schema for ``save_inspection_finding`` (10.2, 10.3).

    Carries an explicit status the bare store write cannot express. The
    invariant ``finding is not None iff status == "saved"`` is enforced in
    :meth:`__post_init__`: a rejected save persists nothing and carries no
    finding; a successful save carries the persisted
    :class:`~agent.store.InspectionFinding`.

    Attributes:
        status: ``"saved"`` when the finding was persisted (10.3), or
            ``"rejected"`` when the save was refused (10.2) or the input was
            invalid.
        finding: The persisted finding on success, else ``None``.
        rejection_reason: A non-empty explanation when ``status == "rejected"``,
            else ``None`` -- the rejection is surfaced honestly, not hidden.
    """

    status: SaveStatus
    finding: InspectionFinding | None = None
    rejection_reason: str | None = None

    def __post_init__(self) -> None:
        if self.status == "saved":
            if self.finding is None:
                raise ValueError("a 'saved' result must carry the persisted finding")
            if self.rejection_reason is not None:
                raise ValueError("a 'saved' result must not carry a rejection_reason")
        else:  # rejected
            if self.finding is not None:
                raise ValueError("a 'rejected' result must not carry a finding (10.2)")
            if not self.rejection_reason:
                raise ValueError("a 'rejected' result must carry a non-empty reason")

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view for the trace/persistence layer."""
        return {
            "status": self.status,
            "finding": dict(self.finding) if self.finding is not None else None,
            "rejection_reason": self.rejection_reason,
        }


def save_inspection_finding(
    store: Store,
    inspection_id: str,
    state: InspectionState,
    perception: AnalysisResult,
    arguments: Mapping[str, Any],
) -> SaveInspectionFindingResult:
    """``save_inspection_finding`` tool: record a confirmed hazard (Requirement 10).

    Contract (see module docstring): the tool takes the durable
    :class:`~agent.store.Store`, the ``inspection_id``, the loop's
    :class:`~agent.state.InspectionState`, the **real**
    :class:`~vision.result_schema.AnalysisResult` the decision was based on
    (``perception``), and the decider's ``arguments``.

    Enforcement order:

    1. Parse ``arguments`` into a :class:`SaveInspectionFindingInput` (10.1).
       Invalid input is an honest ``rejected`` result, never raised.
    2. **10.2 - reject failed/flag-rejected evidence.** If ``perception`` has
       ``status == "failed"`` or carries any rejecting
       :class:`~vision.result_schema.QualityFlag`
       (:data:`REJECTING_QUALITY_CODES`), the save is rejected and nothing is
       persisted. This is the invariant that satisfies Competition 3.
    3. The proposed ``evidence_ref`` must match the successful perception's own
       ``evidence_ref`` (or, if the perception has no evidence_ref, its
       ``frame_ref``). We never attach a finding to evidence the perception did
       not produce -- that would fabricate the link (3.3, 10.3).
    4. The proposed ``hazard_type`` must correspond to a real detection in the
       perception. We do not invent a hazard category the vision output did not
       report.
    5. **10.3 - persist with references to the real evidence.** Build an
       :class:`~agent.store.InspectionFinding` from the real perception fields
       (method from ``perception["method"]``, timestamp from the perception) and
       persist it via :meth:`Store.put_finding`, mirroring it into
       :meth:`InspectionState.add_finding`.

    Args:
        store: The durable store to persist the finding into.
        inspection_id: The current inspection id (store partition key).
        state: The loop's carried state; the saved finding is mirrored into it.
        perception: The real analysis result the decision reasoned over.
        arguments: The decider's :attr:`~agent.state.AgentDecision.arguments`.

    Returns:
        A :class:`SaveInspectionFindingResult`: ``status="saved"`` with the
        persisted finding on success (10.3), or ``status="rejected"`` with a
        reason and nothing persisted on any rejection (10.2) or invalid input.
    """
    # 1. input validation ---------------------------------------------------
    try:
        parsed = SaveInspectionFindingInput.from_arguments(arguments)
    except ToolInputError as exc:
        return SaveInspectionFindingResult(
            status="rejected",
            rejection_reason=f"save_inspection_finding input invalid: {exc}",
        )

    # 2. 10.2 - failed evidence can never become a finding (Competition 3) ---
    if perception.get("status") == "failed":
        reason = perception.get("failure_reason") or "analysis_failed"
        return SaveInspectionFindingResult(
            status="rejected",
            rejection_reason=(
                f"refusing to save a finding from a FAILED analysis of "
                f"{perception.get('frame_ref')!r} ({reason}); a failed vision result "
                f"can never become an inspection finding (10.2, Competition 3)"
            ),
        )

    # 2b. 10.2 - a rejecting Quality_Flag also blocks the save ---------------
    rejecting = rejecting_quality_flags(perception)
    if rejecting:
        return SaveInspectionFindingResult(
            status="rejected",
            rejection_reason=(
                f"refusing to save a finding from evidence carrying rejecting quality "
                f"flag(s) {rejecting}; a flagged observation is escalated, not saved (10.2)"
            ),
        )

    # 3. the evidence_ref must belong to this successful perception ----------
    perception_evidence = perception.get("evidence_ref") or perception.get("frame_ref")
    if not perception_evidence:
        return SaveInspectionFindingResult(
            status="rejected",
            rejection_reason=(
                "the backing perception has no resolvable evidence_ref/frame_ref; "
                "cannot attach a finding to real evidence (10.3)"
            ),
        )
    if parsed.evidence_ref != perception_evidence:
        return SaveInspectionFindingResult(
            status="rejected",
            rejection_reason=(
                f"evidence_ref {parsed.evidence_ref!r} does not match the backing "
                f"perception's evidence {perception_evidence!r}; refusing to attach a "
                f"finding to evidence the analysis did not produce (10.3)"
            ),
        )

    # 4. the hazard_type must correspond to a real detection -----------------
    detected_types = {
        d.get("hazard_type") for d in (perception.get("detections") or [])
    }
    if parsed.hazard_type not in detected_types:
        return SaveInspectionFindingResult(
            status="rejected",
            rejection_reason=(
                f"hazard_type {parsed.hazard_type!r} is not among the real detections "
                f"{sorted(t for t in detected_types if t)} on this evidence; refusing to "
                f"record a hazard the vision output did not report (3.3, 10.3)"
            ),
        )

    # 5. 10.3 - persist the finding from the REAL perception fields ----------
    method = perception.get("method")
    if method not in METHODS:  # defensive: perception method must be real
        return SaveInspectionFindingResult(
            status="rejected",
            rejection_reason=(
                f"backing perception has an invalid method {method!r}; cannot record "
                f"the detection path for the finding (4.3)"
            ),
        )

    finding = InspectionFinding(
        hazard_type=parsed.hazard_type,
        evidence_ref=parsed.evidence_ref,
        confidence=parsed.confidence,
        # Timestamp comes from the REAL perception, linking the finding to its
        # position in the video; we prefer the perception's own timestamp over
        # the (possibly stale) argument so the persisted link is never fabricated.
        source_timestamp_ms=int(perception.get("source_timestamp_ms", parsed.source_timestamp)),
        method=method,  # type: ignore[typeddict-item]  # narrowed by METHODS check
    )

    store.put_finding(inspection_id, finding)
    state.add_finding(dict(finding))

    return SaveInspectionFindingResult(status="saved", finding=finding)


# --------------------------------------------------------------------------- #
# request_human_review (Requirement 11)
# --------------------------------------------------------------------------- #

#: Status channel for the review tool's result.
ReviewStatus = Literal["enqueued", "failed"]


@dataclass(frozen=True)
class RequestHumanReviewInput:
    """Input schema for ``request_human_review`` (11.1).

    Carries the observation reference, the referenced Evidence, and the
    escalation reason -- exactly the arguments the decider emits (including on
    the failed-analysis escalation path, where the evidence may itself be a
    failed result; escalation is precisely how such observations reach a human).

    Attributes:
        observation_ref: Reference to the observation being escalated
            (non-empty). The decider uses the ``frame_ref``.
        evidence_ref: The referenced Evidence key. May be ``None`` when the
            backing analysis produced no evidence (e.g. a decode failure); we do
            not fabricate one -- it is recorded as the explicit "unavailable"
            sentinel so the human sees there is no annotated frame.
        reason: A non-empty escalation reason (e.g. a failure reason or a quality
            flag code).
    """

    observation_ref: str
    evidence_ref: str | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view for the trace/decision-argument layer."""
        return {
            "observation_ref": self.observation_ref,
            "evidence_ref": self.evidence_ref,
            "reason": self.reason,
        }

    @classmethod
    def from_arguments(cls, arguments: Mapping[str, Any]) -> "RequestHumanReviewInput":
        """Build an input from a loosely-typed :class:`AgentDecision.arguments` map.

        Raises :class:`~agent.tools.analysis.ToolInputError` if the observation
        reference or reason is missing/empty. ``evidence_ref`` is optional (an
        escalated failed observation may have no annotated frame) but, when
        present, must be a string. The error is caught at the tool boundary
        (:func:`request_human_review`) and surfaced as an honest ``failed``
        result, never raised into the agent loop.
        """
        if not isinstance(arguments, Mapping):
            raise ToolInputError(
                f"arguments must be a mapping, got {type(arguments).__name__}"
            )

        observation_ref = arguments.get("observation_ref")
        if not isinstance(observation_ref, str) or not observation_ref:
            raise ToolInputError(
                f"'observation_ref' must be a non-empty string, got {observation_ref!r}"
            )

        reason = arguments.get("reason")
        if not isinstance(reason, str) or not reason:
            raise ToolInputError(
                f"'reason' must be a non-empty string, got {reason!r}"
            )

        evidence_ref = arguments.get("evidence_ref")
        if evidence_ref is not None and not isinstance(evidence_ref, str):
            raise ToolInputError(
                f"'evidence_ref' must be a string or None, got {evidence_ref!r}"
            )
        # Treat an empty string as "no evidence" rather than a real reference.
        if evidence_ref == "":
            evidence_ref = None

        return cls(
            observation_ref=observation_ref,
            evidence_ref=evidence_ref,
            reason=reason,
        )


@dataclass(frozen=True)
class RequestHumanReviewResult:
    """Output schema for ``request_human_review`` (11.2).

    Carries an explicit status. The invariant ``item is not None iff status ==
    "enqueued"`` is enforced in :meth:`__post_init__`.

    Attributes:
        status: ``"enqueued"`` when the item was added to the Human_Review_Queue
            (11.2), or ``"failed"`` on invalid input.
        item: The enqueued :class:`~agent.store.HumanReviewItem` on success, else
            ``None``.
        failure_reason: A non-empty explanation when ``status == "failed"``, else
            ``None``.
    """

    status: ReviewStatus
    item: HumanReviewItem | None = None
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if self.status == "enqueued":
            if self.item is None:
                raise ValueError("an 'enqueued' result must carry the enqueued item")
            if self.failure_reason is not None:
                raise ValueError("an 'enqueued' result must not carry a failure_reason")
        else:  # failed
            if self.item is not None:
                raise ValueError("a 'failed' result must not carry an item")
            if not self.failure_reason:
                raise ValueError("a 'failed' result must carry a non-empty reason")

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view for the trace/persistence layer."""
        return {
            "status": self.status,
            "item": dict(self.item) if self.item is not None else None,
            "failure_reason": self.failure_reason,
        }


def request_human_review(
    store: Store,
    inspection_id: str,
    state: InspectionState,
    perception: AnalysisResult,
    arguments: Mapping[str, Any],
) -> RequestHumanReviewResult:
    """``request_human_review`` tool: escalate an observation (Requirement 11).

    Contract (see module docstring): same five-argument shape as
    :func:`save_inspection_finding`. Unlike the save tool, this tool does **not**
    apply the 10.2 rejection rule -- escalating a *failed* or *flagged*
    observation so a human can adjudicate it is the whole purpose of the Human
    Review Queue (11.3). It parses ``arguments`` into a
    :class:`RequestHumanReviewInput` (observation ref + evidence ref + reason,
    11.1) and enqueues a :class:`~agent.store.HumanReviewItem` (``status="open"``)
    via :meth:`Store.enqueue_review`, mirroring it into
    :meth:`InspectionState.enqueue_review` (11.2).

    ``perception`` is accepted for signature uniformity with the loop's
    recording-tool dispatch and to let the tool fall back to the real
    observation/evidence refs when the decider omits them; it does not gate the
    enqueue. Only missing/empty ``observation_ref`` or ``reason`` is a validation
    failure, which enqueues nothing and is returned honestly rather than raised.

    Args:
        store: The durable store to enqueue the review item into.
        inspection_id: The current inspection id (store partition key).
        state: The loop's carried state; the item is mirrored into it.
        perception: The real analysis result the escalation concerns (used only
            as a fallback source for the observation/evidence refs).
        arguments: The decider's :attr:`~agent.state.AgentDecision.arguments`.

    Returns:
        A :class:`RequestHumanReviewResult`: ``status="enqueued"`` with the item
        on success (11.2), or ``status="failed"`` with a reason and nothing
        enqueued on invalid input.
    """
    # Fall back to the real perception refs when the decider omitted them, so a
    # well-formed escalation is not blocked by a missing optional argument. We
    # never fabricate values -- these come from the real perception.
    merged: dict[str, Any] = dict(arguments) if isinstance(arguments, Mapping) else {}
    if not merged.get("observation_ref"):
        fallback_obs = perception.get("frame_ref")
        if fallback_obs:
            merged["observation_ref"] = fallback_obs
    if merged.get("evidence_ref") is None:
        merged["evidence_ref"] = perception.get("evidence_ref")

    try:
        parsed = RequestHumanReviewInput.from_arguments(merged)
    except ToolInputError as exc:
        return RequestHumanReviewResult(
            status="failed",
            failure_reason=f"request_human_review input invalid: {exc}",
        )

    item = HumanReviewItem(
        observation_ref=parsed.observation_ref,
        # Record the explicit "unavailable" sentinel when there is no annotated
        # frame, rather than fabricating a reference (3.3, 13.4).
        evidence_ref=parsed.evidence_ref if parsed.evidence_ref else "unavailable",
        reason=parsed.reason,
        status="open",
    )

    store.enqueue_review(inspection_id, item)
    state.enqueue_review(dict(item))

    return RequestHumanReviewResult(status="enqueued", item=item)


# --------------------------------------------------------------------------- #
# Self-validation (import-time smoke check against the in-memory store)
# --------------------------------------------------------------------------- #


def _self_check() -> None:
    """Exercise both recording tools against a real :class:`InMemoryStore`.

    Minimal real guard (no mocks), not the optional property/unit suite
    (tasks 6.12 / 6.14 / 6.17). Runs under ``python -m agent.tools.recording``.

    Asserts:

    * ``save_inspection_finding`` from a **successful** perception persists the
      finding with the real evidence/method/timestamp (10.1, 10.3);
    * from a **failed** perception it is rejected and persists nothing (10.2,
      Competition 3);
    * from a successful perception carrying a rejecting ``Quality_Flag`` it is
      rejected and persists nothing (10.2);
    * a mismatched evidence_ref or hazard_type is rejected (10.3, 3.3);
    * ``request_human_review`` enqueues an item with evidence + reason (11.1,
      11.2), including for a failed observation (11.3 spirit).
    """
    from agent.state import InspectionState
    from agent.store import InMemoryStore
    from vision.result_schema import (
        detection,
        failed_result,
        pixel_measurement,
        quality_flag,
        success_result,
    )

    inspection_id = "insp-recording-selfcheck"
    m = pixel_measurement(12, 24)

    # -- successful perception with a strong, clean pothole detection -------- #
    good = success_result(
        method="model",
        frame_ref="frame_7",
        source_timestamp_ms=1500,
        detections=[detection("pothole", (4, 5, 12, 24), 0.91, m)],
        evidence_ref="evidence/insp/frame_7.png",
    )

    store = InMemoryStore()
    state = InspectionState()

    saved = save_inspection_finding(
        store,
        inspection_id,
        state,
        good,
        {
            "hazard_type": "pothole",
            "evidence_ref": "evidence/insp/frame_7.png",
            "confidence": 0.91,
            "source_timestamp": 1500,
        },
    )
    assert saved.status == "saved", saved
    assert saved.finding is not None
    assert saved.finding["evidence_ref"] == "evidence/insp/frame_7.png"
    assert saved.finding["method"] == "model", "method must come from the real perception (4.3)"
    assert saved.finding["source_timestamp_ms"] == 1500
    # Persisted in BOTH the store and the carried state.
    bundle = store.get_inspection_bundle(inspection_id)
    assert len(bundle["findings"]) == 1, bundle["findings"]
    assert len(state.findings) == 1

    # -- FAILED perception -> reject, persist NOTHING (10.2, Competition 3) --- #
    bad = failed_result(
        method="classical",
        frame_ref="frame_9",
        source_timestamp_ms=1800,
        failure_reason="decode_failure",
    )
    store2 = InMemoryStore()
    state2 = InspectionState()
    rejected = save_inspection_finding(
        store2,
        inspection_id,
        state2,
        bad,
        {
            "hazard_type": "pothole",
            "evidence_ref": "frame_9",
            "confidence": 0.95,
            "source_timestamp": 1800,
        },
    )
    assert rejected.status == "rejected", rejected
    assert rejected.finding is None
    assert rejected.rejection_reason
    assert store2.get_inspection_bundle(inspection_id)["findings"] == []
    assert state2.findings == []

    # -- successful but Quality_Flag-rejected perception -> reject (10.2) ----- #
    flagged = success_result(
        method="classical",
        frame_ref="frame_11",
        source_timestamp_ms=2000,
        detections=[detection("pothole", (1, 1, 10, 10), 0.88, m)],
        quality_flags=[quality_flag("low_confidence", "borderline texture response")],
        evidence_ref="evidence/insp/frame_11.png",
    )
    store3 = InMemoryStore()
    state3 = InspectionState()
    flag_rejected = save_inspection_finding(
        store3,
        inspection_id,
        state3,
        flagged,
        {
            "hazard_type": "pothole",
            "evidence_ref": "evidence/insp/frame_11.png",
            "confidence": 0.88,
            "source_timestamp": 2000,
        },
    )
    assert flag_rejected.status == "rejected", flag_rejected
    assert store3.get_inspection_bundle(inspection_id)["findings"] == []

    # -- mismatched evidence_ref -> reject (10.3) ----------------------------- #
    store4 = InMemoryStore()
    state4 = InspectionState()
    mismatch = save_inspection_finding(
        store4,
        inspection_id,
        state4,
        good,
        {
            "hazard_type": "pothole",
            "evidence_ref": "evidence/insp/SOME_OTHER_frame.png",
            "confidence": 0.91,
            "source_timestamp": 1500,
        },
    )
    assert mismatch.status == "rejected", mismatch
    assert store4.get_inspection_bundle(inspection_id)["findings"] == []

    # -- hazard_type not among real detections -> reject (3.3, 10.3) ---------- #
    store5 = InMemoryStore()
    state5 = InspectionState()
    wrong_type = save_inspection_finding(
        store5,
        inspection_id,
        state5,
        good,
        {
            "hazard_type": "debris",  # perception detected a pothole, not debris
            "evidence_ref": "evidence/insp/frame_7.png",
            "confidence": 0.91,
            "source_timestamp": 1500,
        },
    )
    assert wrong_type.status == "rejected", wrong_type
    assert store5.get_inspection_bundle(inspection_id)["findings"] == []

    # -- invalid input -> rejected, never raised ------------------------------ #
    store6 = InMemoryStore()
    state6 = InspectionState()
    invalid = save_inspection_finding(store6, inspection_id, state6, good, {})
    assert invalid.status == "rejected"
    assert invalid.finding is None

    # -- request_human_review enqueues with evidence + reason (11.1, 11.2) ---- #
    store7 = InMemoryStore()
    state7 = InspectionState()
    enq = request_human_review(
        store7,
        inspection_id,
        state7,
        flagged,
        {
            "observation_ref": "frame_11",
            "evidence_ref": "evidence/insp/frame_11.png",
            "reason": "low_confidence",
        },
    )
    assert enq.status == "enqueued", enq
    assert enq.item is not None
    assert enq.item["evidence_ref"] == "evidence/insp/frame_11.png"
    assert enq.item["reason"] == "low_confidence"
    assert enq.item["status"] == "open"
    rb = store7.get_inspection_bundle(inspection_id)
    assert len(rb["review_queue"]) == 1
    assert len(state7.review_queue) == 1

    # -- escalation of a FAILED observation (no evidence) still enqueues (11.3) #
    store8 = InMemoryStore()
    state8 = InspectionState()
    enq_failed = request_human_review(
        store8,
        inspection_id,
        state8,
        bad,  # failed perception: no evidence_ref
        {"observation_ref": "frame_9", "reason": "decode_failure"},
    )
    assert enq_failed.status == "enqueued", enq_failed
    assert enq_failed.item is not None
    # No annotated frame -> explicit unavailable sentinel, not a fabricated ref.
    assert enq_failed.item["evidence_ref"] == "unavailable"

    # -- missing observation_ref/reason -> validation failure, enqueues nothing #
    store9 = InMemoryStore()
    state9 = InspectionState()
    # perception has no frame_ref fallback usable -> missing reason still fails.
    bad_review = request_human_review(
        store9, inspection_id, state9, good, {"observation_ref": "frame_7"}
    )
    assert bad_review.status == "failed", bad_review
    assert store9.get_inspection_bundle(inspection_id)["review_queue"] == []


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    _self_check()
    print(
        "[agent.tools.recording] OK: save_inspection_finding persists a finding "
        "from a successful perception (10.3) and REJECTS failed/flag-rejected or "
        "mismatched evidence, persisting nothing (10.2, Competition 3); "
        "request_human_review enqueues an item with evidence + reason (11.1, 11.2), "
        "including for failed observations (11.3)."
    )
