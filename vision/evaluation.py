"""Detection evaluation metrics for RoadGuard AI (task 16.1).

Computes precision, recall, F1, and IoU for road-damage detections against
labeled ground truth (Requirement 15.1). Pure and deterministic - no OpenCV, no
network, no AWS - so it runs in the offline backbone and in tests.

Matching model: a predicted box is a true positive if it overlaps a
ground-truth box of the SAME hazard type with IoU >= ``iou_threshold``. Matching
is greedy by descending IoU and one-to-one (each GT box matches at most one
prediction), which is the standard PASCAL-VOC-style association. Unmatched
predictions are false positives; unmatched GT boxes are false negatives.

Known-answer behavior (so empty/degenerate sets are defined, not fabricated):
  * no predictions AND no ground truth  -> precision=recall=f1=1.0 (nothing to
    find, nothing wrongly reported: a perfect vacuous result);
  * predictions but no ground truth     -> precision=0.0, recall=1.0 (all FPs);
  * ground truth but no predictions     -> precision=1.0 (vacuous), recall=0.0;
  * identical sets                       -> 1.0 across the board;
  * disjoint sets                        -> 0.0 precision/recall/f1.
All metrics are bounded in [0, 1]. mean IoU is over matched pairs (``None`` when
there are no matches, reported explicitly rather than as a fabricated 0.0).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

# A labeled box: (x, y, w, h) in pixels plus a hazard-type label.
Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class LabeledBox:
    """A ground-truth or predicted box with its hazard-type label."""

    hazard_type: str
    bbox_px: Box


@dataclass(frozen=True)
class DetectionMetrics:
    """Detection-quality metrics over one evaluation set (all in [0, 1])."""

    precision: float
    recall: float
    f1: float
    #: Mean IoU over matched (TP) pairs; ``None`` when there are no matches.
    mean_iou: float | None
    true_positives: int
    false_positives: int
    false_negatives: int


def iou(a: Box, b: Box) -> float:
    """Intersection-over-union of two (x, y, w, h) pixel boxes, in [0, 1]."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def evaluate_detections(
    predictions: Sequence[LabeledBox],
    ground_truth: Sequence[LabeledBox],
    *,
    iou_threshold: float = 0.5,
) -> DetectionMetrics:
    """Compute precision/recall/F1/mean-IoU for predictions vs. ground truth (15.1).

    Greedy one-to-one matching by descending IoU among same-hazard-type pairs
    above ``iou_threshold``. See the module docstring for the known-answer
    behavior on empty/degenerate sets.
    """
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError(f"iou_threshold must be in [0, 1], got {iou_threshold}")

    n_pred, n_gt = len(predictions), len(ground_truth)

    # Known answers for empty sets (defined, not fabricated).
    if n_pred == 0 and n_gt == 0:
        return DetectionMetrics(1.0, 1.0, 1.0, None, 0, 0, 0)
    if n_gt == 0:  # all predictions are false positives
        return DetectionMetrics(0.0, 1.0, 0.0, None, 0, n_pred, 0)
    if n_pred == 0:  # everything missed
        return DetectionMetrics(1.0, 0.0, 0.0, None, 0, 0, n_gt)

    # Candidate matches: same hazard type, IoU >= threshold.
    candidates: list[tuple[float, int, int]] = []
    for pi, p in enumerate(predictions):
        for gi, g in enumerate(ground_truth):
            if p.hazard_type != g.hazard_type:
                continue
            score = iou(p.bbox_px, g.bbox_px)
            if score >= iou_threshold:
                candidates.append((score, pi, gi))
    # Greedy by descending IoU; tie-break deterministically by indices.
    candidates.sort(key=lambda c: (-c[0], c[1], c[2]))

    matched_pred: set[int] = set()
    matched_gt: set[int] = set()
    matched_iou: list[float] = []
    for score, pi, gi in candidates:
        if pi in matched_pred or gi in matched_gt:
            continue
        matched_pred.add(pi)
        matched_gt.add(gi)
        matched_iou.append(score)

    tp = len(matched_iou)
    fp = n_pred - tp
    fn = n_gt - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    mean_iou = sum(matched_iou) / len(matched_iou) if matched_iou else None
    return DetectionMetrics(
        precision=precision, recall=recall, f1=f1, mean_iou=mean_iou,
        true_positives=tp, false_positives=fp, false_negatives=fn,
    )


def _self_check() -> None:
    """Assert the metric known-answers hold. Runs under ``python -m``."""
    # IoU basics
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (100, 100, 10, 10)) == 0.0
    assert abs(iou((0, 0, 10, 10), (5, 0, 10, 10)) - (50 / 150)) < 1e-9

    gt = [LabeledBox("pothole", (0, 0, 10, 10)), LabeledBox("crack", (50, 50, 20, 4))]

    # identical -> 1.0 everywhere, mean_iou 1.0
    m = evaluate_detections(list(gt), gt)
    assert (m.precision, m.recall, m.f1) == (1.0, 1.0, 1.0), m
    assert m.mean_iou == 1.0 and m.true_positives == 2

    # disjoint (wrong places) -> 0.0
    wrong = [LabeledBox("pothole", (200, 200, 10, 10)), LabeledBox("crack", (300, 300, 20, 4))]
    m = evaluate_detections(wrong, gt)
    assert m.precision == 0.0 and m.recall == 0.0 and m.f1 == 0.0, m
    assert m.mean_iou is None and m.false_positives == 2 and m.false_negatives == 2

    # wrong label at the right place is NOT a match
    mislabel = [LabeledBox("crack", (0, 0, 10, 10))]
    m = evaluate_detections(mislabel, [LabeledBox("pothole", (0, 0, 10, 10))])
    assert m.true_positives == 0 and m.precision == 0.0 and m.recall == 0.0

    # both empty -> perfect vacuous
    m = evaluate_detections([], [])
    assert (m.precision, m.recall, m.f1) == (1.0, 1.0, 1.0) and m.mean_iou is None

    # predictions, no GT -> all FP
    m = evaluate_detections([LabeledBox("pothole", (0, 0, 5, 5))], [])
    assert m.precision == 0.0 and m.false_positives == 1

    # GT, no predictions -> all missed
    m = evaluate_detections([], [LabeledBox("pothole", (0, 0, 5, 5))])
    assert m.recall == 0.0 and m.false_negatives == 1

    # partial: one of two found
    m = evaluate_detections([LabeledBox("pothole", (0, 0, 10, 10))], gt)
    assert m.true_positives == 1 and m.false_negatives == 1
    assert abs(m.precision - 1.0) < 1e-9 and abs(m.recall - 0.5) < 1e-9

    # all metrics bounded
    for mm in [evaluate_detections(wrong, gt), evaluate_detections(list(gt), gt)]:
        for v in (mm.precision, mm.recall, mm.f1):
            assert 0.0 <= v <= 1.0


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    _self_check()
    print("[evaluation] OK: IoU + precision/recall/F1 with defined known answers, all bounded.")
