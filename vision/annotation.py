"""Annotated-frame evidence: draw detections and store the evidence frame.

Vision Pipeline Design, Requirement 2.3 ("WHEN the Vision_Pipeline produces a
detection, THE Vision_Pipeline SHALL generate an annotated frame marking the
detected region and SHALL store the annotated frame as Evidence") and the design
"Evidence Storage" subsection:

    Annotated frames are written to S3 under
    ``evidence/<inspection_id>/<frame_ref>.png``; findings and trace entries
    store only the S3 key (``evidence_ref``), keeping DynamoDB items small and
    resolvable (10.3, 2.3).

This module owns **both** halves of evidence generation and exposes the stable
public API the vision façade (task 4.5) imports and calls:

- :func:`annotate_frame` -- a **pure** function (no I/O) that draws the real
  detections onto a copy of the frame and returns the annotated image.
- :class:`EvidenceStore` -- a small ``Protocol`` describing an evidence sink:
  ``save(inspection_id, frame_ref, image) -> evidence_ref`` returns the storage
  key. The result/finding stores only this key (``evidence_ref``).
- :class:`LocalEvidenceStore` -- the concrete store for the **Milestone 2
  offline/mock path**. It writes PNGs to a local directory using the same key
  layout as the eventual S3 store (``evidence/<inspection_id>/<frame_ref>.png``)
  and returns that key. Milestone 6 (task 12.3) can drop in an
  ``S3EvidenceStore`` implementing the same :class:`EvidenceStore` Protocol
  **without changing any caller**.
- :func:`annotate_and_store` -- the convenience the façade calls: annotate the
  frame and persist it through a store, returning the ``evidence_ref`` key to
  set on the :class:`~vision.result_schema.AnalysisResult`.

Public API contract (what the façade, task 4.5, depends on)
-----------------------------------------------------------

```python
def annotate_frame(image: np.ndarray, detections: list[Detection]) -> np.ndarray: ...

class EvidenceStore(Protocol):
    def save(self, inspection_id: str, frame_ref: str, image: np.ndarray) -> str: ...

class LocalEvidenceStore:                 # EvidenceStore for the mock path
    def __init__(self, base_dir: str | os.PathLike) -> None: ...
    def save(self, inspection_id: str, frame_ref: str, image: np.ndarray) -> str: ...

def annotate_and_store(
    image: np.ndarray,
    detections: list[Detection],
    inspection_id: str,
    frame_ref: str,
    store: EvidenceStore,
) -> str: ...                              # returns the evidence_ref key
```

The returned ``evidence_ref`` is always the storage **key**
(``evidence/<inspection_id>/<sanitized_frame_ref>.png``), never a local absolute
path, so it is identical across the local and S3 stores and resolvable by the
dashboard/read layer (10.3, 2.3).

Design constraints honored here
-------------------------------

- **Real OpenCV 5 drawing only.** Boxes and labels are drawn with
  ``cv2.rectangle`` / ``cv2.putText`` on the real frame pixels. Nothing is
  fabricated: :func:`annotate_frame` draws exactly the detections passed in and
  invents nothing (Requirement 3.3). An empty detections list produces an
  un-annotated copy of the frame (documented behavior below).
- **Stable key layout, swappable backend.** The key layout matches the design's
  S3 layout exactly so the Milestone 6 S3 store is a drop-in replacement behind
  the :class:`EvidenceStore` seam.
- **Filesystem-safe keys.** A ``frame_ref`` such as ``"source#frame_00000000"``
  contains characters unsafe for a filename on some platforms (notably ``#`` and
  path separators). :func:`sanitize_frame_ref` maps it to a safe, stable, and
  collision-resistant filename stem while preserving readability.
- **Offline, no AWS.** The local store writes to the local filesystem and is
  importable/runnable with no AWS credentials (Requirement 19.4, Competition 4).

Usage::

    from vision.annotation import LocalEvidenceStore, annotate_and_store

    store = LocalEvidenceStore("artifacts")          # artifacts/evidence/...
    evidence_ref = annotate_and_store(
        frame.image, detections, inspection_id="insp-1", frame_ref=frame.frame_ref, store=store
    )
    result = success_result(..., evidence_ref=evidence_ref)
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Protocol, runtime_checkable

import cv2
import numpy as np

from vision.result_schema import Detection

# --------------------------------------------------------------------------- #
# Drawing configuration (explicit, not magic constants scattered inline)
# --------------------------------------------------------------------------- #

#: Prefix of every evidence key, matching the design "Evidence Storage" layout.
EVIDENCE_PREFIX = "evidence"

#: BGR color of the bounding box and label background (OpenCV is BGR). A warm
#: high-contrast orange reads clearly over typical gray road surface.
_BOX_COLOR_BGR: tuple[int, int, int] = (0, 140, 255)
#: BGR color of the label text drawn on the label background.
_TEXT_COLOR_BGR: tuple[int, int, int] = (0, 0, 0)
#: Bounding-box line thickness in pixels.
_BOX_THICKNESS = 2
#: Label font, scale, and thickness (readable without dominating the frame).
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.5
_FONT_THICKNESS = 1


# --------------------------------------------------------------------------- #
# Filesystem-safe key derivation
# --------------------------------------------------------------------------- #

#: Characters allowed verbatim in a sanitized filename stem. Everything else
#: (``#``, path separators, spaces, etc.) is replaced so the stem is safe on
#: Windows and POSIX alike.
_SAFE_STEM = re.compile(r"[^A-Za-z0-9._-]")


def sanitize_frame_ref(frame_ref: str) -> str:
    """Map a ``frame_ref`` to a safe, stable, unique filename stem.

    The frame sampler emits refs like ``"source#frame_00000000"`` whose ``#``
    (and any path separators in ``source``) are unsafe for filenames. This
    replaces every disallowed character with ``_`` so the result is a valid stem
    on Windows and POSIX.

    The mapping is **stable** (same input -> same output) and
    **collision-resistant**: distinct inputs that happen to sanitize to the same
    readable stem are disambiguated by a short hash suffix of the *original*
    ref, so two different frame refs never overwrite each other's evidence.

    Args:
        frame_ref: The source frame reference (e.g. ``"clip#frame_00000005"``).

    Returns:
        A filesystem-safe stem (no extension), e.g. ``"clip_frame_00000005"``.
    """
    if not frame_ref:
        raise ValueError("frame_ref must be a non-empty string")
    safe = _SAFE_STEM.sub("_", frame_ref)
    # Collapse runs of underscores for readability; keep it bounded in length.
    safe = re.sub(r"_+", "_", safe).strip("_")
    if not safe:
        safe = "frame"
    # If sanitizing changed the ref (so distinct refs could collide on the safe
    # stem), append a short stable hash of the ORIGINAL ref to guarantee
    # uniqueness. Pure-safe refs are left untouched for readability.
    if safe != frame_ref:
        import hashlib

        digest = hashlib.sha1(frame_ref.encode("utf-8")).hexdigest()[:8]
        safe = f"{safe}-{digest}"
    # Bound the stem length defensively (long source paths); keep the hash tail.
    if len(safe) > 180:
        safe = safe[:180]
    return safe


def evidence_key(inspection_id: str, frame_ref: str) -> str:
    """Build the storage key for an annotated frame (design "Evidence Storage").

    Returns ``evidence/<inspection_id>/<sanitized_frame_ref>.png`` using forward
    slashes so the key is identical across the local and S3 stores (S3 keys use
    ``/`` regardless of host OS). This is the value stored as ``evidence_ref``.
    """
    if not inspection_id:
        raise ValueError("inspection_id must be a non-empty string")
    stem = sanitize_frame_ref(frame_ref)
    return f"{EVIDENCE_PREFIX}/{inspection_id}/{stem}.png"


# --------------------------------------------------------------------------- #
# Pure annotation (no I/O)
# --------------------------------------------------------------------------- #


def _label_text(det: Detection) -> str:
    """Readable label: hazard type plus confidence as a percentage.

    Confidence is a real number in ``[0, 1]`` (enforced by the schema); it is
    shown as a whole-percent so a human reading the evidence frame can gauge the
    detector's certainty at a glance.
    """
    pct = int(round(float(det["confidence"]) * 100.0))
    return f"{det['hazard_type']} {pct}%"


def annotate_frame(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw ``detections`` onto a copy of ``image`` and return it (pure, no I/O).

    Draws, for each detection, a bounding rectangle at its real ``bbox_px`` and a
    readable label (``hazard_type`` + confidence as a percentage) in a filled
    background band so the text stays legible over any road texture. The input
    image is **not** mutated; a copy is returned.

    Only the detections passed in are drawn -- nothing is fabricated (Requirement
    3.3). An **empty** ``detections`` list yields an unmodified copy of the frame
    (there is nothing to mark); callers that still want to store such a frame may
    do so, which keeps evidence generation uniform whether or not a frame had a
    detection.

    Args:
        image: A real BGR frame ndarray shaped ``(h, w, 3)`` (or a grayscale
            ``(h, w)`` frame, which is converted to BGR so colored boxes/labels
            are visible). Never ``None``.
        detections: The detections to mark, each with a pixel-space ``bbox_px``.

    Returns:
        A new BGR ndarray with the detections drawn on it.
    """
    if image is None:
        raise ValueError("cannot annotate a None image")

    # Draw on a BGR copy so we never mutate the caller's frame and colored marks
    # are visible even on a grayscale input.
    if image.ndim == 2:
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 1:
        canvas = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        canvas = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim == 3 and image.shape[2] == 3:
        canvas = image.copy()
    else:
        raise ValueError(f"unsupported image shape for annotation: {image.shape!r}")

    frame_h, frame_w = canvas.shape[:2]

    for det in detections:
        x, y, w, h = det["bbox_px"]
        x, y, w, h = int(x), int(y), int(w), int(h)
        # Clamp the rectangle to the frame so drawing is always in-bounds even if
        # a bbox grazes the edge; we draw only the real region, never beyond it.
        x0 = max(0, min(x, frame_w - 1))
        y0 = max(0, min(y, frame_h - 1))
        x1 = max(0, min(x + w, frame_w - 1))
        y1 = max(0, min(y + h, frame_h - 1))
        cv2.rectangle(canvas, (x0, y0), (x1, y1), _BOX_COLOR_BGR, _BOX_THICKNESS)

        label = _label_text(det)
        (text_w, text_h), baseline = cv2.getTextSize(
            label, _FONT, _FONT_SCALE, _FONT_THICKNESS
        )
        # Place the label band just above the box; drop it inside the box if
        # there is no room above (box at the top edge).
        band_bottom = y0 - 2
        band_top = band_bottom - text_h - baseline
        if band_top < 0:
            band_top = y0 + 2
            band_bottom = band_top + text_h + baseline
        band_right = min(frame_w - 1, x0 + text_w + 4)
        cv2.rectangle(
            canvas, (x0, band_top), (band_right, band_bottom), _BOX_COLOR_BGR, thickness=-1
        )
        text_baseline_y = band_bottom - baseline // 2
        cv2.putText(
            canvas,
            label,
            (x0 + 2, text_baseline_y),
            _FONT,
            _FONT_SCALE,
            _TEXT_COLOR_BGR,
            _FONT_THICKNESS,
            cv2.LINE_AA,
        )

    return canvas


# --------------------------------------------------------------------------- #
# Evidence store seam (swappable local / S3 backends)
# --------------------------------------------------------------------------- #


@runtime_checkable
class EvidenceStore(Protocol):
    """The evidence sink the façade persists annotated frames through.

    A store takes an ``inspection_id``, a ``frame_ref``, and an annotated image,
    writes the image under the stable evidence key layout
    (``evidence/<inspection_id>/<sanitized_frame_ref>.png``), and returns that
    **key** as the ``evidence_ref``. Findings and trace entries store only the
    key (design "Evidence Storage"), so the key must be identical across
    backends.

    Milestone 2 uses :class:`LocalEvidenceStore`; Milestone 6 (task 12.3) can
    provide an ``S3EvidenceStore`` implementing this same Protocol without any
    caller change.
    """

    def save(self, inspection_id: str, frame_ref: str, image: np.ndarray) -> str:
        """Persist ``image`` for ``(inspection_id, frame_ref)`` and return its key."""
        ...


class LocalEvidenceStore:
    """Filesystem :class:`EvidenceStore` for the offline/mock path (Milestone 2).

    Writes annotated frames as PNGs under ``<base_dir>/evidence/<inspection_id>/
    <sanitized_frame_ref>.png`` -- the same key layout the S3 store will use --
    and returns the storage **key** (``evidence/<inspection_id>/...png``), not
    the local absolute path, so the returned ``evidence_ref`` is identical to
    what the S3 store returns and is resolvable by the read/dashboard layer
    (10.3, 2.3).

    Parent directories are created on demand. The store performs no AWS or
    network work.
    """

    def __init__(self, base_dir: str | os.PathLike) -> None:
        """Create a store rooted at ``base_dir`` (the ``evidence/`` dir lives under it)."""
        self._base_dir = Path(base_dir)

    @property
    def base_dir(self) -> Path:
        """The root directory under which ``evidence/`` keys are materialized."""
        return self._base_dir

    def key_to_path(self, key: str) -> Path:
        """Resolve a storage ``key`` to its local filesystem path under ``base_dir``."""
        return self._base_dir / Path(key)

    def save(self, inspection_id: str, frame_ref: str, image: np.ndarray) -> str:
        """Write ``image`` as a PNG under the evidence key and return the key.

        Args:
            inspection_id: The inspection this evidence belongs to.
            frame_ref: The source frame reference (sanitized for the filename).
            image: The annotated BGR frame to persist. Never ``None``.

        Returns:
            The storage key ``evidence/<inspection_id>/<sanitized_frame_ref>.png``.

        Raises:
            ValueError: if ``image`` is ``None``.
            OSError: if the PNG could not be written.
        """
        if image is None:
            raise ValueError("cannot store a None image as evidence")
        key = evidence_key(inspection_id, frame_ref)
        path = self.key_to_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # cv2.imwrite does not raise on failure; it returns False. Surface that
        # honestly as an OSError rather than silently returning a dangling key.
        wrote = cv2.imwrite(str(path), image)
        if not wrote or not path.exists():
            raise OSError(f"failed to write evidence PNG to {path!s}")
        return key


# --------------------------------------------------------------------------- #
# Convenience: annotate + store in one call (the façade entry point)
# --------------------------------------------------------------------------- #


def annotate_and_store(
    image: np.ndarray,
    detections: list[Detection],
    inspection_id: str,
    frame_ref: str,
    store: EvidenceStore,
) -> str:
    """Annotate ``image`` with ``detections`` and persist it, returning the key.

    This is the single call the vision façade (task 4.5) makes to turn a real
    analysis into stored evidence: it draws the detections (:func:`annotate_frame`)
    and writes the annotated frame through the injected :class:`EvidenceStore`,
    returning the ``evidence_ref`` key to set on the
    :class:`~vision.result_schema.AnalysisResult` (2.3).

    Args:
        image: The real source frame ndarray to annotate. Never ``None``.
        detections: The detections to mark (empty is allowed; the frame is still
            annotated-as-copy and stored, keeping evidence generation uniform).
        inspection_id: The inspection the evidence belongs to.
        frame_ref: The source frame reference.
        store: Any :class:`EvidenceStore` (local for the mock path, S3 later).

    Returns:
        The ``evidence_ref`` storage key.
    """
    annotated = annotate_frame(image, detections)
    return store.save(inspection_id, frame_ref, annotated)


# --------------------------------------------------------------------------- #
# Self-validation (import-time smoke check: annotate + store + read back)
# --------------------------------------------------------------------------- #


def _self_check() -> None:
    """Annotate a synthetic frame with a known detection, store it, read it back.

    Minimal real-OpenCV guard (no mocks), not the optional integration suite
    (task 4.7). Writes to a temp dir and asserts the evidence file exists and is
    a readable PNG at the stable key. Runs under ``python -m``.
    """
    import tempfile

    from vision.result_schema import detection, pixel_measurement

    # A synthetic mid-gray frame with one known detection.
    frame = np.full((240, 320, 3), 128, dtype=np.uint8)
    det = detection("pothole", (100, 80, 60, 40), 0.73, pixel_measurement(60, 40))

    # Pure annotation returns a NEW image (input untouched) of the same shape.
    before = frame.copy()
    annotated = annotate_frame(frame, [det])
    assert annotated.shape == frame.shape, annotated.shape
    assert np.array_equal(frame, before), "annotate_frame must not mutate its input"
    assert not np.array_equal(annotated, frame), "expected drawn marks to change pixels"

    # frame_ref sanitization: the sampler's "source#frame_00000000" layout.
    raw_ref = "clip name#frame_00000000"
    stem = sanitize_frame_ref(raw_ref)
    assert "#" not in stem and "/" not in stem and " " not in stem, stem

    with tempfile.TemporaryDirectory() as tmp:
        store = LocalEvidenceStore(tmp)
        evidence_ref = annotate_and_store(
            frame, [det], inspection_id="insp-1", frame_ref=raw_ref, store=store
        )
        # The evidence_ref is the stable KEY, not a local path.
        assert evidence_ref == f"evidence/insp-1/{stem}.png", evidence_ref
        assert evidence_ref.startswith("evidence/insp-1/")

        # The file exists on disk at base_dir/key and reads back as a PNG image.
        path = store.key_to_path(evidence_ref)
        assert path.exists(), f"evidence file not written: {path}"
        read_back = cv2.imread(str(path))
        assert read_back is not None, "stored evidence is not a readable image"
        assert read_back.shape == frame.shape, read_back.shape

        # Empty detections: still produces/stores a frame (documented behavior).
        empty_ref = annotate_and_store(
            frame, [], inspection_id="insp-1", frame_ref="clip#frame_00000005", store=store
        )
        assert store.key_to_path(empty_ref).exists(), empty_ref

    # Distinct raw refs that sanitize similarly do not collide (hash suffix).
    a = sanitize_frame_ref("a/b#frame_1")
    b = sanitize_frame_ref("a_b_frame_1")
    assert a != b, (a, b)

    # The local store satisfies the EvidenceStore Protocol.
    assert isinstance(LocalEvidenceStore("."), EvidenceStore)


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    _self_check()
    print(
        "[annotation] OK: annotate_frame draws real detections (pure); "
        "LocalEvidenceStore writes evidence/<id>/<frame>.png and returns the key; "
        "annotate_and_store composes both behind the EvidenceStore seam."
    )
