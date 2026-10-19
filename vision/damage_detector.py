"""Damage detector: the Classical_Path (classical CV) hazard detector.

Vision Pipeline Design, ``damage_detector.py`` row of the component-responsibilities
table:

    | Input                              | Output                                              | Failure behavior                                           |
    | preprocessed frame or region, method | ``Detection[]`` with ``bbox_px``, ``confidence``, ``hazard_type`` | On Model_Path load failure, raises ``ModelLoadError`` (4.5) |

This module implements the **Classical_Path** (Requirement 4.1) with real
OpenCV 5 operations and exposes the detector seam the rest of the pipeline
composes against. The Model_Path (DNN/ONNX) is a later task (14.2); this module
is honest about that boundary -- see :func:`detect_regions`.

The Classical_Path (design "Detection Paths", Classical_Path)
-------------------------------------------------------------

The pipeline mirrors the design description exactly, all on real pixels:

1. **Grayscale normalization.** Accepts either a BGR frame (raw sampled frame)
   or an already-grayscale frame (the image preprocessor's output) and converts
   to single-channel gracefully. Nothing downstream assumes a channel count.
2. **Canny edge detection** (``cv2.Canny``) with thresholds derived from the
   image's own median intensity, so the edge map adapts to the frame's exposure
   instead of using brittle fixed thresholds.
3. **Morphological open then close** (``cv2.morphologyEx``) to remove speckle
   noise (open) and bridge the broken edges of a single hazard into one
   connected region (close).
4. **Contour extraction and filtering** (``cv2.findContours``). Contours that
   are too small (noise) or too large (whole-frame lighting gradients) relative
   to the image area are discarded.
5. **Texture analysis** -- the local variance of the Laplacian response inside
   each candidate region (``cv2.Laplacian``). Road damage has broken,
   high-variance texture; a flat shadow has low-variance texture. This is how
   the classical path distinguishes genuine surface damage from a mere dark
   patch / shadow (design: "texture analysis ... to distinguish surface damage
   from shadows").

Every surviving contour becomes a pixel-space
:class:`~vision.result_schema.Detection` built through the
:func:`~vision.result_schema.detection` factory, so the responsible-use
invariants (pixel-labeled measurement, bounded confidence) hold by construction
(Requirements 2.2, 3.1, 3.2, 3.3).

Heuristic confidence (NOT a validated score)
---------------------------------------------

Confidence is a **real** function of measured features, never fabricated
(Requirement 3.3). It combines, each normalized to ``[0, 1]``:

* **edge strength** -- the density of Canny edge pixels inside the region,
* **solidity** -- contour area divided by its convex-hull area (how "filled in"
  the shape is; potholes/patches are solid, scattered gravel is not),
* **texture score** -- the saturating Laplacian-variance texture measure above.

The three are averaged and then **capped at**
:data:`CLASSICAL_CONFIDENCE_CEILING` (0.85). Classical thresholds/edges are
heuristics, not a validated detector, so the score is kept modest and clearly
bounded below 1.0. This is the responsible-use stance (R3 / Requirement 3.3):
the classical path must not present an edge-and-threshold response as a
validated, high-certainty detection.

Hazard types the Classical_Path honestly supports (4.6)
-------------------------------------------------------

Classical CV reasons about **shape and texture**, not semantics, so it labels
only what shape/texture can actually distinguish, rather than labeling every
blob as every hazard:

* **pothole** -- a reasonably solid, compact dark region with high-variance
  (broken) texture: a filled contour whose aspect ratio is near-square and whose
  solidity is high.
* **surface_damage** -- a solid-ish region with high texture that is *not*
  compact enough to call a pothole (irregular patches of degraded surface).
* **crack** -- a long, thin, low-solidity region (high elongation / extent):
  shape alone distinguishes a crack from a blob, which is exactly what classical
  CV can do.
* **debris** -- the classical path does **not** emit ``debris``. Distinguishing
  an object-on-road from surface texture needs semantics the classical path does
  not have; claiming it would be dishonest (Requirement 4.6 says "where the
  active detection path supports" the type -- the classical path does not
  support debris, and the Model_Path (14.2) is the right place for it).

So the Classical_Path supports ``pothole``, ``surface_damage``, and ``crack``.

Offline, no AWS
---------------

Pure CPU/OpenCV work on a local ndarray; importable and runnable with no AWS
credentials (Requirement 19.4, Competition 4).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from vision.result_schema import (
    Detection,
    HazardType,
    Method,
    detection,
    pixel_measurement,
)


class ModelLoadError(RuntimeError):
    """Raised when the Model_Path cannot load its pinned ONNX model (4.5).

    Defined here (not in the Model_Path task) so the error type the agent
    catches to fall back to the Classical_Path has a stable home. The classical
    path never raises it; the Model_Path (task 14.2) will.
    """


#: Classical confidence is a heuristic, not a validated score. Cap it well below
#: 1.0 so an edge/threshold response is never presented as a certain detection
#: (Requirement 3.3, responsible-use R3).
CLASSICAL_CONFIDENCE_CEILING: float = 0.85

#: Confidence floor below which a candidate is discarded as too weak to report.
CLASSICAL_CONFIDENCE_FLOOR: float = 0.15

#: Laplacian-variance texture value at which the texture sub-score saturates to
#: 1.0. Road damage has broken, high-variance texture; a flat shadow is low.
TEXTURE_SATURATION: float = 350.0


@dataclass(frozen=True)
class ClassicalConfig:
    """Configurable knobs for the classical detection path.

    All thresholds are explicit and overridable so detection behavior is a
    configured policy rather than magic constants buried in the code.
    """

    #: Minimum contour area as a fraction of the image area. Smaller -> noise.
    min_area_frac: float = 0.0008
    #: Maximum contour area as a fraction of the image area. Larger -> a
    #: whole-frame lighting gradient, not a discrete hazard.
    max_area_frac: float = 0.60
    #: Morphology kernel size (pixels) for the open/close noise cleanup.
    morph_kernel: int = 5
    #: Fraction below/above the image median used to derive the Canny
    #: hysteresis thresholds (adaptive to exposure).
    canny_sigma: float = 0.33
    #: Texture (Laplacian variance) at or above which a region is texture-rich
    #: enough to be a hazard rather than a flat shadow.
    min_texture: float = 40.0
    #: Confidence ceiling (see module docstring).
    confidence_ceiling: float = CLASSICAL_CONFIDENCE_CEILING
    #: Confidence floor; weaker candidates are dropped.
    confidence_floor: float = CLASSICAL_CONFIDENCE_FLOOR
    #: Texture-saturation value for the texture sub-score.
    texture_saturation: float = TEXTURE_SATURATION

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_area_frac < self.max_area_frac <= 1.0:
            raise ValueError(
                "require 0 <= min_area_frac < max_area_frac <= 1, got "
                f"min={self.min_area_frac}, max={self.max_area_frac}"
            )
        if self.morph_kernel < 1:
            raise ValueError(f"morph_kernel must be >= 1, got {self.morph_kernel}")
        if not 0.0 <= self.confidence_floor <= self.confidence_ceiling <= 1.0:
            raise ValueError(
                "require 0 <= confidence_floor <= confidence_ceiling <= 1, got "
                f"floor={self.confidence_floor}, ceiling={self.confidence_ceiling}"
            )
        if self.texture_saturation <= 0:
            raise ValueError(f"texture_saturation must be > 0, got {self.texture_saturation}")


DEFAULT_CONFIG = ClassicalConfig()

#: Hazard types the Classical_Path can honestly distinguish by shape+texture.
#: Note the deliberate absence of ``debris`` (see module docstring, 4.6).
CLASSICAL_HAZARD_TYPES: tuple[HazardType, ...] = ("pothole", "surface_damage", "crack")


# --------------------------------------------------------------------------- #
# Image normalization
# --------------------------------------------------------------------------- #


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Return a single-channel uint8 view of ``image`` (BGR or already gray).

    The preprocessor emits grayscale; a raw sampled frame is BGR. Both are
    accepted gracefully so the detector composes with either upstream source.
    """
    if image is None:  # defensive: never analyze absent pixels
        raise ValueError("cannot detect on a None image")
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    elif image.ndim == 3 and image.shape[2] == 1:
        gray = image[:, :, 0]
    else:
        raise ValueError(f"unsupported image shape for detection: {image.shape!r}")
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray


# --------------------------------------------------------------------------- #
# Classical CV stages (real OpenCV 5)
# --------------------------------------------------------------------------- #


def _auto_canny(gray: np.ndarray, sigma: float) -> np.ndarray:
    """Canny edge map with thresholds adapted to the image's own median.

    Deriving the hysteresis thresholds from the median intensity makes the edge
    map robust to exposure instead of relying on brittle fixed thresholds.
    """
    median = float(np.median(gray))
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    if upper <= lower:  # degenerate (near-flat) image: widen a little
        lower, upper = 50, 150
    return cv2.Canny(gray, lower, upper)


def _clean_edges(edges: np.ndarray, kernel_size: int) -> np.ndarray:
    """Morphological close then open to connect edge fragments and de-noise.

    A Canny edge map is thin (~1px) and broken. Applying *open* first would
    erase those thin lines entirely, so the order is deliberate:

    1. **Close** (dilate then erode) bridges the broken edges of a single
       hazard boundary into one connected, filled region, so the hazard yields
       one contour rather than a scatter of fragments.
    2. **Open** (erode then dilate) then removes the isolated speckle that
       survived closing (stray texture edges), without destroying the now-solid
       hazard region.

    Closing uses a kernel large enough to span the typical gap between adjacent
    edge fragments; opening uses a smaller kernel so it only strips speckle.
    """
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size * 2 + 1, kernel_size * 2 + 1)
    )
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_kernel)
    return opened


def _texture_score(gray: np.ndarray, bbox: tuple[int, int, int, int], saturation: float) -> float:
    """Saturating [0,1] texture score: Laplacian variance inside the region.

    Road damage has broken, high-variance surface texture; a flat shadow has
    low variance. This is the signal that separates genuine surface damage from
    a merely dark patch (design: texture analysis to distinguish damage from
    shadows).
    """
    x, y, w, h = bbox
    roi = gray[y : y + h, x : x + w]
    if roi.size == 0:
        return 0.0
    variance = float(cv2.Laplacian(roi, cv2.CV_64F).var())
    return float(min(1.0, variance / saturation))


def _edge_strength(edges: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    """[0,1] density of Canny edge pixels inside the region.

    A strong, well-defined hazard boundary produces a denser edge response than
    a faint gradient. Computed on the real edge map.
    """
    x, y, w, h = bbox
    roi = edges[y : y + h, x : x + w]
    if roi.size == 0:
        return 0.0
    return float(np.count_nonzero(roi)) / float(roi.size)


def _solidity(contour: np.ndarray, area: float) -> float:
    """Contour area over convex-hull area, in [0,1] (how 'filled in' the shape).

    Potholes and surface patches are solid; scattered gravel or an open crack
    network is not. Falls back to 0 for a degenerate hull.
    """
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    if hull_area <= 0:
        return 0.0
    return float(min(1.0, area / hull_area))


def _classify(
    bbox: tuple[int, int, int, int], solidity: float, extent: float
) -> HazardType:
    """Label a region by its shape, honestly within classical CV's reach (4.6).

    - A long, thin, low-solidity region is a **crack** (elongation is a shape
      cue classical CV genuinely resolves).
    - A compact, near-square, solid region is a **pothole**.
    - Anything else solid-and-textured is **surface_damage**.

    ``extent`` is contour area / bounding-box area; ``solidity`` is contour area
    / convex-hull area. ``debris`` is intentionally never returned here.
    """
    x, y, w, h = bbox
    long_side = max(w, h)
    short_side = max(1, min(w, h))
    elongation = long_side / short_side

    # Crack: elongated and sparse (a thin line does not fill its bbox).
    if elongation >= 4.0 and extent <= 0.45:
        return "crack"
    # Pothole: compact (near-square) and solid.
    aspect = w / max(1, h)
    if 0.6 <= aspect <= 1.7 and solidity >= 0.80:
        return "pothole"
    # Otherwise an irregular degraded patch.
    return "surface_damage"


def _confidence(edge_strength: float, solidity: float, texture: float, config: ClassicalConfig) -> float:
    """Blend measured features into a modest, bounded heuristic confidence.

    The mean of the three real sub-scores, clamped into
    ``[0, confidence_ceiling]``. Never 1.0: classical heuristics are not a
    validated detector (Requirement 3.3).
    """
    raw = (edge_strength + solidity + texture) / 3.0
    return float(max(0.0, min(config.confidence_ceiling, raw)))


# --------------------------------------------------------------------------- #
# Public detector seam
# --------------------------------------------------------------------------- #


def detect_classical(
    image: np.ndarray, config: ClassicalConfig = DEFAULT_CONFIG
) -> list[Detection]:
    """Run the full Classical_Path on ``image`` and return pixel-space detections.

    The returned :class:`~vision.result_schema.Detection` bboxes are in the
    coordinate frame of the *image passed in* (crop-local when called on an ROI),
    which is exactly the contract :mod:`vision.region_analyzer` expects from the
    detector seam -- it translates crop-local bboxes back into frame coordinates.

    Args:
        image: A real BGR or grayscale frame (or ROI). Never ``None``.
        config: Optional :class:`ClassicalConfig` policy overrides.

    Returns:
        A list of pixel-space detections (possibly empty). Confidence is a real,
        bounded heuristic; hazard types are limited to those the classical path
        honestly supports (``pothole``, ``surface_damage``, ``crack``).
    """
    gray = _to_gray(image)
    height, width = gray.shape[:2]
    image_area = float(height * width)
    if image_area <= 0:
        return []

    edges = _auto_canny(gray, config.canny_sigma)
    cleaned = _clean_edges(edges, config.morph_kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = config.min_area_frac * image_area
    max_area = config.max_area_frac * image_area

    detections: list[Detection] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue  # too small (noise) or too large (lighting gradient)

        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue
        bbox = (int(x), int(y), int(w), int(h))

        texture = _texture_score(gray, bbox, config.texture_saturation)
        # A flat, low-texture dark patch is a shadow, not damage -> reject.
        raw_texture_var = texture * config.texture_saturation
        if raw_texture_var < config.min_texture:
            continue

        bbox_area = float(w * h)
        extent = area / bbox_area if bbox_area > 0 else 0.0
        solidity = _solidity(contour, area)
        edge_strength = _edge_strength(edges, bbox)

        confidence = _confidence(edge_strength, solidity, texture, config)
        if confidence < config.confidence_floor:
            continue  # too weak to report honestly

        hazard = _classify(bbox, solidity, extent)
        detections.append(
            detection(
                hazard,
                bbox,
                confidence,
                pixel_measurement(w, h, area_px=int(round(area))),
            )
        )

    return detections


# --------------------------------------------------------------------------- #
# Model_Path (ONNX via OpenCV DNN) -- task 14.2
# --------------------------------------------------------------------------- #

from pathlib import Path as _Path

#: Pinned ONNX artifact path (see vision/models/MODEL_MANIFEST.md). Fetched on
#: demand, git-ignored; absent by default (Classical_Path is the default).
_MODEL_PATH = _Path(__file__).resolve().parent / "models" / "road_damage_yolov11s.onnx"

#: Class id -> hazard_type for the pinned RDD2022-fine-tuned model (4 classes).
_MODEL_CLASSES: tuple[HazardType, ...] = ("crack", "crack", "crack", "pothole")

#: Model input size and score threshold (modest; model confidence is the net's).
_MODEL_INPUT = 640
_MODEL_SCORE_THRESHOLD = 0.25

# Cache the loaded net so repeated calls do not re-read the file. Keyed by path
# so a swapped artifact reloads. A load failure is NOT cached (so a later fetch
# can succeed without a process restart).
_net_cache: dict[str, "cv2.dnn.Net"] = {}


def _load_model(model_path: _Path = _MODEL_PATH) -> "cv2.dnn.Net":
    """Load the pinned ONNX model via OpenCV DNN, or raise :class:`ModelLoadError`.

    Raises ModelLoadError when the artifact is absent or when this OpenCV build
    cannot parse the ONNX graph (newer YOLO export ops are not universally
    supported by cv2.dnn). The agent catches ModelLoadError and falls back to the
    Classical_Path (R4.5) -- we never silently substitute classical output.
    """
    key = str(model_path)
    cached = _net_cache.get(key)
    if cached is not None:
        return cached
    if not model_path.exists():
        raise ModelLoadError(
            f"ONNX model not found at {model_path}; fetch per vision/models/MODEL_MANIFEST.md "
            "(the Classical_Path remains the default)"
        )
    try:
        net = cv2.dnn.readNetFromONNX(str(model_path))
    except cv2.error as exc:  # pragma: no cover - depends on the OpenCV build
        raise ModelLoadError(
            f"OpenCV DNN could not load the ONNX model at {model_path}: {exc}"
        ) from exc
    _net_cache[key] = net
    return net


def detect_model(image: np.ndarray, model_path: _Path = _MODEL_PATH) -> list[Detection]:
    """Run the Model_Path (ONNX DNN) and return pixel-space detections (14.2).

    Letterboxes ``image`` to the model input, runs a forward pass, and decodes
    the YOLO-style output into :class:`~vision.result_schema.Detection` records
    in the coordinate frame of the image passed in. Confidence is the network's
    own score (real, not a heuristic). Raises :class:`ModelLoadError` if the
    model cannot be loaded (handled by the agent's classical fallback, R4.5).

    The output-decoding supports the common YOLO ONNX layout (one tensor shaped
    [1, 4+C, N] or [1, N, 4+C]); if the actual graph differs, decoding yields no
    detections rather than fabricating boxes (3.3).
    """
    net = _load_model(model_path)
    h0, w0 = image.shape[:2]
    if h0 == 0 or w0 == 0:
        return []
    bgr = image if (image.ndim == 3 and image.shape[2] == 3) else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    blob = cv2.dnn.blobFromImage(
        bgr, scalefactor=1 / 255.0, size=(_MODEL_INPUT, _MODEL_INPUT), swapRB=True, crop=False
    )
    net.setInput(blob)
    try:
        out = net.forward()
    except cv2.error as exc:  # pragma: no cover - build/graph dependent
        raise ModelLoadError(f"OpenCV DNN forward pass failed: {exc}") from exc

    pred = np.squeeze(out)
    if pred.ndim != 2:
        return []
    # Normalize to rows = detections, cols = [cx, cy, w, h, class scores...].
    if pred.shape[0] < pred.shape[1]:
        pred = pred.transpose()
    num_classes = len(_MODEL_CLASSES)
    if pred.shape[1] < 4 + num_classes:
        return []

    sx, sy = w0 / _MODEL_INPUT, h0 / _MODEL_INPUT
    boxes: list[list[int]] = []
    scores: list[float] = []
    class_ids: list[int] = []
    for row in pred:
        cls_scores = row[4 : 4 + num_classes]
        cid = int(np.argmax(cls_scores))
        score = float(cls_scores[cid])
        if score < _MODEL_SCORE_THRESHOLD:
            continue
        cx, cy, bw, bh = (float(v) for v in row[:4])
        x = int((cx - bw / 2) * sx)
        y = int((cy - bh / 2) * sy)
        boxes.append([x, y, int(bw * sx), int(bh * sy)])
        scores.append(score)
        class_ids.append(cid)

    if not boxes:
        return []
    # NMS to dedupe overlapping boxes (standard OpenCV DNN post-processing).
    idxs = cv2.dnn.NMSBoxes(boxes, scores, _MODEL_SCORE_THRESHOLD, 0.45)
    detections: list[Detection] = []
    for i in np.array(idxs).flatten():
        x, y, bw, bh = boxes[int(i)]
        x, y = max(0, x), max(0, y)
        bw, bh = max(1, bw), max(1, bh)
        detections.append(
            detection(
                _MODEL_CLASSES[class_ids[int(i)]],
                (x, y, bw, bh),
                min(1.0, scores[int(i)]),
                pixel_measurement(bw, bh),
            )
        )
    return detections


def detect_regions(image: np.ndarray, method: Method) -> list[Detection]:
    """Detector-seam entry point resolved by :mod:`vision.region_analyzer` (4.2).

    This is the function the region analyzer lazily imports and the vision
    façade wires in. It dispatches on ``method``:

    * ``"classical"`` -> run the real :func:`detect_classical` path.
    * ``"model"``     -> the Model_Path (ONNX DNN, :func:`detect_model`). Loads
      the pinned ONNX model via OpenCV DNN and returns its detections; raises
      :class:`ModelLoadError` when the artifact is absent or this OpenCV build
      cannot parse it. The agent's method-selection/fallback (task 14.3) catches
      ``ModelLoadError`` and falls back to the Classical_Path (4.5); it is never
      silently replaced with classical output.

    Args:
        image: A real BGR or grayscale image (or ROI). Never ``None``.
        method: The selected detection method. Only ``"classical"`` is handled
            by this module.

    Returns:
        Crop-local pixel-space detections for the classical method.

    Raises:
        ModelLoadError: when ``method == "model"`` and the pinned ONNX model
            cannot be loaded (absent artifact or unsupported graph) - the agent
            falls back to the Classical_Path (4.5).
        ValueError: when ``method`` is not a recognized method literal.
    """
    if method == "classical":
        return detect_classical(image)
    if method == "model":
        # Real Model_Path (14.2): raises ModelLoadError if the ONNX cannot load,
        # which the agent catches to fall back to the Classical_Path (4.5).
        return detect_model(image)
    raise ValueError(f"unknown method {method!r}; expected 'classical' or 'model'")


class DamageDetector:
    """Class form of the detector seam (``RegionDetector`` Protocol).

    :mod:`vision.region_analyzer` accepts either the module-level
    :func:`detect_regions` function or a ``DamageDetector`` class exposing a
    ``detect(image, method) -> list[Detection]`` method. Both resolve to the same
    real classical path; this class lets the façade/tests inject a configured
    detector.
    """

    def __init__(self, config: ClassicalConfig = DEFAULT_CONFIG) -> None:
        self._config = config

    def detect(self, image: np.ndarray, method: Method) -> list[Detection]:
        """Run the classical path (or reject the model path) -- see :func:`detect_regions`."""
        if method == "classical":
            return detect_classical(image, self._config)
        if method == "model":
            # Real Model_Path (14.2); ModelLoadError -> agent falls back (4.5).
            return detect_model(image)
        raise ValueError(f"unknown method {method!r}; expected 'classical' or 'model'")


# --------------------------------------------------------------------------- #
# Self-validation (import-time smoke check against a synthetic dark-blob frame)
# --------------------------------------------------------------------------- #


def _make_pothole_frame() -> np.ndarray:
    """A synthetic road-like frame with one dark, textured pothole-like blob.

    Mid-gray road surface with fine texture, plus a dark, high-variance filled
    ellipse standing in for a pothole. Real pixels only -- no fabricated result.
    """
    rng = np.random.default_rng(42)
    # Mid-gray road with light surface texture.
    frame = np.full((240, 320, 3), 128, dtype=np.uint8)
    texture = rng.integers(-15, 15, (240, 320, 1), dtype=np.int16)
    frame = np.clip(frame.astype(np.int16) + np.repeat(texture, 3, axis=2), 0, 255).astype(
        np.uint8
    )
    # A dark, compact, broken-textured blob: pothole-like.
    cv2.ellipse(frame, (160, 120), (34, 30), 0, 0, 360, (35, 35, 35), -1)
    blob_noise = rng.integers(-40, 40, (240, 320, 1), dtype=np.int16)
    dark = np.clip(frame.astype(np.int16) + np.repeat(blob_noise, 3, axis=2), 0, 255).astype(
        np.uint8
    )
    # Blend the heavy noise only inside the blob so its texture variance is high.
    mask = np.zeros((240, 320), dtype=np.uint8)
    cv2.ellipse(mask, (160, 120), (34, 30), 0, 0, 360, 255, -1)
    frame[mask > 0] = dark[mask > 0]
    return frame


def _self_check() -> None:
    """Detect a known dark blob and assert a plausible, bounded detection (4.1).

    Lightweight smoke check (real OpenCV 5, no mocks), not the property/
    integration suite (optional tasks 4.6/4.7). Runs under ``python -m``.
    """
    frame = _make_pothole_frame()
    h, w = frame.shape[:2]

    dets = detect_classical(frame)
    assert dets, "expected at least one detection on a synthetic dark-blob frame"

    for d in dets:
        # Pixel-labeled, bounded confidence, honest hazard type.
        assert d["measurement"]["unit"] == "pixels"
        assert d["measurement"]["physical"] == "unavailable"
        assert 0.0 <= d["confidence"] <= CLASSICAL_CONFIDENCE_CEILING, d["confidence"]
        assert d["hazard_type"] in CLASSICAL_HAZARD_TYPES, d["hazard_type"]
        x, y, bw, bh = d["bbox_px"]
        assert 0 <= x < w and 0 <= y < h, d["bbox_px"]
        assert bw > 0 and bh > 0 and x + bw <= w and y + bh <= h, d["bbox_px"]

    # The blob is centered around (160, 120); at least one detection should
    # overlap that neighborhood, proving the bbox is plausible (not arbitrary).
    def _covers_center(det: Detection) -> bool:
        x, y, bw, bh = det["bbox_px"]
        return x <= 160 <= x + bw and y <= 120 <= y + bh

    assert any(_covers_center(d) for d in dets), (
        "no detection covered the known blob center (160,120)"
    )

    # Seam dispatch: classical runs; model is honestly not implemented here.
    assert detect_regions(frame, "classical")
    # Model_Path (14.2): with no fetched ONNX artifact present, the model path
    # raises ModelLoadError (which the agent catches to fall back to classical,
    # 4.5). It must NOT silently return classical results mislabeled as model.
    try:
        detect_regions(frame, "model")
    except ModelLoadError:
        pass
    else:  # pragma: no cover - only reached if a real ONNX artifact is present
        pass  # a present, loadable model returning detections is also valid

    # A flat, textureless frame yields no fabricated detections.
    flat = np.full((240, 320, 3), 128, dtype=np.uint8)
    assert detect_classical(flat) == [], "flat frame must produce no detections"

    # Grayscale input is accepted gracefully (preprocessor output).
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    assert isinstance(detect_classical(gray), list)

    # The class form resolves to the same seam.
    assert isinstance(DamageDetector().detect(frame, "classical"), list)


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    _self_check()
    print(
        "[damage_detector] OK: classical path (Canny + morphology + contours + "
        "texture) produces pixel-labeled detections with bounded heuristic "
        "confidence; supports pothole/surface_damage/crack (not debris)."
    )
