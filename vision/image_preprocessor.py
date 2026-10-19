"""Image preprocessor: normalize a sampled frame and score its input quality.

Vision Pipeline Design, ``image_preprocessor.py`` row of the
component-responsibilities table:

    | Input   | Output                                      | Failure behavior                                      |
    | ``Frame`` | preprocessed ``numpy.ndarray`` + quality score | Low quality score propagates a ``Quality_Flag`` (2.5) |

This is the third stage of the pipeline: it consumes a :class:`~vision.frame_sampler.Frame`
produced by the sampler and emits a :class:`PreprocessedFrame` carrying the
preprocessed pixels, a real quality score derived from genuine image properties,
and any :class:`~vision.result_schema.QualityFlag`. The preprocessed frame feeds
the detection paths (task 4.2).

Design constraints honored here:

- **Real OpenCV 5 only.** Resize, denoise, and grayscale are genuine ``cv2``
  operations; the quality score is computed from real pixel statistics - the
  variance of the Laplacian (a standard focus/blur measure) and the mean
  brightness - never fabricated (Requirements 2.1, 2.5, 3.3).
- **Honest low-quality flagging.** When the computed quality score falls below a
  **configurable threshold**, the result carries a ``poor_input_quality``
  :class:`~vision.result_schema.QualityFlag` describing the measured deficiency
  (Requirement 2.5). The flag reuses the sanctioned
  :func:`~vision.result_schema.quality_flag` factory and the ``QualityFlag`` type;
  nothing about the schema is reinvented here.
- **Decode failures are propagated, not masked.** A :class:`Frame` that failed to
  decode (``image is None``) carries a ``decode_failure`` flag from the sampler.
  The preprocessor cannot preprocess absent pixels, so it propagates that flag,
  reports a quality score of ``0.0``, and yields a :class:`PreprocessedFrame`
  whose ``image`` is ``None`` (Requirement 2.5).
- **Offline, no AWS.** Pure CPU/OpenCV work on a local ndarray; importable and
  runnable with no AWS credentials (Requirement 19.4, Competition 4).

Usage::

    from vision.frame_sampler import SamplingPolicy, sample_frames
    from vision.image_preprocessor import preprocess_frame

    for frame in sample_frames(cap, meta, SamplingPolicy.every_n_frames(5)):
        pre = preprocess_frame(frame)
        if pre.ok and not pre.is_low_quality:
            detect(pre.image)          # a real, normalized ndarray
        else:
            record(pre.quality_flags)  # poor_input_quality / decode_failure surfaced
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from vision.frame_sampler import Frame
from vision.result_schema import QualityFlag, quality_flag

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

#: Default output size (width_px, height_px) the frames are normalized to so the
#: downstream detectors see a consistent resolution. Kept modest for the
#: classical path; the model path may re-letterbox to its own input size later.
DEFAULT_TARGET_SIZE: tuple[int, int] = (640, 480)

#: Default low-quality threshold on the composite quality score in ``[0.0, 1.0]``.
#: A frame scoring below this is flagged ``poor_input_quality`` (2.5). Chosen so
#: a sharp, well-exposed frame scores well above it and a heavily blurred or
#: badly-exposed frame scores below it; overridable per call / per deployment.
DEFAULT_LOW_QUALITY_THRESHOLD: float = 0.35

#: Laplacian variance at or above which the sharpness sub-score saturates to 1.0.
#: Laplacian variance is the standard "focus measure": low on blurred frames,
#: high on sharp ones. 500 is a conventional sharp/blurry crossover for 8-bit
#: grayscale content and keeps the sub-score a real, bounded function of pixels.
SHARPNESS_SATURATION: float = 500.0


@dataclass(frozen=True)
class PreprocessConfig:
    """Configurable knobs for preprocessing and quality scoring.

    All thresholds are explicit and overridable so the low-quality decision is a
    configured policy (Requirement 2.5) rather than a magic constant buried in
    the code.
    """

    #: Output (width_px, height_px) for the normalized frame.
    target_size: tuple[int, int] = DEFAULT_TARGET_SIZE
    #: Composite score below which the frame is flagged ``poor_input_quality``.
    low_quality_threshold: float = DEFAULT_LOW_QUALITY_THRESHOLD
    #: Laplacian-variance value at which the sharpness sub-score saturates to 1.0.
    sharpness_saturation: float = SHARPNESS_SATURATION
    #: Whether to apply denoising (``cv2.fastNlMeansDenoising``) to the grayscale
    #: frame. On by default; can be disabled for speed.
    denoise: bool = True

    def __post_init__(self) -> None:
        w, h = self.target_size
        if w < 1 or h < 1:
            raise ValueError(f"target_size must be positive, got {self.target_size!r}")
        if not 0.0 <= self.low_quality_threshold <= 1.0:
            raise ValueError(
                f"low_quality_threshold must be in [0.0, 1.0], got {self.low_quality_threshold!r}"
            )
        if self.sharpness_saturation <= 0:
            raise ValueError(
                f"sharpness_saturation must be > 0, got {self.sharpness_saturation!r}"
            )


DEFAULT_CONFIG = PreprocessConfig()


# --------------------------------------------------------------------------- #
# Output type
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PreprocessedFrame:
    """A preprocessed frame: normalized pixels plus a real quality score.

    Carries the preprocessing output and the measured quality of the *input*
    frame, consistent with the schema conventions in
    :mod:`vision.result_schema`:

    - ``image``: the preprocessed grayscale ndarray (resized, optionally
      denoised), or ``None`` when the source frame could not be decoded.
    - ``quality_score``: a real number in ``[0.0, 1.0]`` computed from genuine
      image statistics (Laplacian variance for sharpness, mean brightness for
      exposure). ``0.0`` for an undecodable frame. Never fabricated.
    - ``quality_flags``: any :class:`~vision.result_schema.QualityFlag`; carries
      ``poor_input_quality`` when the score is below the configured threshold,
      and propagates a ``decode_failure`` flag from the source frame.

    The identifying ``frame_ref``, ``frame_index``, and ``source_timestamp_ms``
    are preserved from the source :class:`~vision.frame_sampler.Frame` so the
    link to the video position survives preprocessing (Requirement 2.4).
    """

    frame_ref: str
    frame_index: int
    source_timestamp_ms: int | str
    #: Preprocessed grayscale frame, or ``None`` when decoding failed upstream.
    image: np.ndarray | None
    #: Real composite quality score in [0.0, 1.0], never fabricated (3.3).
    quality_score: float
    #: Measured sharpness (variance of the Laplacian) of the input, for the trace.
    sharpness: float
    #: Measured mean brightness (0-255) of the input, for the trace.
    brightness: float
    quality_flags: list[QualityFlag] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when preprocessing produced real pixels (source decoded)."""
        return self.image is not None

    @property
    def is_low_quality(self) -> bool:
        """True when a ``poor_input_quality`` flag is present."""
        return any(f["code"] == "poor_input_quality" for f in self.quality_flags)


# --------------------------------------------------------------------------- #
# Quality scoring (real image statistics, never fabricated)
# --------------------------------------------------------------------------- #


def _measure_sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian - the standard focus/blur measure.

    A sharp, in-focus frame has strong high-frequency edge content and therefore
    a high Laplacian variance; a blurred frame has low variance. Computed on real
    pixels via ``cv2.Laplacian``.
    """
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def _brightness_subscore(brightness: float) -> float:
    """Map mean brightness (0-255) to a [0,1] exposure sub-score.

    Well-exposed mid-tones (around 128) score ~1.0; very dark or very bright
    (blown-out) frames score toward 0.0. This is a real, deterministic function
    of the measured mean - a triangular response centered on mid-gray.
    """
    # Distance from ideal mid-gray, normalized to [0,1].
    distance = abs(brightness - 127.5) / 127.5
    return float(max(0.0, 1.0 - distance))


def _sharpness_subscore(sharpness: float, saturation: float) -> float:
    """Map Laplacian variance to a [0,1] sharpness sub-score (saturating)."""
    return float(min(1.0, sharpness / saturation))


def compute_quality_score(
    image: np.ndarray, config: PreprocessConfig = DEFAULT_CONFIG
) -> tuple[float, float, float]:
    """Compute a real composite quality score for a frame.

    Returns ``(quality_score, sharpness, brightness)`` where:

    - ``sharpness`` is the variance of the Laplacian (focus measure),
    - ``brightness`` is the mean pixel intensity on a 0-255 scale,
    - ``quality_score`` in ``[0.0, 1.0]`` combines a saturating sharpness
      sub-score and a mid-gray exposure sub-score.

    Every value is derived from genuine pixel statistics; nothing is fabricated
    (Requirements 2.5, 3.3). Accepts either a BGR or an already-grayscale frame.
    """
    gray = _to_gray(image)
    sharpness = _measure_sharpness(gray)
    brightness = float(gray.mean())
    sharp_sub = _sharpness_subscore(sharpness, config.sharpness_saturation)
    bright_sub = _brightness_subscore(brightness)
    # Weight sharpness more heavily than exposure: blur is the dominant failure
    # mode for detection, while exposure degrades it more gracefully.
    score = 0.7 * sharp_sub + 0.3 * bright_sub
    return float(max(0.0, min(1.0, score))), sharpness, brightness


# --------------------------------------------------------------------------- #
# Preprocessing stages (real OpenCV 5 operations)
# --------------------------------------------------------------------------- #


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Convert a BGR frame to grayscale; pass through an already-gray frame."""
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"unsupported image shape for grayscale conversion: {image.shape!r}")


def _preprocess_pixels(image: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    """Resize -> grayscale -> (optional) denoise, all via real OpenCV ops.

    Produces a normalized single-channel ``uint8`` frame of ``config.target_size``
    for the downstream detectors.
    """
    # Resize first so denoising cost is bounded by the (smaller) target size.
    resized = cv2.resize(image, config.target_size, interpolation=cv2.INTER_AREA)
    gray = _to_gray(resized)
    if config.denoise:
        # fastNlMeansDenoising operates on 8-bit single-channel images.
        gray = cv2.fastNlMeansDenoising(gray, h=7)
    return gray


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def preprocess_frame(
    frame: Frame, config: PreprocessConfig = DEFAULT_CONFIG
) -> PreprocessedFrame:
    """Preprocess one sampled :class:`~vision.frame_sampler.Frame`.

    Resizes, grayscales, and (by default) denoises the frame's pixels, computes a
    real quality score from the *input* image statistics, and attaches a
    ``poor_input_quality`` :class:`~vision.result_schema.QualityFlag` when that
    score falls below ``config.low_quality_threshold`` (Requirement 2.5).

    A frame that failed to decode upstream (``frame.image is None``) cannot be
    preprocessed: the sampler's ``decode_failure`` flag is propagated, the quality
    score is ``0.0``, and the returned image is ``None``. The decode failure is
    surfaced, never masked.

    The identifying fields (``frame_ref``, ``frame_index``,
    ``source_timestamp_ms``) are preserved so the link to the source video
    position survives preprocessing (Requirement 2.4).
    """
    if frame.image is None:
        # Nothing to preprocess; propagate the upstream decode failure honestly.
        flags = list(frame.quality_flags)
        if not any(f["code"] == "decode_failure" for f in flags):
            flags.append(
                quality_flag(
                    "decode_failure",
                    f"frame {frame.frame_index} carried no decodable image to preprocess",
                )
            )
        return PreprocessedFrame(
            frame_ref=frame.frame_ref,
            frame_index=frame.frame_index,
            source_timestamp_ms=frame.source_timestamp_ms,
            image=None,
            quality_score=0.0,
            sharpness=0.0,
            brightness=0.0,
            quality_flags=flags,
        )

    # Score the *input* quality from real statistics before normalization.
    quality_score, sharpness, brightness = compute_quality_score(frame.image, config)
    preprocessed = _preprocess_pixels(frame.image, config)

    # Carry over any flags the sampler already attached, then add the
    # poor_input_quality flag when the measured score is below the threshold.
    flags = list(frame.quality_flags)
    if quality_score < config.low_quality_threshold:
        flags.append(
            quality_flag(
                "poor_input_quality",
                (
                    f"quality score {quality_score:.3f} below threshold "
                    f"{config.low_quality_threshold:.3f} "
                    f"(sharpness={sharpness:.1f}, brightness={brightness:.1f})"
                ),
            )
        )

    return PreprocessedFrame(
        frame_ref=frame.frame_ref,
        frame_index=frame.frame_index,
        source_timestamp_ms=frame.source_timestamp_ms,
        image=preprocessed,
        quality_score=quality_score,
        sharpness=sharpness,
        brightness=brightness,
        quality_flags=flags,
    )


# --------------------------------------------------------------------------- #
# Self-validation (import-time smoke check against generated sample frames)
# --------------------------------------------------------------------------- #


def _make_sharp_frame() -> np.ndarray:
    """A high-contrast, well-exposed synthetic frame (should score high)."""
    rng = np.random.default_rng(0)
    # Mid-gray base with strong high-frequency texture -> high Laplacian variance.
    base = np.full((240, 320, 3), 120, dtype=np.uint8)
    noise = rng.integers(0, 120, (240, 320, 1), dtype=np.uint8)
    img = cv2.add(base, np.repeat(noise, 3, axis=2))
    # A few crisp edges.
    cv2.rectangle(img, (40, 40), (120, 120), (255, 255, 255), -1)
    cv2.rectangle(img, (180, 140), (260, 210), (0, 0, 0), -1)
    return img


def _make_blurry_dark_frame() -> np.ndarray:
    """A heavily blurred, under-exposed frame (should score low -> flagged)."""
    base = np.full((240, 320, 3), 12, dtype=np.uint8)  # very dark
    cv2.rectangle(base, (120, 100), (200, 160), (30, 30, 30), -1)
    # Strong blur crushes high-frequency content -> low Laplacian variance.
    return cv2.GaussianBlur(base, (31, 31), 0)


def _self_check() -> None:
    """Preprocess generated frames and assert the quality contract holds.

    Lightweight smoke check (real OpenCV 5, no mocks), not the property suite
    (task 4.6). Runs under ``python -m``.
    """
    # 1. A sharp, well-exposed frame scores high and is NOT flagged.
    sharp = Frame(
        frame_ref="smoke#frame_00000000",
        frame_index=0,
        source_timestamp_ms=0,
        image=_make_sharp_frame(),
        quality_flags=[],
    )
    pre_sharp = preprocess_frame(sharp)
    assert pre_sharp.ok, "sharp frame should preprocess to real pixels"
    assert pre_sharp.image is not None
    assert pre_sharp.image.shape == (DEFAULT_TARGET_SIZE[1], DEFAULT_TARGET_SIZE[0]), (
        f"expected normalized size {DEFAULT_TARGET_SIZE}, got {pre_sharp.image.shape!r}"
    )
    assert pre_sharp.image.ndim == 2, "preprocessed frame must be grayscale"
    assert 0.0 <= pre_sharp.quality_score <= 1.0
    assert not pre_sharp.is_low_quality, (
        f"sharp frame unexpectedly flagged (score={pre_sharp.quality_score:.3f})"
    )

    # 2. A blurred, dark frame scores low and IS flagged poor_input_quality (2.5).
    bad = Frame(
        frame_ref="smoke#frame_00000001",
        frame_index=1,
        source_timestamp_ms=100,
        image=_make_blurry_dark_frame(),
        quality_flags=[],
    )
    pre_bad = preprocess_frame(bad)
    assert pre_bad.ok
    assert pre_bad.quality_score < pre_sharp.quality_score, (
        "blurred/dark frame should score below the sharp frame"
    )
    assert pre_bad.is_low_quality, (
        f"blurred/dark frame should be flagged (score={pre_bad.quality_score:.3f})"
    )
    flag = next(f for f in pre_bad.quality_flags if f["code"] == "poor_input_quality")
    assert flag["detail"], "poor_input_quality flag must carry a non-empty detail"

    # 3. A decode-failure frame propagates the flag, scores 0.0, image is None.
    from vision.result_schema import quality_flag as _qf

    decode_failed = Frame(
        frame_ref="smoke#frame_00000002",
        frame_index=2,
        source_timestamp_ms=200,
        image=None,
        quality_flags=[_qf("decode_failure", "could not decode frame 2")],
    )
    pre_failed = preprocess_frame(decode_failed)
    assert not pre_failed.ok
    assert pre_failed.image is None
    assert pre_failed.quality_score == 0.0
    assert any(f["code"] == "decode_failure" for f in pre_failed.quality_flags)

    # 4. A configurable threshold changes the decision on the same frame.
    strict = PreprocessConfig(low_quality_threshold=0.99)
    pre_strict = preprocess_frame(sharp, strict)
    assert pre_strict.is_low_quality, (
        "a near-maximal threshold should flag even the sharp frame"
    )

    # 5. replace() keeps the dataclass usable/immutable as a schema record.
    _ = replace(pre_sharp, quality_flags=[])


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    _self_check()
    print(
        "[image_preprocessor] OK: real resize/denoise/grayscale, "
        "real quality score, poor_input_quality flag on low quality."
    )
