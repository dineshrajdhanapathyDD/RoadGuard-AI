"""Frame sampler: sample frames from a decoded video and timestamp them.

Vision Pipeline Design, ``frame_sampler.py`` row of the component-responsibilities
table:

    | Input                         | Output                                   | Failure behavior                                             |
    | capture handle, sampling policy | ordered ``Frame`` records each with ``source_timestamp_ms`` | On unreadable frame, emits a frame with a decode ``Quality_Flag`` (2.5) |

This is the second stage of the pipeline: it consumes the live
``cv2.VideoCapture`` handle produced by :mod:`vision.video_loader` and yields an
ordered sequence of :class:`Frame` records, each carrying the decoded pixels and
the real ``source_timestamp_ms`` that links it to its position in the source
video (Requirement 2.4). The sampled frames feed the image preprocessor (task
4.1) and the detection paths.

Design constraints honored here:

- **Real OpenCV 5 only.** Frames and positions come from the live capture
  handle; nothing about the pixels or their count is fabricated (Requirement
  2.1).
- **Honest timestamps (never fabricated).** ``source_timestamp_ms`` is derived
  from the *real* frame index and the video's fps (``index / fps * 1000``). When
  fps is genuinely :data:`UNAVAILABLE`, the sampler falls back to the capture's
  own ``CAP_PROP_POS_MSEC`` reading; only when *that* is also unavailable does
  the timestamp become the explicit :data:`UNAVAILABLE` sentinel. No value is
  guessed (Requirements 2.4, 3.3, 13.4).
- **Decode failures are surfaced, not skipped.** When a frame within the
  sampling schedule cannot be decoded, the sampler emits a :class:`Frame` whose
  ``image`` is ``None`` and whose ``quality_flags`` carry a ``decode_failure``
  :class:`~vision.result_schema.QualityFlag` (Requirement 2.5). It does not
  silently drop the frame nor fabricate pixels.
- **Ordered, monotonic output.** Frames are produced in increasing frame-index
  order, so their real timestamps are non-decreasing by construction.
- **Configurable policy.** The sampling schedule is either every-N-frames or an
  fps-based interval (sample roughly ``target_fps`` frames per second), chosen
  via :class:`SamplingPolicy`.
- **Offline, no AWS.** Operates purely on a local capture handle.

Usage::

    from vision.video_loader import load_video
    from vision.frame_sampler import SamplingPolicy, sample_frames

    with load_video("vision/samples/sample_road_pothole.mp4") as video:
        policy = SamplingPolicy.every_n_frames(5)
        for frame in sample_frames(video.capture, video.metadata, policy):
            if frame.ok:
                analyze(frame.image)      # a real ndarray
            else:
                record(frame.quality_flags)  # decode_failure surfaced honestly
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Union

import cv2
import numpy as np

from vision.result_schema import (
    UNAVAILABLE,
    QualityFlag,
    quality_flag,
)
from vision.video_loader import VideoMetadata

# A timestamp is either a real millisecond reading or the explicit sentinel.
TimestampMs = Union[int, str]  # int milliseconds, or the literal UNAVAILABLE


@dataclass(frozen=True)
class SamplingPolicy:
    """A configurable frame-sampling schedule.

    Exactly one mode is active:

    - **every-N-frames**: keep frame indices ``0, step, 2*step, ...`` where
      ``step = frame_step``. Deterministic and independent of fps.
    - **fps-based interval**: keep roughly ``target_fps`` frames per second of
      source video. The concrete integer step is derived from the video's real
      fps at sampling time (``round(source_fps / target_fps)``, clamped to >= 1);
      if the source fps is :data:`UNAVAILABLE`, the fps-based policy falls back
      to every frame (step 1) rather than guessing a cadence.

    Build with the :meth:`every_n_frames` or :meth:`fps_interval` constructors.
    """

    #: Fixed frame step for the every-N-frames mode. ``None`` in fps mode.
    frame_step: int | None = None
    #: Target frames-per-second for the fps-based mode. ``None`` in frame mode.
    target_fps: float | None = None

    def __post_init__(self) -> None:
        both_set = self.frame_step is not None and self.target_fps is not None
        neither_set = self.frame_step is None and self.target_fps is None
        if both_set or neither_set:
            raise ValueError(
                "SamplingPolicy requires exactly one of frame_step or target_fps; "
                "use SamplingPolicy.every_n_frames(...) or SamplingPolicy.fps_interval(...)"
            )
        if self.frame_step is not None and self.frame_step < 1:
            raise ValueError(f"frame_step must be >= 1, got {self.frame_step}")
        if self.target_fps is not None and self.target_fps <= 0:
            raise ValueError(f"target_fps must be > 0, got {self.target_fps}")

    @classmethod
    def every_n_frames(cls, step: int) -> "SamplingPolicy":
        """Sample every ``step`` frames (indices 0, step, 2*step, ...)."""
        return cls(frame_step=step)

    @classmethod
    def fps_interval(cls, target_fps: float) -> "SamplingPolicy":
        """Sample roughly ``target_fps`` frames per second of source video."""
        return cls(target_fps=target_fps)

    def resolve_step(self, source_fps: Union[float, str]) -> int:
        """Resolve the concrete integer frame step for this video's real fps.

        For every-N-frames this is simply ``frame_step``. For the fps-based mode
        it is ``round(source_fps / target_fps)`` clamped to at least 1; when the
        source fps is genuinely :data:`UNAVAILABLE` the fps policy falls back to
        every frame (step 1) rather than fabricating a cadence.
        """
        if self.frame_step is not None:
            return self.frame_step
        # fps-based mode
        if not isinstance(source_fps, (int, float)) or source_fps == UNAVAILABLE:
            # Source fps unknown -> do not guess a cadence; sample every frame.
            return 1
        assert self.target_fps is not None  # guaranteed by __post_init__
        step = round(float(source_fps) / self.target_fps)
        return max(1, int(step))


@dataclass(frozen=True)
class Frame:
    """One sampled frame: real pixels (or a decode failure) plus its timestamp.

    Carries the decoded ``image`` ndarray, the real ``source_timestamp_ms`` that
    links it to its position in the source video (Requirement 2.4), its
    ``frame_index`` in the source stream, a stable ``frame_ref`` string, and any
    :class:`~vision.result_schema.QualityFlag`.

    When the frame could not be decoded, ``image`` is ``None`` and
    ``quality_flags`` carries a ``decode_failure`` flag (Requirement 2.5); the
    failure is surfaced rather than hidden.
    """

    frame_ref: str
    frame_index: int
    #: Real milliseconds from stream start, or the explicit UNAVAILABLE sentinel.
    source_timestamp_ms: TimestampMs
    #: The decoded BGR frame, or ``None`` when decoding failed.
    image: np.ndarray | None
    quality_flags: list[QualityFlag] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when the frame decoded to real pixels (no decode failure)."""
        return self.image is not None


def _frame_ref(source: str, frame_index: int) -> str:
    """Build a stable, ordered reference for a sampled frame.

    Zero-padded so lexical ordering matches numeric ordering, consistent with
    the zero-padded trace ordering used elsewhere in the design.
    """
    return f"{source}#frame_{frame_index:08d}"


def _timestamp_ms(
    frame_index: int,
    source_fps: Union[float, str],
    capture: "cv2.VideoCapture",
) -> TimestampMs:
    """Derive a real ``source_timestamp_ms`` for a frame, never fabricated.

    Preference order:

    1. ``frame_index / source_fps * 1000`` when the video reports a genuine fps.
       This is the exact, reproducible position of the sampled frame.
    2. The capture's own ``CAP_PROP_POS_MSEC`` reading when fps is
       :data:`UNAVAILABLE` but the backend still reports a real position.
    3. The explicit :data:`UNAVAILABLE` sentinel when neither signal exists -
       honesty over a guessed number (Requirements 3.3, 13.4).
    """
    if isinstance(source_fps, (int, float)) and source_fps != UNAVAILABLE and source_fps > 0:
        return int(round((frame_index / float(source_fps)) * 1000.0))

    # fps genuinely unavailable: fall back to the backend's real position reading.
    pos_msec = capture.get(cv2.CAP_PROP_POS_MSEC)
    if pos_msec and pos_msec > 0:
        return int(round(float(pos_msec)))

    # POS_MSEC can legitimately be 0.0 for the very first frame; treat index 0
    # with a real position of 0 as a genuine zero rather than unavailable.
    if frame_index == 0 and pos_msec == 0.0:
        return 0

    return UNAVAILABLE


def sample_frames(
    capture: "cv2.VideoCapture",
    metadata: VideoMetadata,
    policy: SamplingPolicy,
) -> Iterator[Frame]:
    """Yield ordered :class:`Frame` records from a decoded video per ``policy``.

    Walks the stream frame by frame from the start, keeping only the indices the
    resolved sampling step selects. Each kept index yields exactly one
    :class:`Frame`:

    - a successful decode carries the real pixels and a derived
      ``source_timestamp_ms`` (Requirement 2.4), or
    - a decode failure carries ``image=None`` and a ``decode_failure``
      :class:`~vision.result_schema.QualityFlag` (Requirement 2.5).

    Frames are produced in increasing index order, so their timestamps are
    non-decreasing by construction.

    Args:
        capture: A live OpenCV 5 capture handle (owned by the caller).
        metadata: The :class:`~vision.video_loader.VideoMetadata` for this
            capture; its ``fps`` drives timestamping and fps-based step
            resolution.
        policy: The :class:`SamplingPolicy` schedule.

    Yields:
        :class:`Frame` records in ascending ``frame_index`` order.
    """
    source_fps = metadata.fps
    step = policy.resolve_step(source_fps)
    source = metadata.source

    # Start from the beginning so output ordering/timestamps are deterministic,
    # regardless of where the loader left the read cursor.
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_index = 0
    while True:
        # grab() advances the decoder without copying pixels; for frames we are
        # not sampling this skips the expensive retrieve() entirely.
        grabbed = capture.grab()
        if not grabbed:
            # End of stream (or an unreadable tail). Stop walking; we never
            # fabricate frames beyond what the decoder yields.
            break

        if frame_index % step == 0:
            ok, image = capture.retrieve()
            timestamp = _timestamp_ms(frame_index, source_fps, capture)
            if ok and image is not None:
                yield Frame(
                    frame_ref=_frame_ref(source, frame_index),
                    frame_index=frame_index,
                    source_timestamp_ms=timestamp,
                    image=image,
                    quality_flags=[],
                )
            else:
                # The index was scheduled for sampling but could not be decoded.
                # Surface it honestly with a decode Quality_Flag (2.5) rather than
                # skipping silently or fabricating pixels.
                yield Frame(
                    frame_ref=_frame_ref(source, frame_index),
                    frame_index=frame_index,
                    source_timestamp_ms=timestamp,
                    image=None,
                    quality_flags=[
                        quality_flag(
                            "decode_failure",
                            f"OpenCV 5 could not decode frame {frame_index} of {source!r}",
                        )
                    ],
                )

        frame_index += 1


# --------------------------------------------------------------------------- #
# Self-validation (import-time smoke check against a generated sample clip)
# --------------------------------------------------------------------------- #


def _self_check() -> None:
    """Decode+sample a generated sample clip and assert ordered timestamps.

    Lightweight smoke check (real OpenCV 5, no mocks), not the property/
    integration suite (tasks 2.4/2.6). Runs under ``python -m``.
    """
    import tempfile
    from pathlib import Path

    from vision.samples.make_samples import FPS, SECONDS, make_samples
    from vision.video_loader import load_video

    with tempfile.TemporaryDirectory() as tmp:
        clips = make_samples(Path(tmp))
        clip = clips[0]

        # every-N-frames policy
        with load_video(str(clip)) as video:
            policy = SamplingPolicy.every_n_frames(5)
            frames = list(sample_frames(video.capture, video.metadata, policy))

        assert frames, "sampler produced no frames from a real clip"
        # all scheduled indices are multiples of the step, in ascending order
        indices = [f.frame_index for f in frames]
        assert indices == sorted(indices), "frame indices are not ascending"
        assert all(i % 5 == 0 for i in indices), f"unexpected sampled indices: {indices}"

        # timestamps are real and monotonic non-decreasing
        ts = [f.source_timestamp_ms for f in frames]
        assert all(isinstance(t, int) for t in ts), f"timestamps not all real ints: {ts}"
        assert ts == sorted(ts), f"timestamps not monotonic: {ts}"
        # first sampled frame is at index 0 -> 0 ms
        assert ts[0] == 0, f"expected first timestamp 0 ms, got {ts[0]}"
        # derived from real fps: index 5 at 15 fps -> 333 ms
        assert ts[1] == round((5 / FPS) * 1000.0), f"unexpected second timestamp {ts[1]}"
        # every sampled frame carries real pixels of the known sample size
        assert all(f.ok for f in frames), "a sample clip frame failed to decode"
        assert all(f.image.shape == (240, 320, 3) for f in frames)

        # fps-based policy: target 5 fps against 15 fps source -> step 3
        with load_video(str(clip)) as video:
            fps_policy = SamplingPolicy.fps_interval(5.0)
            assert fps_policy.resolve_step(video.metadata.fps) == 3
            fps_frames = list(sample_frames(video.capture, video.metadata, fps_policy))
        assert all(f.frame_index % 3 == 0 for f in fps_frames)
        # roughly target_fps * seconds frames (ceil of frame_count/step)
        assert len(fps_frames) == len(range(0, FPS * SECONDS, 3))


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    _self_check()
    print("[frame_sampler] OK: real decode + sampling with ordered, monotonic timestamps.")
