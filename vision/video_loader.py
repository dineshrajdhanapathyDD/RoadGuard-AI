"""Video loader: decode a validated road video with OpenCV 5.

Vision Pipeline Design, ``video_loader.py`` row of the component-responsibilities
table: given an S3 object ref or local path, this module opens a
``cv2.VideoCapture`` handle and exposes the container/codec/fps/resolution/
duration metadata, raising :class:`DecodeError` on failure so the caller can
surface it as a validation failure (Requirement 1.3). It is the first stage of
the pipeline and feeds the frame sampler (task 2.3).

Design constraints honored here:

- **Real OpenCV 5 only.** Metadata is read from the live capture handle
  (``cv2.VideoCapture.get``); nothing is fabricated (Requirement 2.1).
- **Honest unavailability.** Where the container genuinely does not report a
  value, it is recorded as the explicit :data:`UNAVAILABLE` sentinel rather than
  guessed - the same responsible-use posture as the result schema
  (Requirements 3.3, 13.4).
- **Fail loudly.** Anything that prevents decoding (missing file, unreadable
  container, unopenable handle, no decodable frames) raises
  :class:`DecodeError` (Requirement 1.4 / 1.3).
- **Offline, no AWS.** A local filesystem path runs end to end with no
  credentials. ``s3://`` refs are recognized at the type level and reserved for
  Milestone 6; resolving them is deliberately deferred.

Usage::

    from vision.video_loader import load_video, DecodeError

    with load_video("vision/samples/sample_road_pothole.mp4") as video:
        print(video.metadata.fps, video.metadata.resolution)
        ok, frame = video.capture.read()
"""

from __future__ import annotations

import os
from contextlib import AbstractContextManager
from dataclasses import dataclass
from types import TracebackType
from typing import Union

import cv2

# Explicit "we genuinely do not know" indicator. Mirrors result_schema's sentinel
# (Requirements 3.3, 13.4): we report unavailability rather than fabricating a value.
UNAVAILABLE = "unavailable"

# A value that is either a real reading or the explicit unavailable sentinel.
Unavailable = str  # the literal UNAVAILABLE at runtime


class DecodeError(RuntimeError):
    """Raised when OpenCV 5 cannot decode the video.

    Covers a missing/empty source, a container OpenCV cannot open, and a handle
    that opens but yields no decodable frames. The agent/backend surfaces this as
    a validation failure (Requirement 1.3) rather than starting an inspection.
    """


@dataclass(frozen=True)
class VideoMetadata:
    """Container/codec/fps/resolution/duration read from the capture handle.

    Every field is either a genuine reading from ``cv2.VideoCapture.get`` /
    the source path, or the explicit :data:`UNAVAILABLE` sentinel. No value is
    inferred or guessed.
    """

    # Container is derived from the source file extension (the only honest signal
    # available without a demuxer); UNAVAILABLE when there is no extension.
    container: Union[str, "Unavailable"]
    # Four-character codec code decoded from CAP_PROP_FOURCC; UNAVAILABLE when the
    # backend reports no usable fourcc.
    codec: Union[str, "Unavailable"]
    # Frames per second; UNAVAILABLE when the backend reports a non-positive fps.
    fps: Union[float, "Unavailable"]
    # (width_px, height_px); UNAVAILABLE when either dimension is non-positive.
    resolution: Union[tuple[int, int], "Unavailable"]
    # Frame count reported by the container; UNAVAILABLE when non-positive.
    frame_count: Union[int, "Unavailable"]
    # Duration in milliseconds, derived from frame_count / fps only when BOTH are
    # genuinely available; UNAVAILABLE otherwise (never guessed).
    duration_ms: Union[float, "Unavailable"]
    # The resolved source the handle was opened from (local path or s3:// ref).
    source: str


class LoadedVideo(AbstractContextManager):
    """An open OpenCV 5 capture plus its read metadata.

    Use as a context manager so the underlying ``cv2.VideoCapture`` is always
    released::

        with load_video(path) as video:
            ok, frame = video.capture.read()
    """

    def __init__(self, capture: "cv2.VideoCapture", metadata: VideoMetadata) -> None:
        self._capture = capture
        self._metadata = metadata
        self._released = False

    @property
    def capture(self) -> "cv2.VideoCapture":
        """The live OpenCV 5 capture handle (owned by this object)."""
        return self._capture

    @property
    def metadata(self) -> VideoMetadata:
        """Container/codec/fps/resolution/duration read from the handle."""
        return self._metadata

    def release(self) -> None:
        """Release the capture handle. Idempotent."""
        if not self._released:
            self._capture.release()
            self._released = True

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()


def _decode_fourcc(raw_fourcc: float) -> Union[str, "Unavailable"]:
    """Turn the numeric CAP_PROP_FOURCC into a 4-char codec code, honestly.

    Returns the explicit UNAVAILABLE sentinel when the backend reports 0 or the
    decoded characters are not printable (i.e. no real codec signal).
    """
    code = int(raw_fourcc)
    if code == 0:
        return UNAVAILABLE
    chars = [chr((code >> (8 * i)) & 0xFF) for i in range(4)]
    fourcc = "".join(chars).strip("\x00 ")
    if not fourcc or not all(ch.isprintable() for ch in fourcc):
        return UNAVAILABLE
    return fourcc


def _container_from_source(source: str) -> Union[str, "Unavailable"]:
    """Report the container from the source file extension, or UNAVAILABLE.

    The file extension is the only honest container signal available without
    running a demuxer, so we report it as such and do not claim more.
    """
    _, ext = os.path.splitext(source)
    ext = ext.lower().lstrip(".")
    return ext if ext else UNAVAILABLE


def _is_s3_ref(source: str) -> bool:
    return source.startswith("s3://")


def read_metadata(capture: "cv2.VideoCapture", source: str) -> VideoMetadata:
    """Read container/codec/fps/resolution/duration from an open capture handle.

    Pure with respect to the handle: it only *reads* properties and never seeks
    or consumes frames. Genuinely-unavailable values become the explicit
    UNAVAILABLE sentinel (Requirements 3.3, 13.4).
    """
    container = _container_from_source(source)
    codec = _decode_fourcc(capture.get(cv2.CAP_PROP_FOURCC))

    raw_fps = capture.get(cv2.CAP_PROP_FPS)
    fps: Union[float, str] = raw_fps if raw_fps and raw_fps > 0 else UNAVAILABLE

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    resolution: Union[tuple[int, int], str] = (
        (width, height) if width > 0 and height > 0 else UNAVAILABLE
    )

    raw_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_count: Union[int, str] = (
        int(raw_count) if raw_count and raw_count > 0 else UNAVAILABLE
    )

    # Duration is derived only when both inputs are genuinely available.
    if isinstance(fps, float) and isinstance(frame_count, int):
        duration_ms: Union[float, str] = (frame_count / fps) * 1000.0
    else:
        duration_ms = UNAVAILABLE

    return VideoMetadata(
        container=container,
        codec=codec,
        fps=fps,
        resolution=resolution,
        frame_count=frame_count,
        duration_ms=duration_ms,
        source=source,
    )


def load_video(source: Union[str, os.PathLike[str]]) -> LoadedVideo:
    """Open a validated video with OpenCV 5 and read its metadata.

    Args:
        source: A local filesystem path (``str`` or ``os.PathLike``). An
            ``s3://`` ref is recognized but not yet resolved (Milestone 6); passing
            one raises :class:`DecodeError` with an explicit reason today.

    Returns:
        A :class:`LoadedVideo` owning the live ``cv2.VideoCapture`` handle and its
        :class:`VideoMetadata`. Use it as a context manager to guarantee release.

    Raises:
        DecodeError: if the source is empty, an unresolved ``s3://`` ref, a
            missing local file, a container OpenCV cannot open, or a handle that
            opens but yields no decodable frame.
    """
    source_str = os.fspath(source)
    if not source_str:
        raise DecodeError("no video source provided (empty path)")

    if _is_s3_ref(source_str):
        # Reserved for Milestone 6 (S3-backed intake). Fail honestly today rather
        # than pretend to decode something we cannot reach offline.
        raise DecodeError(
            f"s3 object refs are not resolved in the offline path yet: {source_str!r}"
        )

    if not os.path.exists(source_str):
        raise DecodeError(f"video source does not exist: {source_str!r}")
    if not os.path.isfile(source_str):
        raise DecodeError(f"video source is not a file: {source_str!r}")
    if os.path.getsize(source_str) == 0:
        raise DecodeError(f"video source is empty: {source_str!r}")

    capture = cv2.VideoCapture(source_str)
    if not capture.isOpened():
        capture.release()
        raise DecodeError(f"OpenCV 5 could not open video: {source_str!r}")

    # Prove the container is genuinely decodable: a handle can "open" on a file
    # OpenCV cannot actually decode. grab() advances without copying pixels, so it
    # confirms decodability without consuming the first frame's data unnecessarily.
    if not capture.grab():
        capture.release()
        raise DecodeError(f"OpenCV 5 opened but could not decode any frame: {source_str!r}")

    # Rewind so the first frame is available to the downstream frame sampler.
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    metadata = read_metadata(capture, source_str)
    return LoadedVideo(capture, metadata)
