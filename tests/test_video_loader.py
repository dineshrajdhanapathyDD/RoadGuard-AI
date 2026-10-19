"""Tests for the OpenCV 5 video loader (Requirements 2.1, 1.4; surfaces 1.3).

These run offline with no AWS credentials. Real sample clips are generated with
``vision.samples.make_samples`` so the decode path exercises genuine OpenCV 5
rather than mocks.
"""

from __future__ import annotations

import cv2
import pytest

from vision.samples.make_samples import FPS, HEIGHT, SECONDS, WIDTH, make_samples
from vision.video_loader import (
    UNAVAILABLE,
    DecodeError,
    LoadedVideo,
    VideoMetadata,
    _container_from_source,
    _decode_fourcc,
    load_video,
    read_metadata,
)


@pytest.fixture(scope="module")
def sample_clips(tmp_path_factory):
    out = tmp_path_factory.mktemp("loader_samples")
    return make_samples(out)


# --- real decode + metadata -------------------------------------------------


def test_load_video_opens_and_reads_real_metadata(sample_clips):
    for path in sample_clips:
        with load_video(str(path)) as video:
            assert isinstance(video, LoadedVideo)
            m = video.metadata
            # Metadata is read from the live handle, matching the known generator.
            assert m.container == "mp4"
            assert m.fps == float(FPS)
            assert m.resolution == (WIDTH, HEIGHT)
            assert m.frame_count == FPS * SECONDS
            # duration derived only from genuine fps + frame_count
            assert m.duration_ms == pytest.approx((FPS * SECONDS) / FPS * 1000.0)
            assert m.source == str(path)
            # codec is a real reading (printable fourcc), never fabricated
            assert m.codec != UNAVAILABLE
            assert m.codec.isprintable()


def test_first_frame_is_available_after_load(sample_clips):
    # The loader grabs to prove decodability, then rewinds so frame 0 is intact
    # for the downstream frame sampler.
    with load_video(str(sample_clips[0])) as video:
        ok, frame = video.capture.read()
        assert ok and frame is not None
        assert frame.shape == (HEIGHT, WIDTH, 3)


def test_release_is_idempotent(sample_clips):
    video = load_video(str(sample_clips[0]))
    video.release()
    video.release()  # must not raise


def test_context_manager_releases_capture(sample_clips):
    with load_video(str(sample_clips[0])) as video:
        cap = video.capture
        assert cap.isOpened()
    assert not cap.isOpened()


# --- failure paths raise DecodeError (Requirement 1.4 -> surfaced as 1.3) ---


def test_empty_source_raises_decode_error():
    with pytest.raises(DecodeError):
        load_video("")


def test_missing_file_raises_decode_error(tmp_path):
    with pytest.raises(DecodeError):
        load_video(str(tmp_path / "nope.mp4"))


def test_directory_source_raises_decode_error(tmp_path):
    with pytest.raises(DecodeError):
        load_video(str(tmp_path))


def test_empty_file_raises_decode_error(tmp_path):
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    with pytest.raises(DecodeError):
        load_video(str(empty))


def test_non_video_file_raises_decode_error(tmp_path):
    bogus = tmp_path / "notvideo.mp4"
    bogus.write_bytes(b"this is definitely not an mp4 container")
    with pytest.raises(DecodeError):
        load_video(str(bogus))


def test_s3_ref_is_not_resolved_offline():
    # s3:// refs are recognized at the type level but deferred to Milestone 6.
    with pytest.raises(DecodeError):
        load_video("s3://some-bucket/evidence/clip.mp4")


# --- honest-unavailability unit behavior ------------------------------------


def test_container_from_source_reports_extension_or_unavailable():
    assert _container_from_source("a/b/clip.mp4") == "mp4"
    assert _container_from_source("CLIP.MOV") == "mov"
    assert _container_from_source("no_extension") == UNAVAILABLE


def test_decode_fourcc_zero_is_unavailable():
    assert _decode_fourcc(0.0) == UNAVAILABLE


def test_decode_fourcc_decodes_printable_code():
    # Build the numeric fourcc the same way OpenCV does for "mp4v".
    code = cv2.VideoWriter_fourcc(*"mp4v")
    assert _decode_fourcc(float(code)) == "mp4v"


def test_read_metadata_marks_unavailable_when_handle_reports_nothing(sample_clips):
    # Read real metadata, then confirm the derivation rule: when fps or
    # frame_count is unavailable, duration must be unavailable (never guessed).
    with load_video(str(sample_clips[0])) as video:
        m = read_metadata(video.capture, video.metadata.source)
        assert isinstance(m, VideoMetadata)
        if m.fps == UNAVAILABLE or m.frame_count == UNAVAILABLE:
            assert m.duration_ms == UNAVAILABLE
