"""Real OpenCV 5 decode smoke test over the generated sample clips.

Supports Task 1's "real-decode integration tests" intent: it generates the
sample videos and confirms OpenCV 5 can open them and read ordered frames with
sane metadata. Runs offline with no AWS credentials. The full timestamp-ordering
integration test lands with the frame sampler in task 2.6.
"""

from __future__ import annotations

import cv2
import pytest

from vision.samples.make_samples import make_samples


@pytest.fixture(scope="module")
def sample_clips(tmp_path_factory):
    out = tmp_path_factory.mktemp("samples")
    return make_samples(out)


def test_three_samples_are_generated(sample_clips):
    assert len(sample_clips) == 3
    for path in sample_clips:
        assert path.exists(), path
        assert path.stat().st_size > 0, f"{path} is empty"


def test_opencv5_decodes_each_sample(sample_clips):
    for path in sample_clips:
        cap = cv2.VideoCapture(str(path))
        try:
            assert cap.isOpened(), f"OpenCV could not open {path.name}"
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            assert fps > 0, f"{path.name} reports non-positive fps {fps}"
            assert width == 320 and height == 240, f"{path.name} {width}x{height}"

            ok, frame = cap.read()
            assert ok and frame is not None, f"could not read a frame from {path.name}"
            assert frame.shape == (240, 320, 3), frame.shape

            # read a few more frames to prove it is a real multi-frame clip
            read_count = 1
            for _ in range(5):
                ok, _ = cap.read()
                if not ok:
                    break
                read_count += 1
            assert read_count > 1, f"{path.name} decoded only one frame"
        finally:
            cap.release()
