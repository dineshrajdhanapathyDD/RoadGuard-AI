"""Generate tiny, license-clean synthetic road clips with OpenCV 5.

These are real, decodable MP4 files produced entirely by our own code - no
third-party footage - so the real-decode integration tests (task 2.6) are
reproducible offline. Run with::

    python -m vision.samples.make_samples

Each clip is short (~1-2 s) and small (320x240) by design.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

SAMPLES_DIR = Path(__file__).resolve().parent

WIDTH = 320
HEIGHT = 240
FPS = 15
SECONDS = 2
# mp4v is a widely available encoder in the opencv-python wheel.
FOURCC = cv2.VideoWriter_fourcc(*"mp4v")


def _road_background(frame_idx: int) -> np.ndarray:
    """A gray road surface with subtle lane markings that scroll to create motion.

    The lane dashes are deliberately kept to the LEFT edge and low-contrast so
    they do not dominate a centred road hazard: the classical detector keys on
    edges/texture, and we want a drawn pothole/crack to be the strongest real
    signal in the frame, not the lane paint.
    """
    img = np.full((HEIGHT, WIDTH, 3), 95, dtype=np.uint8)  # asphalt gray
    # subtle surface texture so detectors have something real to chew on
    noise = np.random.default_rng(frame_idx).integers(0, 16, (HEIGHT, WIDTH, 1), dtype=np.uint8)
    img = cv2.add(img, np.repeat(noise, 3, axis=2))
    # scrolling dashed lane line near the LEFT edge, low-contrast
    offset = (frame_idx * 12) % 40
    for y in range(-40 + offset, HEIGHT, 40):
        cv2.rectangle(img, (70, y), (76, y + 18), (165, 165, 165), -1)
    return img


def _write_clip(path: Path, hazard: str) -> Path:
    writer = cv2.VideoWriter(str(path), FOURCC, FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():  # pragma: no cover - encoder availability is env-dependent
        raise RuntimeError(f"OpenCV could not open a VideoWriter for {path}")
    try:
        for i in range(FPS * SECONDS):
            frame = _road_background(i)
            if hazard == "pothole":
                # A large, dark, irregular pothole with BROKEN high-variance
                # interior texture (real surface damage, not a flat shadow) near
                # the centre-right of the lane. Sized and textured so it survives
                # mp4 encoding and is the dominant hazard the classical path finds.
                cx, cy = 200, 140
                cv2.ellipse(frame, (cx, cy), (34, 26), 0, 0, 360, (22, 22, 22), -1)
                rng = np.random.default_rng(7)
                for _ in range(90):
                    x = int(rng.integers(cx - 30, cx + 30))
                    y = int(rng.integers(cy - 22, cy + 22))
                    if ((x - cx) / 34.0) ** 2 + ((y - cy) / 26.0) ** 2 <= 1.0:
                        val = int(rng.integers(5, 75))
                        cv2.circle(frame, (x, y), 1, (val, val, val), -1)
                cv2.ellipse(frame, (cx, cy), (34, 26), 0, 0, 360, (8, 8, 8), 2)
            elif hazard == "crack":
                # A long, thin, jagged dark crack (elongated, low-solidity shape
                # the classical path classifies as a crack).
                pts = np.array(
                    [[150, 40], [172, 95], [158, 140], [188, 200]], dtype=np.int32
                )
                cv2.polylines(frame, [pts], False, (15, 15, 15), 3)
            writer.write(frame)
    finally:
        writer.release()
    return path


def make_samples(out_dir: Path | None = None) -> list[Path]:
    """Write the three sample clips and return their paths."""
    out_dir = out_dir or SAMPLES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = {
        "sample_road_clean.mp4": "clean",
        "sample_road_pothole.mp4": "pothole",
        "sample_road_crack.mp4": "crack",
    }
    written: list[Path] = []
    for name, hazard in clips.items():
        written.append(_write_clip(out_dir / name, hazard))
    return written


def main() -> int:
    paths = make_samples()
    for p in paths:
        size = p.stat().st_size if p.exists() else 0
        print(f"[make_samples] wrote {p.name} ({size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
