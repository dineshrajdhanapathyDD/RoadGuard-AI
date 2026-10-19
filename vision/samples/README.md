# Vision sample videos

Real-decode integration tests (`tests/`, Milestone 1 task 2.6) need at least one
genuine video file that OpenCV 5 can decode and sample into timestamped frames.

This directory ships a **generator** rather than committed binaries, so the repo
stays small and every clip is unambiguously license-clean (synthesized locally by
our own code with OpenCV). An optional **fetch script** documents real-world
road-video sources for richer manual testing.

## Option A - generate tiny synthetic clips (default, offline, deterministic)

```bash
python -m vision.samples.make_samples
```

Writes 3 short MP4 clips (~1-2 s, 320x240) into this directory:

- `sample_road_clean.mp4`   - a plain road surface, no hazards
- `sample_road_pothole.mp4` - a dark pothole-like blob on the road
- `sample_road_crack.mp4`   - a thin crack-like line on the road

These are real, decodable videos with moving content and correct timestamps.
They are produced by `make_samples.py`, contain no third-party footage, and are
safe to use and redistribute. They are intended for decode/sampling plumbing
tests, not as a detection-accuracy benchmark.

## Option B - fetch real road videos (optional, documented)

```bash
python -m vision.samples.fetch_samples            # prints curated sources
python -m vision.samples.fetch_samples --url <URL> --out my_clip.mp4
```

`fetch_samples.py` does not bundle any copyrighted media. It lists public,
openly licensed sources (for example short CC0 / CC-BY dashcam and road clips)
and will download a user-supplied URL only when you pass `--url`. Verify the
license of anything you download before committing it; prefer CC0 / public
domain. Do not commit large binaries - keep real clips out of git or use Git LFS.

## Why not commit real MP4s directly?

- Keeps the repository lightweight and clone-fast.
- Avoids shipping third-party footage with uncertain licensing.
- The synthetic clips are deterministic, which makes the integration tests
  reproducible on any machine with OpenCV 5 and no network access.
