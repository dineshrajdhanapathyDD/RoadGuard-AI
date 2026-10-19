"""Optional fetcher for real, openly licensed road videos.

This script bundles NO copyrighted media. It documents curated public sources
and downloads only a URL you explicitly pass. Always verify the license of
anything you download; prefer CC0 / public-domain footage. Keep large binaries
out of git (or use Git LFS).

Usage::

    python -m vision.samples.fetch_samples                 # list sources
    python -m vision.samples.fetch_samples --url <URL> --out clip.mp4
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

SAMPLES_DIR = Path(__file__).resolve().parent

# Curated starting points for openly licensed road/dashcam footage. Verify the
# per-asset license before use; licensing on these platforms is per-item.
CURATED_SOURCES = [
    ("Pexels - free stock videos (per-clip license)", "https://www.pexels.com/search/videos/road/"),
    ("Pixabay - free stock videos (per-clip license)", "https://pixabay.com/videos/search/road/"),
    ("Wikimedia Commons - road/dashcam (check CC terms)", "https://commons.wikimedia.org/wiki/Category:Dashcam_videos"),
    ("Internet Archive - public-domain footage", "https://archive.org/details/movies"),
]


def list_sources() -> None:
    print("Openly licensed road-video sources (verify each asset's license):\n")
    for name, url in CURATED_SOURCES:
        print(f"  - {name}\n      {url}")
    print(
        "\nNo media is bundled with this repo. For reproducible offline tests, "
        "prefer `python -m vision.samples.make_samples`."
    )


def fetch(url: str, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[fetch_samples] downloading {url} -> {out}")
    # Only downloads the explicit, user-supplied URL.
    urllib.request.urlretrieve(url, out)  # noqa: S310 - user-provided URL by design
    print(f"[fetch_samples] wrote {out} ({out.stat().st_size} bytes)")
    print("[fetch_samples] REMINDER: verify the license before committing this file.")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", help="direct URL to an openly licensed video to download")
    parser.add_argument("--out", default="sample_fetched.mp4", help="output filename")
    args = parser.parse_args(argv)

    if not args.url:
        list_sources()
        return 0

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = SAMPLES_DIR / out_path
    fetch(args.url, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
