# RoadGuard AI

Agentic road-safety inspection platform for the OpenCV + AWS AI Competition 2026
(Agentic Vision focus, submission deadline October 26, 2026).

RoadGuard AI ingests a road video, runs a **real OpenCV 5** computer-vision
pipeline, and drives a bounded reasoning agent whose next action is determined
by genuine vision output. The full perception-decision-action loop and its test
suite run locally with no AWS credentials via a deterministic Mock_Runtime.

See the spec in `.kiro/specs/roadguard-ai/` (`requirements.md`, `design.md`,
`tasks.md`) for the authoritative requirements and architecture.

## Monorepo layout

```
roadguard-ai/
  vision/            Python OpenCV 5 pipeline (decode, sample, detect, annotate)
    verify_opencv.py   credential-free OpenCV-5 smoke check
    samples/           sample-video generator + fetch script (real-decode tests)
  agent/             Python agent layer (bounded loop, 7 tools, decider/store abstractions)
  frontend/          TypeScript dashboard (React + Vite + Tailwind + Recharts)
  infra/             Infrastructure as Code (AWS CDK/CloudFormation) - Milestone 6
  tests/             Python tests (pytest + Hypothesis property tests)
  pyproject.toml     Python project + exact pinned dependencies
  requirements.txt   pip mirror of runtime deps (exact pins)
  requirements-dev.txt  pip mirror of test/dev deps (exact pins)
```

## Pinned dependencies

All versions are pinned exactly (no open ranges) for reproducible deliverables.

Python (`pyproject.toml` / `requirements*.txt`):

| Package | Version | Role |
| --- | --- | --- |
| opencv-python | 5.0.0.93 | OpenCV 5 vision pipeline |
| numpy | 2.2.6 | array math |
| boto3 | 1.43.114 | AWS SDK (Milestone 6) |
| strands-agents | 1.61.1 | agent runtime SDK (Milestone 7) |
| pytest | 9.1.1 | test runner (dev) |
| hypothesis | 6.170.2 | property-based tests (dev) |
| onnx | 1.23.2 | Model_Path tooling (optional, Milestone 7) |

Frontend (`frontend/package.json`):

| Package | Version | Role |
| --- | --- | --- |
| react / react-dom | 19.3.1 | UI |
| vite | 8.3.4 | build/dev server |
| @vitejs/plugin-react | 6.1.2 | React plugin |
| tailwindcss / @tailwindcss/vite | 4.3.3 | styling |
| recharts | 3.10.1 | metrics charts |
| fast-check | 4.10.2 | property-based tests (TS) |
| vitest | 5.0.5 | test runner |
| typescript | 5.9.3 | types (see note below) |
| @testing-library/react | 16.3.3 | render tests |
| jsdom | 30.1.3 | test DOM |

**TypeScript pin note:** npm's `latest` tag for `typescript` is currently the
`7.x` native-compiler preview line. We pin the stable **5.9.3** for
compatibility with the React type definitions, the Vite React plugin, and
Vitest. Revisit when the 7.x toolchain ecosystem stabilizes.

## OpenCV 5 availability (verified)

The task flagged that an OpenCV 5.x prebuilt wheel might not be installable.
**Verified on this environment** (Windows, Python 3.13):

- `opencv-python==5.0.0.93` **is** published on PyPI as a prebuilt wheel and
  installs cleanly.
- After install, `import cv2; cv2.__version__` reports **`5.0.0`** (major 5).

So RoadGuard pins a genuine OpenCV 5 wheel - it does **not** fall back to
OpenCV 4 while claiming 5.x compliance. If you are on a platform or Python
version without a 5.x wheel, `vision/verify_opencv.py` reports the **actual**
installed version and whether it meets the major-version-5 requirement, and
exits non-zero (fails loudly) rather than passing silently.

## Quick start

### Python (vision + agent, credential-free)

```bash
python -m pip install -r requirements-dev.txt     # or: pip install -e ".[test,model]"

# OpenCV 5 smoke check (no AWS needed) - exits 0 only if major version is 5
python -m vision.verify_opencv

# Generate tiny license-clean sample road clips for real-decode tests
python -m vision.samples.make_samples

# Run the Python test suite
python -m pytest
```

### Frontend

```bash
cd frontend
npm install
npm run test        # vitest
npm run build       # type-check + production build
npm run dev         # local dev server
```

## Reproducibility and security

- The Mock_Runtime + real OpenCV 5 path (Milestones 1-5, 8) runs the full agent
  loop and test suite with **no AWS credentials** (Requirement 19.4, Competition 4).
- No credentials or secrets are committed; secrets are resolved at runtime from a
  managed source (Requirement 18.5). `.env*` and key files are git-ignored.
- Sample videos are generated locally (or fetched from documented, openly
  licensed sources) rather than committing third-party footage.
