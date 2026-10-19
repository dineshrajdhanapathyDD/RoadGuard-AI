# RoadGuard AI - submission guide & competition-criteria mapping (task 16.6)

Target: OpenCV + AWS AI Competition 2026 (Agentic Vision Award), deadline
**October 26, 2026, 11:59 p.m. PT**.

## Reproducible run (no AWS credentials required)

The Mock_Runtime + real OpenCV 5 backbone runs the whole perception-decision-
action loop and the full test suite offline.

```bash
# Python backbone
python -m pip install -r requirements.txt
python -m vision.verify_opencv          # asserts OpenCV major version 5
python -m pytest -q                     # full suite (offline, no AWS)
python scripts/secret_scan.py           # no hardcoded secrets gate

# module self-checks (real OpenCV, deterministic mock, no AWS)
python -m agent.loop                    # bounded loop over a real sample clip
python -m vision.evaluation             # detection metrics known-answers
python -m agent.metrics                 # operational metrics

# Frontend dashboard (offline fixtures by default)
cd frontend && npm install && npm run build && npm run test
```

Everything above is credential-free and deterministic (Competition 4).

## Competition-criteria mapping

| Criterion | Where it is satisfied |
| --- | --- |
| Real OpenCV 5 performs substantive analysis | `vision/` pipeline (loader, sampler, preprocessor, classical detector, region analyzer); `verify_opencv` asserts major v5 |
| Agent selects an action from ACTUAL vision output | `agent/decider_service.py` (MockDecider branches on real AnalysisResult fields); `agent/loop.py` threads real perception into each decision (5.6) |
| A subsequent tool call is caused by the first analysis (Competition 1) | Demonstrated on real OpenCV: `analyze_frame` (surface_damage ~0.75) -> `analyze_region`; persisted in the trace (`agent/loop.py`, trace step 0 + the region step) |
| Fixed result + AI narrative does NOT count (Competition 2) | The decision is a function of the real AnalysisResult; the trace records before/after vision output, not a narrative |
| Failed analysis cannot become a finding (Competition 3) | `result_schema.failed_result` (empty detections by construction); `save_inspection_finding` rejects failed/flagged evidence; audited in `tests/test_failure_handling.py` |
| Ambiguous cases & tool failures handled safely | retry cap -> escalate/complete; quality flags; classical fallback on model-load failure; `tests/test_failure_handling.py` |
| A meaningful component runs on AWS | `infra/` CDK stack (S3, DynamoDB, API Gateway, Lambda, Fargate Vision_Worker) - authored + `cdk synth --strict`-verified; live deploy gated on approval |
| Automated tests pass | `pytest` (32 tests) + frontend Vitest (6) + module self-checks, all green |
| Evaluation reproducible | `vision/evaluation.py` (precision/recall/F1/IoU, known answers) + `agent/metrics.py` (operational), detection vs. agent quality kept separate |
| Limitations & responsible use documented | pixel-only measurements (no physical claims without calibration), explicit `unavailable` sentinels, model license/attribution in `vision/models/MODEL_MANIFEST.md` |

## Execution trace (proof artifact)

Run `python -m agent.loop` (or an inspection via `run_inspection`) and read the
persisted trace from the store: trace step 0 is the first real OpenCV analysis
(`before=unavailable`, `after=`the real AnalysisResult), and the next step is
the evidence-driven `analyze_region` whose `before_output` is that real result.
This before -> decision -> after chain is the Competition-1 proof; it is a
VERIFIED trace from real OpenCV, never an AI-generated explanation.

## AWS deploy & cost (when approved)

See `infra/DEPLOYMENT.md` for the resource list, the standard tag set
(Project/Environment/ManagedBy/Component/CostCenter on every resource), expected
cost categories, security controls, and teardown. No live deploy has been run;
it is gated on a confirmed Region + credentials + explicit spend approval.

## Known limitations / responsible use
- Classical detection confidence is a bounded heuristic (<= 0.85), not a
  validated detector; pixel areas are not physical measurements.
- The ONNX Model_Path is optional and may not load on every OpenCV build; the
  agent falls back to the classical path and records it (never fabricates model
  output).
- Live Bedrock reasoning and a real ONNX load require AWS credentials + a fetched
  model; both degrade safely to the mock/classical path offline.
- Phase B (unsafe-driving), AccessLens, live surveillance, facial/plate
  recognition, and OpenCV COOL are intentionally out of scope for this release.
