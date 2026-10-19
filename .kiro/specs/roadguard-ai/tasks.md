# Implementation Plan: RoadGuard AI

## Overview

This plan implements RoadGuard AI incrementally along the eight competition milestones. The backbone is the **Mock_Runtime + real OpenCV 5** path: everything in Milestones 1-5 and 8 runs end to end with no AWS credentials, so the perception-decision-action loop and the full test suite stay reproducible by default. AWS integration (Milestone 6) and the live Bedrock/Strands reasoning backends (Milestone 7) are layered behind the `DeciderService`, `Store`, and vision façade abstractions so the mock path remains the deterministic default.

Each milestone ends with a checkpoint that states its definition of done. Test sub-tasks are marked optional with `*` and include the property-based tests for the 24 correctness properties (Hypothesis in Python, fast-check in TypeScript), wired to the pure functions, the mock-driven agent loop, and the metamorphic critical-criterion checks per the design testing strategy. Core implementation sub-tasks are never optional.

The critical acceptance criterion - real OpenCV 5 output changing a subsequent agent tool call - is established as early as the schema and agent loop allow (Milestone 3) and then proven and persisted as its own milestone (Milestone 4).

## Tasks

- [x] 1. Establish monorepo structure, pinned dependencies, and OpenCV 5 verification
  - Create the polyglot monorepo layout: `vision/` and `agent/` Python packages, `frontend/` TypeScript app, `infra/` for IaC, `tests/` for Python tests, and a top-level README
  - Pin Python dependencies (OpenCV 5.x, NumPy, Hypothesis, boto3, Strands Agents SDK) in `pyproject.toml`/requirements with exact versions; pin frontend dependencies (React, Vite, Tailwind, Recharts, fast-check, Vitest) in `frontend/package.json`
  - Add a `verify_opencv.py` check that imports `cv2`, asserts the major version is 5, and fails loudly otherwise; wire it as a smoke check runnable without AWS
  - Add 1-3 small sample road videos under `vision/samples/` (or a documented fetch script) for real-decode integration tests
  - _Requirements: 2.1, 19.4, Competition 4_

- [x] 2. Milestone 1 - Verify OpenCV 5 and process a real sample road video
  - [x] 2.1 Define the structured result schema (`vision/result_schema.py`)
    - Implement `Measurement`, `QualityFlag`, `Detection`, `AnalysisResult`, `FrameRef`, `ComparisonResult` types and the `UNAVAILABLE` sentinel
    - Enforce responsible-use invariants at the type level: measurements always pixel-labeled, `physical="unavailable"`, failed results carry an empty detections list
    - _Requirements: 2.2, 2.4, 3.1, 3.2, 3.3, 6.2, 13.4_
  - [x] 2.2 Implement the video loader (`vision/video_loader.py`)
    - Decode a validated video with OpenCV 5, expose container/codec/fps/resolution/duration, raise `DecodeError` on failure
    - _Requirements: 2.1, 1.4_
  - [x] 2.3 Implement the frame sampler (`vision/frame_sampler.py`)
    - Sample frames per a sampling policy, attach `source_timestamp_ms` to each `Frame`, emit a decode `Quality_Flag` on unreadable frames
    - _Requirements: 2.1, 2.4, 2.5_
  - [ ]* 2.4 Write property test for the analysis output contract
    - **Property 4: Analysis output contract** (detections list, confidence in [0,1], timestamp equals source frame timestamp, method recorded)
    - **Validates: Requirements 2.2, 2.4, 6.1, 6.2, 8.1, 8.2, 4.3**
  - [ ]* 2.5 Write property test for responsible-use reporting integrity
    - **Property 7: Responsible-use reporting integrity** (pixel-labeled units, physical unavailable, no fabricated values)
    - **Validates: Requirements 3.1, 3.2, 3.3, 13.4**
  - [ ]* 2.6 Write integration test for real OpenCV 5 decode and sampling
    - Decode and sample the checked-in sample videos, assert timestamps are ordered and monotonic
    - _Requirements: 2.1_

- [x] 3. Checkpoint - Milestone 1 definition of done
  - OpenCV 5 is verified, a real sample video decodes and samples into timestamped frames, and the result schema is in place. Ensure all tests pass, ask the user if questions arise.

- [x] 4. Milestone 2 - Annotated visual evidence and structured analysis results
  - [x] 4.1 Implement the image preprocessor (`vision/image_preprocessor.py`)
    - Resize/denoise/grayscale, produce a quality score, propagate a `Quality_Flag` on low quality
    - _Requirements: 2.5_
  - [x] 4.2 Implement the classical detection path in `vision/damage_detector.py`
    - Canny edges, contour extraction/filtering, morphological open/close, texture analysis; produce pixel-space `Detection[]` with heuristic confidence for potholes and surface damage (plus cracks/debris where supported)
    - _Requirements: 4.1, 4.6, 2.2_
  - [x] 4.3 Implement annotated-frame evidence generation
    - Draw detected regions on the frame, write the annotated frame as evidence, set `evidence_ref` on the result (local filesystem store for the mock path)
    - _Requirements: 2.3_
  - [x] 4.4 Implement the region analyzer (`vision/region_analyzer.py`)
    - Region-scoped detection with a selected method; return a validation failure when the region falls outside frame bounds
    - _Requirements: 8.1, 8.2, 8.3_
  - [x] 4.5 Wire the vision façade (`VisionPipeline` protocol implementation)
    - Expose `analyze_frame`, `analyze_region`, `select_frames`, `compare_evidence` returning `result_schema` types; mark failed analyses so they can never become findings
    - _Requirements: 2.6, 6.3, Competition 3_
  - [ ]* 4.6 Write property test for quality flags on uncertain/failed analysis
    - **Property 5: Quality flags on uncertain or failed analysis**
    - **Validates: Requirements 2.5**
  - [ ]* 4.7 Write integration test for annotated-frame generation and evidence storage
    - Run the classical path on labeled samples, assert annotated frames are produced and `evidence_ref` resolves, and that supported hazard types are detected
    - _Requirements: 2.3, 4.6_
  - [ ]* 4.8 Write unit test for region bounds enforcement
    - **Property 15: Region bounds are enforced** (out-of-bounds region -> validation failure, no detection output)
    - **Validates: Requirements 8.3**

- [x] 5. Checkpoint - Milestone 2 definition of done
  - The classical path produces structured, pixel-labeled detections with annotated-frame evidence and region analysis, and failed results cannot become findings. Ensure all tests pass, ask the user if questions arise.

- [x] 6. Milestone 3 - Agent loop with deterministic mock decisions
  - [x] 6.1 Define agent state and decision types
    - Implement `InspectionState` (iteration, max_iterations, retries_by_tool, findings, review_queue, stop_reason) and `AgentDecision`
    - _Requirements: 5.1, 5.2_
  - [x] 6.2 Define the `DeciderService` protocol and implement `MockDecider` (`agent/decider_service.py`)
    - `MockDecider.decide(state, perception)` is a pure function of real `AnalysisResult` fields (confidence, quality flags, detection counts); identical inputs yield identical decisions
    - _Requirements: 19.2, 19.3, 19.5_
  - [x] 6.3 Define the agent-trace entry schema and the `Store` protocol with an in-memory fake
    - `TraceEntry` captures perception, reasoning, tool, before_output, after_output, decision, ts; implement an in-memory `Store` fake for credential-free runs
    - _Requirements: 14.1, 19.4_
  - [x] 6.4 Implement `analyze_frame` and `analyze_region` tools
    - Return detections, confidences, timestamp, quality flag, and method; return failure status with reason and no fabricated detections on failure
    - _Requirements: 6.1, 6.2, 6.3, 8.1, 8.2, 8.3_
  - [x] 6.5 Implement `select_additional_frames` and `compare_visual_evidence` tools
    - Ordered timestamped frame refs with a no-matching-frames empty-list status; comparison result requiring at least two valid evidence items
    - _Requirements: 7.1, 7.2, 7.3, 9.1, 9.2, 9.3_
  - [x] 6.6 Implement `save_inspection_finding` and `request_human_review` tools
    - Reject saves backed by failed/flag-rejected evidence; enqueue escalations with evidence refs and a non-empty reason
    - _Requirements: 10.1, 10.2, 10.3, 11.1, 11.2, 11.3_
  - [x] 6.7 Implement `complete_inspection` tool
    - Finalize the inspection, generate and persist the report and full trace, record zero findings explicitly when none exist
    - _Requirements: 12.1, 12.2, 12.3_
  - [x] 6.8 Implement the bounded agent loop and runtime resolver
    - One tool per iteration; enforce max iterations and per-tool retry cap; stop on complete/ max iterations/ unrecoverable failure; escalate-or-complete past the retry cap; restrict invocation to the seven tools; select `MockDecider` when credentials absent or `RUNTIME=mock`
    - _Requirements: 5.1, 5.3, 5.4, 5.5, 5.6, 5.7, 19.1, 19.2_
  - [ ]* 6.9 Write property test for exactly one tool per iteration
    - **Property 9: Exactly one tool per iteration**
    - **Validates: Requirements 5.1**
  - [ ]* 6.10 Write property test for bounded termination
    - **Property 10: Bounded termination** (iterations/retries within caps, loop terminates, valid stop reason)
    - **Validates: Requirements 5.2, 5.3, 5.4**
  - [ ]* 6.11 Write property test for tool restriction
    - **Property 11: Only the seven defined tools are invoked**
    - **Validates: Requirements 5.5**
  - [ ]* 6.12 Write property test for failed-evidence rejection
    - **Property 6: Failed evidence can never become a finding**
    - **Validates: Requirements 2.6, 10.2 (and Competition 3)**
  - [ ]* 6.13 Write property test for tool-failure escalation
    - **Property 13: Tool failure past the retry cap escalates or completes**
    - **Validates: Requirements 5.7**
  - [ ]* 6.14 Write property test for escalation preference on flagged observations
    - **Property 17: Escalation is preferred over saving for unresolved flagged observations**
    - **Validates: Requirements 11.1, 11.2, 11.3**
  - [ ]* 6.15 Write property tests for frame selection and evidence comparison
    - **Property 14: Additional-frame selection is ordered and timestamped**; **Property 16: Evidence comparison requires at least two items**
    - **Validates: Requirements 7.1, 7.2, 7.3, 9.1, 9.2, 9.3**
  - [ ]* 6.16 Write property test for mock runtime determinism and vision-driven decisions
    - **Property 24: Mock runtime is deterministic and vision-driven**
    - **Validates: Requirements 19.2, 19.3, 19.5**
  - [ ]* 6.17 Write unit tests for tool happy paths and edge cases
    - Tool input-schema happy paths, zero-findings completion, no-match frame selection, fewer-than-two evidence
    - _Requirements: 10.1, 12.3, 7.3, 9.3_

- [x] 7. Checkpoint - Milestone 3 definition of done
  - The bounded agent loop runs end to end under the deterministic Mock_Runtime with the in-memory store, invokes only the seven tools, and makes decisions from real vision output with no AWS credentials. Ensure all tests pass, ask the user if questions arise.

- [x] 8. Milestone 4 - Demonstrate a second vision-tool call caused by the first analysis (critical criterion)
  - [x] 8.1 Persist the before/reasoning/tool/after/decision trace per iteration
    - Append each iteration's perception input, reasoning step, invoked tool, before-output, after-output, and resulting decision in order to the store
    - _Requirements: 14.1, 14.2, 14.3_
  - [x] 8.2 Implement the decision branch that triggers a second vision tool from the first analysis
    - On low confidence / quality flag from the first `analyze_frame`, drive `analyze_region` (or a second analysis) as the next tool; record the causal link in the trace
    - DONE (stub/unit level): a low-confidence first `analyze_frame` drives `analyze_region` and the loop persists the causal before/after chain (verified via scripted stub + loop self-check); the initial perception is now traced as step 0; the re-save spin after a successful finding is fixed by retiring the consumed perception
    - VERIFIED (real OpenCV, this cycle): on the pothole sample, analyze_frame returns a real mid-confidence detection (0.754) and the decider drives a second real OpenCV analyze_region call at ts=1; the differing region result then changes the next action. Competition 1 demonstrated end to end on real OpenCV (8.2a/8.2b below)
    - _Requirements: 5.6, 14.2, 19.5, Competition 1, Competition 2_
  - [x] 8.2a Fix select_additional_frames loop progression
    - After `select_additional_frames`, the loop analyzes a newly selected frame (new real perception) rather than re-selecting the same one; verified on the real pipeline (ts=2 select -> ts=3 analyze_frame on a distinct frame). KNOWN FOLLOW-UP (M7/M8): when region investigation corroborates nothing on a mid-confidence detection, the decider currently re-widens rather than converging to a finding/escalation; tighten the decider policy (escalate persistently-uncorroborated mid-confidence observations per 11.3) in task 14.3/16.3. Does not block Competition 1.
    - _Requirements: 5.1, 5.4, 7.2_
  - [x] 8.2b Provide a detection-bearing real sample for the evidence-driven second call
    - The pothole sample yields real classical detections (pothole 0.696 + surface_damage ~0.75) on the first frame; the facade already detects on the RAW sampled frame (not the over-denoised preprocessor output, which erased detections) so the evidence-driven analyze_region fires on REAL OpenCV. Competition core acceptance test demonstrated end to end offline.
    - _Requirements: 2.2, 5.6, Competition 1_
  - [ ]* 8.3 Write the metamorphic property test for the critical criterion
    - **Property 12: Real vision output changes the subsequent action** - run two inspections identical up to a decision point with materially different real OpenCV outputs; assert the next tool invocation differs and the differing before-output/reasoning/decision is persisted
    - **Validates: Requirements 5.6, 14.2, 19.5 (and Competition 1, 2)**
  - [ ]* 8.4 Write property tests for finding evidence and report/trace round-trip
    - **Property 18: Saved findings reference real evidence**; **Property 20: Report and trace persistence round-trip** (ordering preserved)
    - **Validates: Requirements 10.3, 12.2, 13.3, 14.1, 14.3**
  - [ ]* 8.5 Write e2e test of the full inspection under the Mock_Runtime
    - Upload -> validate -> real OpenCV process -> agent loop -> report + trace + metrics, demonstrating vision output changing a subsequent tool call with no AWS credentials
    - _Requirements: Competition 1, Competition 4_

- [x] 9. Checkpoint - Milestone 4 definition of done
  - A persisted trace shows a first real OpenCV analysis causing a materially different second vision tool call, and the metamorphic test proves two different vision outputs yield two different next tool calls. Ensure all tests pass, ask the user if questions arise.
  - STATUS (verified this cycle, 24 tests + all self-checks green, no AWS): the persisted trace shows the first real OpenCV analyze_frame (surface_damage 0.754) causing a real analyze_region second call at ts=1, whose differing result then changes the next action -- Competition 1 demonstrated on real OpenCV (8.1/8.2/8.2a/8.2b done). The metamorphic property test (8.3) and the e2e test (8.5) remain OPTIONAL (`*`) and are not required for the checkpoint. KNOWN FOLLOW-UP (non-blocking, tracked in 8.2a notes -> task 14.3/16.3): on a mid-confidence detection that region analysis fails to corroborate, the decider re-widens instead of converging to a finding/escalation.

- [x] 10. Milestone 5 - React dashboard and inspection report
  - DESIGN INPUT (evaluated this cycle): `github.com/Leonxlnx/taste-skill` (MIT) was inspected for use as the frontend. It is NOT a web app - it is a collection of AI agent "skills" (SKILL.md design-guidance for Claude/Codex/Cursor; install.sh symlinks them into ~/.claude or ~/.codex), with no package.json/src/React. It therefore cannot be dropped in as the dashboard. Decision: use its frontend-design principles as GUIDANCE when building the dashboard in the existing frontend/ React+Vite+Tailwind app; do not merge its code or install its skills into the repo. Nothing from it has been integrated.
  - [x] 10.1 Scaffold the frontend app and API client
    - Vite + React + Tailwind app, typed API client for the read endpoints (inspection, report, trace, metrics), and shared TypeScript result/trace types
    - _Requirements: 16.1_
  - [x] 10.2 Implement the overview view
    - Aggregate statistics across inspections
    - _Requirements: 16.1_
  - [x] 10.3 Implement the video inspection view
    - Upload, validation feedback, 0-100 processing progress, and annotated-frame display
    - _Requirements: 16.2, 1.5_
  - [x] 10.4 Implement the agent-trace view
    - Render the ordered before-output / reasoning / tool / after-output / decision sequence that evidences the critical criterion
    - _Requirements: 16.3, 14.2_
  - [x] 10.5 Implement the inspection report view and inspection report model
    - Each finding with hazard type, evidence references, confidence, source timestamp; escalated observations with reasons; explicit zero-findings and unavailable indicators
    - _Requirements: 16.4, 13.1, 13.2, 13.4, 12.3_
  - [x] 10.6 Implement the evaluation metrics view with Recharts
    - Display precision/recall/F1/IoU, action accuracy, completion rate, escalation rate, latency, failure rate, and cost
    - _Requirements: 16.4, 15.4_
  - [ ]* 10.7 Write property tests for report completeness and operational-rate bounds
    - **Property 19: Report completeness**; **Property 22: Operational rates are bounded and count-consistent** (fast-check)
    - **Validates: Requirements 12.1, 12.3, 13.1, 13.2, 15.2**
  - [x]* 10.8 Write frontend render tests for all five views
    - React Testing Library snapshot/render tests for overview, inspection, agent-trace (before/reasoning/tool/after/decision), report, and metrics views
    - _Requirements: 16.1, 16.2, 16.3, 16.4, 15.4_

- [x] 11. Checkpoint - Milestone 5 definition of done
  - The dashboard renders overview, inspection (with progress and annotated frames), agent trace, report, and Recharts metrics against the read APIs, and the trace view surfaces the critical before/after/decision sequence. Ensure all tests pass, ask the user if questions arise.
  - STATUS (verified this cycle): built in frontend/ (React 19 + Vite + Tailwind v4 + Recharts), applying taste-skill design principles. Added src/types.ts (schemas mirroring the Python result/trace/report/metrics shapes with the explicit "unavailable" sentinel), src/fixtures.ts (deterministic offline SAMPLE data labelled as such), src/api.ts (MockApiClient default + HttpApiClient for M6, resolved via VITE_API_BASE_URL), and the five views under src/views/ wired into App with tab nav. The agent-trace view renders the before/reasoning/tool/after/decision sequence incl. the evidence-driven analyze_frame->analyze_region step. `npm run build` passes (type-check + bundle) and `npm run test` is green (6 tests incl. 5 view render tests, task 10.8). Runs with NO backend and NO AWS; Python suite still 24 passed.
  - REMAINING (optional): 10.7 property tests (report completeness Property 19, operational-rate bounds Property 22 via fast-check). Live read endpoints are wired behind HttpApiClient but exercised in Milestone 6.

- [x] 12. Milestone 6 - AWS integration via infrastructure-as-code
  - STATUS (authored + synth-verified this cycle, NOT deployed): CDK app in infra/ (Python; aws-cdk-lib==2.243.0, CDK CLI 2.1145.0) - app.py, roadguard_stack.py, handlers/ (dynamo_store.py + upload/inspection/read), Dockerfile.worker, DEPLOYMENT.md. Provisions S3 (BPA+SSE+TLS-only+destroyable), DynamoDB single table + GSI1/2/3 (SSE+PITR), 3 Lambda handlers with least-privilege grant*(), API Gateway (HTTPS/TLS-only, access logs, throttling), and the Fargate Vision_Worker. `cdk synth --strict` passes; verified in the template: S3 BlockPublicAccess(all), DynamoDB SSE+PITR, S3 aws:SecureTransport deny, GSI1/2/3.
  - AWS TAGS (as requested): Tags.of(app) applies Project=roadguard-ai, Environment=<env>, ManagedBy=cdk, Component=roadguard-ai-backend, CostCenter=roadguard-ai to EVERY resource (verified present on 31 resources in synth). They ship with any deploy.
  - DEPLOY GATE: a live `cdk deploy` is intentionally NOT run - it creates billable resources and needs a confirmed Region + credentials + explicit spend approval. 12.6-12.9 (optional tests) and the live-deploy checkpoint remain open until deploy is approved.
  - [x] 12.1 Author the IaC template skeleton (`infra/`)
    - CDK (or CloudFormation) app provisioning S3 (private, Block Public Access, SSE), DynamoDB single table + GSI1/GSI2/GSI3 (SSE, PITR), API Gateway (TLS only, request validation), Lambda, CloudWatch, Bedrock access, and the Fargate Vision_Worker
    - _Requirements: 17.1, 17.2, 17.3_
  - [x] 12.2 Implement the DynamoDB-backed `Store`
    - Single-table writes/reads for inspection/finding/trace/review/metrics with zero-padded trace ordering and bounded retries with timeouts
    - _Requirements: 13.3, 14.3, 18.7_
  - [x] 12.3 Implement the S3 evidence store and presigned uploads
    - Write annotated frames to `evidence/<inspection_id>/...`; `upload_handler` issues a presigned PUT scoped to one key in the private bucket
    - _Requirements: 1.1, 2.3, 18.2, 18.4_
  - [x] 12.4 Implement the backend Lambda handlers
    - `upload_handler` (presign + validate container/codec/duration/size, record reason on failure, record metadata and create inspection on success), `inspection_handler` (start/launch Fargate task, status), `read_handler` (report/trace/metrics by id)
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 13.3, 14.3, 15.4_
  - [x] 12.5 Containerize the Vision_Worker and wire least-privilege IAM, secrets, and CloudWatch
    - Dockerfile running the real OpenCV pipeline + agent on Fargate; per-component least-privilege roles; runtime secret resolution; CloudWatch logs/metrics for inspections and tool invocations
    - _Requirements: 17.3, 18.1, 18.3, 18.5, 18.6_
  - [ ]* 12.6 Write property test for video validation soundness and metadata recording
    - **Property 1: Video validation decision is sound**; **Property 2: Successful validation records required metadata**; **Property 3: Progress is bounded**
    - **Validates: Requirements 1.2, 1.3, 1.4, 1.5**
  - [ ]* 12.7 Write property test for bounded AWS retries
    - **Property 23: Bounded AWS retries** (fault injection on a mocked AWS client)
    - **Validates: Requirements 18.7**
  - [ ]* 12.8 Write IaC synth and policy tests
    - `cdk synth` snapshot asserting the required resources; policy assertions for least privilege, Block Public Access, encryption + TLS-only, and destroy support
    - _Requirements: 17.2, 17.5, 18.1, 18.2, 18.3_
  - [ ]* 12.9 Write integration tests for presigned upload and observability
    - Presigned-upload accept vs non-presigned reject; CloudWatch log/metric emission during an inspection
    - _Requirements: 18.4, 18.6_
  - [x] 12.10 Write deployment, cost, security, and cleanup documentation
    - Document provisioned resources, costs, security controls, and the teardown procedure
    - _Requirements: 17.4, 17.5_

- [x] 13. Checkpoint - Milestone 6 definition of done
  - The stack deploys from IaC with private S3, encrypted DynamoDB, presigned uploads, least-privilege roles, and a containerized Vision_Worker; backend handlers serve the dashboard; the Mock_Runtime path still runs offline. Ensure all tests pass, ask the user if questions arise.
  - STATUS: the stack is AUTHORED and `cdk synth --strict`-verified with all required security properties and the AWS tag set; the Mock_Runtime path still runs offline (24 tests green). This checkpoint stays OPEN until a live `cdk deploy` is approved and run (Region + credentials + spend), which is the only part not yet done.
  - DEPLOYED (live, this cycle): account 466742534146, user DD-IAM-ADMIN, region us-east-1, env=dev. `cdk bootstrap` (no changes) + `cdk deploy` succeeded - 93 resources, CREATE_COMPLETE in ~167s. API endpoint: https://7p1i7iqoyb.execute-api.us-east-1.amazonaws.com/dev/ . VERIFIED live: GET /overview -> 200 JSON; POST /uploads -> 200 with a presigned PUT URL scoped to an uploads/ key; DynamoDB RoadGuard-dev ACTIVE with GSI1/2/3 and SSE enabled; S3 evidence bucket with Block Public Access all-true; tags Project/Environment/ManagedBy/CostCenter present. Estimated run cost ~$35-40/mo (NAT Gateway ~$33 of it). Teardown: `cd infra && npx aws-cdk@2.1145.0 destroy -c environment=dev`.   - END-TO-END CLOUD RUN VERIFIED (this cycle): the real Vision_Worker image is now built from Dockerfile.worker (vision/ + agent/) and pushed to ECR; the Fargate task runs vision.worker_main. A real inspection ran on Fargate exit code 0: verified OpenCV 5 in-container, downloaded the uploaded video from S3, ran the pipeline + agent loop, and PERSISTED the trace to DynamoDB (META status=completed, 10 trace steps). The critical criterion is proven on real AWS: ts=0 analyze_frame -> real surface_damage detection 0.754, ts=1 analyze_region driven by that 0.754 output (before_conf=0.754), then select_additional_frames on a new frame. Fixes applied during bring-up: worker OpenCV verify import (check_opencv_version), and a float->Decimal sanitizer in the DynamoDB store (trace confidences). Offline suite still 32 passed.
  - FRONTEND WIRED TO LIVE DATA (this cycle): read_handler /overview + /metrics now return REAL aggregates from DynamoDB (GSI1 query; Decimal->number JSON); verified live: /overview -> 3 inspections (1 completed), /report -> 10 trace steps, /trace -> the real steps incl. ts=1 analyze_region before_conf=0.754. The dashboard now targets the live API via VITE_API_BASE_URL (.env.production + .env.local), and the production bundle embeds the endpoint (verified). Tests stay offline/deterministic via .env.test (blank base URL -> MockApiClient); frontend 6/6 pass, build green. `cd frontend && npm run dev` (or serve dist/) shows real inspections, the agent trace, and reports in the browser. FIX: HttpApiClient stored an unbound `fetch` and called it as this.fetchImpl(...), which throws "Illegal invocation" in the browser; now defaults to window.fetch.bind(window) (still injectable for tests). Rebuilt; 6/6 frontend tests pass.
  - REMAINING FOLLOW-UPS: two deprecation warnings (aws_logs logGroup, ecs containerInsights); wire inspection_handler to ecs:RunTask so an upload auto-triggers the worker (currently launched explicitly/verified).

- [ ] 14. Milestone 7 - Bedrock and optional Strands reasoning backends
  - STATUS (authored + verified offline this cycle, 24 tests + self-checks green, no AWS): 14.1 - vision/models/MODEL_MANIFEST.md pins `cvtechniques/road-damage-detection-yolov11` (YOLOv11s/RDD2022), license VERIFIED CC-BY-4.0 with attribution, weights git-ignored + documented fetch, plus vision/models/verify_model.py CI check (passes with artifact absent). 14.2 - Model_Path `detect_model()` loads the pinned ONNX via cv2.dnn.readNetFromONNX, decodes YOLO output with NMS, raises ModelLoadError on absent/unparseable model. 14.3 - the facade falls back to the Classical_Path on ModelLoadError for both analyze_frame and analyze_region, recording the fallback honestly (result reports method=classical + a poor_input_quality flag naming the model-load failure; never model detections that did not run) (R4.5). 14.4 - BedrockDecider implements DeciderService with lazy Strands/Bedrock imports (module still imports with no AWS); reasons over the SAME real perception, constrains the choice to the closed seven-tool vocabulary, and degrades to the deterministic MockDecider on any error/out-of-vocabulary response. resolve_decider now returns BedrockDecider only when RUNTIME!=mock AND credentials resolve (botocore, no network), else MockDecider (default).
  - LIVE-ONLY (not exercised offline): a real Bedrock reasoning call and a real ONNX model load require AWS credentials + a fetched model; both degrade safely to the mock/classical path here. 14.5/14.6 (optional tests) and 14.7 (optional Strands decider) remain open.
  - [x] 14.1 Source, pin, and license-verify the ONNX road-damage model
    - Record version, source URL, and verified license in `vision/models/MODEL_MANIFEST.md`; add a CI check that the pinned artifact matches the manifest
    - _Requirements: 4.4_
  - [x] 14.2 Implement the Model_Path (DNN) in `vision/damage_detector.py`
    - Load the pinned ONNX model via `cv2.dnn.readNetFromONNX`, produce bounding boxes and per-detection confidence; raise `ModelLoadError` on load failure and record it
    - _Requirements: 4.2, 4.6_
  - [x] 14.3 Wire agent method selection and classical fallback on model-load failure
    - Agent selects classical or model per frame/region and records the method; on `ModelLoadError` it records the failure and re-runs the analysis on the Classical_Path
    - _Requirements: 4.3, 4.5_
  - [x] 14.4 Implement `BedrockDecider` behind the `DeciderService` protocol
    - Default runtime via Strands + Amazon Bedrock; resolves when credentials are present and `RUNTIME != mock`, reasoning over the same `AnalysisResult` decision surface as the mock
    - _Requirements: 19.1_
  - [ ]* 14.5 Write property test for model-load fallback
    - **Property 8: Model-load failure falls back to the classical path** (fault injection on model load; trace records the fallback)
    - **Validates: Requirements 4.5**
  - [ ]* 14.6 Write smoke tests for model path, classical path stages, and default runtime resolution
    - Classical path stages present; model path loads the pinned model; Bedrock resolves as the default runtime when credentials are present
    - _Requirements: 4.1, 4.2, 4.4, 19.1_
  - [ ]* 14.7 Implement and test the optional Strands Decider integration
    - Plug an alternative `StrandsDecider` into the same `DeciderService` protocol; keep it fully optional so the Mock_Runtime remains the default
    - _Requirements: 19.1_

- [x] 15. Checkpoint - Milestone 7 definition of done
  - The pinned, license-verified ONNX model loads on the Model_Path with classical fallback on load failure, and `BedrockDecider` works as the default live runtime behind the decider abstraction while the Mock_Runtime stays the reproducible default. Ensure all tests pass, ask the user if questions arise.
  - STATUS (verified this cycle, offline): Model_Path + classical fallback (R4.5) and BedrockDecider-behind-the-seam are implemented and verified to the extent possible without AWS - the model-load path raises ModelLoadError and falls back to classical with an honest record; BedrockDecider resolves only with credentials and degrades to the mock otherwise. Mock_Runtime remains the reproducible default (24 tests green). The actual live Bedrock reasoning call and a real ONNX load are credential/artifact-gated and not run here.

- [x] 16. Milestone 8 - Evaluation, failure-handling hardening, and submission prep
  - [x] 16.1 Implement detection evaluation metrics (`vision/evaluation.py`)
    - Compute precision, recall, F1, and IoU against ground truth; handle empty sets as defined known answers
    - _Requirements: 15.1_
  - [x] 16.2 Implement operational metrics computation and persistence
    - Agent action accuracy, completion rate, escalation rate across inspections; per-inspection latency, failure rate, and measured AWS cost persisted to the metrics item
    - _Requirements: 15.2, 15.3_
  - [x] 16.3 Harden failure handling end to end
    - Ensure every failure path from the design error-handling table (decode, low quality, failed analysis, model load, region bounds, <2 evidence, retry-cap, unrecoverable, AWS op, missing value) is recorded in the trace/inspection record and surfaced honestly
    - _Requirements: 2.5, 2.6, 4.5, 5.4, 5.7, 8.3, 9.3, 18.7, 3.3, 13.4_
  - [ ]* 16.4 Write property test for detection-metric bounds and known answers
    - **Property 21: Detection metrics are bounded and correct on known answers** (identical sets -> 1.0, disjoint -> 0.0)
    - **Validates: Requirements 15.1**
  - [ ]* 16.5 Write integration test for per-inspection latency/failure-rate/cost recording
    - Assert the metrics item records latency, failure rate, and cost after an inspection
    - _Requirements: 15.3_
  - [x] 16.6 Prepare submission deliverables
    - Verify the full Mock_Runtime loop + test suite run green with no AWS credentials; add a secret-scan CI gate; document reproducible-run instructions and competition-criteria mapping targeting the October 26, 2026 deadline
    - _Requirements: 18.5, 19.4, Competition 3, Competition 4, Competition 5_

- [x] 17. Final checkpoint - Milestone 8 definition of done
  - Evaluation metrics compute within bounds against ground truth, all failure paths are recorded and surfaced honestly, and the reproducible Mock_Runtime deliverables run the loop and full test suite green with no AWS credentials. Ensure all tests pass, ask the user if questions arise.
  - STATUS (verified this cycle, no AWS): 16.1 vision/evaluation.py (IoU + precision/recall/F1 with defined known answers, all bounded, self-check green); 16.2 agent/metrics.py (action accuracy, completion, escalation rate, latency, failure rate; aws_cost/detection metrics stay explicit UNAVAILABLE; detection vs agent quality kept separate; persisted via Store); 16.3 tests/test_failure_handling.py audits every error-handling-table row (decode flag, poor-quality flag, failed!=finding, model-load->classical fallback recorded, region bounds, <2 evidence, retry-cap escalate/complete, explicit UNAVAILABLE) - 8 tests; 16.6 scripts/secret_scan.py gate (clean) + SUBMISSION.md (reproducible run + competition-criteria mapping + execution-trace proof). Full suite 32 passed; verify_opencv + all self-checks green.
  - REMAINING (optional): 16.4/16.5 property/integration tests. The one NON-code item outstanding for the competition is the <=5-minute demo video (a deliverable, not a task).

- [ ] 18. Milestone 9 - Amazon Bedrock AgentCore integration (live deploy, mode B)
  - DEPLOYED LIVE + VERIFIED (this cycle, Option A - agent on AgentCore, vision stays on Fargate): account 466742534146, us-east-1. Used the CURRENT supported tooling (the Python starter-toolkit is deprecated): @aws/agentcore CLI v0.31.1 + AgentCore Python SDK (bedrock-agentcore 1.23.0, @app.entrypoint) + Strands (strands-agents 1.48.0). Project agentcore_app/roadguardagent; the RoadGuard decision agent is app/roadguard/main.py (BedrockAgentCoreApp entrypoint reasoning over a REAL AnalysisResult, constrained to the seven tools; NO OpenCV in the image - cv2 not loaded - so the agent container is small, heavy vision stays on the Fargate Vision_Worker from M6). CloudFormation AgentCore-roadguardagent-default = CREATE_COMPLETE; runtime roadguardagent_roadguard = READY (ARN runtime/roadguardagent_roadguard-4bSl2n6OjU). VERIFIED by live invoke via bedrock-agentcore data plane: PING -> ok; DECIDE with a real 0.75 surface_damage perception -> analyze_region with honest reasoning ('moderately confident 0.75<0.80; corroborating before saving'). AgentCore Observability/OTEL is wired via aws-opentelemetry-distro in the project pyproject. This is the managed-Runtime production agentic-vision path for the Agentic Vision rubric.
  - DONE: 18.0 (design/spec), 18.1 (verified current SDK/CLI/Region), 18.2 (packaged Strands agent for AgentCore Runtime), 18.3 (AgentCore entrypoint behind the DeciderService seam - Mock default, Bedrock when creds+RUNTIME!=mock), 18.4 (Observability via OTEL), 18.8 (live acceptance: real-perception decision on the managed Runtime).
  - REMAINING/OPTIONAL: 18.5 AgentCore Evaluations (CLI `agentcore run`/evals - optional), 18.6 Gateway (CONDITIONAL - direct Fargate tool calls suffice; add only with a concrete need), 18.7 finalize AgentCore cost/security/cleanup docs. Teardown: `cd agentcore_app/roadguardagent && agentcore destroy`. COST NOTE: the AgentCore Runtime + the M6 stack (NAT ~$33/mo) are BOTH live and billing.
  - Scope decision (user, this cycle): FULL LIVE DEPLOY. Requires a confirmed AWS account + Region and incurs real cost (AgentCore Runtime, ECR image, CloudWatch/observability, Bedrock invocations). Deferred to a later cycle pending Region + credentials/spend go-ahead; build is mock-verified first, then one live deploy runs the acceptance test.
  - Verified facts (web, this cycle): AgentCore is GA (Oct 2025); deploy Strands via the `bedrock-agentcore-starter-toolkit` + `bedrock-agentcore` SDK (`BedrockAgentCoreApp`), or direct code deployment. Runtime is NOT in every Region - available in US East (N. Virginia), US East (Ohio), US West (Oregon), AP (Mumbai, Singapore, Sydney, Tokyo), EU (Frankfurt, Ireland), plus later additions. Re-verify SDK/IAM/Region before deploying.
  - [ ] 18.0 Update design.md and tasks.md for AgentCore before implementing
    - Add an AgentCore section to design.md: Runtime hosting of the Strands agent behind the existing `DeciderService` seam, the validated invocation interface, Observability trace export alongside the DynamoDB trace, Evaluations, optional Gateway (only if justified), and IAM/cost/cleanup. Keep the Mock_Runtime the offline default.
    - _Requirements: 17.1, 19.1, 19.2, 19.4_
  - [ ] 18.1 Verify current AgentCore SDKs, Region availability, and IAM requirements
    - Pin `bedrock-agentcore` + `bedrock-agentcore-starter-toolkit` versions; confirm the chosen Region supports Runtime; document the execution/role IAM and Bedrock model access needed
    - _Requirements: 17.1, 18.1, 18.5_
  - [ ] 18.2 Package the Strands agent for AgentCore Runtime with a validated invocation interface
    - Wrap the agent entrypoint in `BedrockAgentCoreApp`; accept a validated inspection-request payload; preserve bounded iterations, the seven-tool restriction, retries, and human-review escalation; keep agent execution separate from the Fargate video worker; support timeouts, service errors, and safe retries; retain the Mock_Runtime for local dev
    - _Requirements: 5.1, 5.2, 5.5, 5.7, 11.x, 18.7, 19.1, 19.4_
  - [ ] 18.3 Add an AgentCoreDecider behind the DeciderService seam
    - Implement a decider that runs the agent via AgentCore Runtime when configured; selected by the runtime resolver only when credentials/Region are viable; falls back to Mock_Runtime otherwise; never relabels mock output as live
    - _Requirements: 19.1, 19.2_
  - [ ] 18.4 Configure AgentCore Observability
    - Export the perception -> decision -> tool -> subsequent-output -> outcome sequence to AgentCore Observability where supported, in addition to the application DynamoDB trace; never claim a successful trace without verifying it
    - _Requirements: 14.1, 14.2, 14.3, 18.6_
  - [ ] 18.5 Configure AgentCore Evaluations where supported
    - Assess task completion, correct tool selection, and expected action sequences; keep the OpenCV detector evaluation separate (precision/recall/F1/IoU via vision/evaluation.py); never invent evaluation results
    - _Requirements: 15.1, 15.2_
  - [ ] 18.6 (Conditional) Expose narrowly-scoped vision operations via AgentCore Gateway
    - ONLY if Gateway provides a genuine benefit over direct tool calls. If adopted: expose analyze_frame/analyze_region/select_additional_frames/compare_visual_evidence as authenticated, input-validated tools with least privilege; no arbitrary shell or unrestricted AWS APIs. Do NOT add if it adds unnecessary complexity to the initial release
    - _Requirements: 5.5, 8.3, 9.3, 18.1, 18.4_
  - [ ] 18.7 Security, cost, and cleanup for AgentCore
    - Least-privilege IAM roles; no credentials in source or frontend bundles; verify Region service availability/permissions; document deployed resources, expected cost categories, monitoring, and teardown
    - _Requirements: 18.1, 18.3, 18.5, 17.4, 17.5_
  - [ ] 18.8 Critical acceptance test on the live Runtime
    - Demonstrate: real OpenCV analyzes a frame -> result indicates insufficient evidence -> Strands agent on AgentCore Runtime receives the structured result -> agent selects another frame or region analysis -> a second real OpenCV tool call executes -> the new evidence changes the next decision or causes human review -> the full sequence is inspectable via the application trace and AgentCore Observability. No AI-generated explanation may substitute for actual vision-tool execution; capture a VERIFIED trace, never a fabricated one
    - _Requirements: 5.6, 14.2, 19.5, Competition 1, Competition 2_

- [ ] 19. Checkpoint - Milestone 9 definition of done
  - The existing app remains functional and OpenCV 5 still performs substantive analysis; AgentCore Runtime hosts the agent in the deployed environment; tool permissions and input validation are enforced; the evidence-driven second tool call is reproducible and visible in the app trace and AgentCore Observability; local mock tests still pass with no AWS credentials; deployment/testing/cost/cleanup docs updated. Add AgentCore Memory/Identity/Gateway only with a concrete per-capability requirement. Ensure all tests pass, ask the user if questions arise.


## Notes

- Tasks marked with `*` are optional (all test sub-tasks, the optional Strands Decider, and the live-model/Bedrock-adjacent verification), and can be skipped for a faster MVP. Core implementation sub-tasks are never optional.
- The Mock_Runtime + real OpenCV path (Milestones 1-5, 8) is the reproducible backbone and runs with no AWS credentials. Milestone 6 (AWS IaC) and Milestone 7 (Bedrock/Strands live backends) layer on behind the `DeciderService`, `Store`, and vision-façade abstractions without changing the default path.
- Each task references specific granular requirements for traceability, and each property sub-task explicitly references a design correctness property (1-24) and the requirement clauses it validates.
- Property-based tests use Hypothesis (Python) and fast-check (TypeScript) at a minimum of 100 iterations; Properties 12 and 24 are metamorphic and prove the critical "vision drives action" criterion.
- Checkpoints mark each milestone's definition of done for incremental validation.
- Milestone 9 (Amazon Bedrock AgentCore) layers on behind the same `DeciderService` seam as a LIVE deploy (mode B); it is deferred to a later cycle pending a confirmed Region + credentials/spend go-ahead, and the Mock_Runtime remains the reproducible, credential-free default. AgentCore Memory/Identity/Gateway are added only with a concrete per-capability requirement.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["2.1", "2.2"] },
    { "id": 1, "tasks": ["2.3", "2.4", "2.5"] },
    { "id": 2, "tasks": ["2.6", "4.1", "4.4"] },
    { "id": 3, "tasks": ["4.2", "4.3", "4.5", "4.8"] },
    { "id": 4, "tasks": ["4.6", "4.7", "6.1", "6.3"] },
    { "id": 5, "tasks": ["6.2", "6.4", "6.5"] },
    { "id": 6, "tasks": ["6.6", "6.7", "6.15", "6.17"] },
    { "id": 7, "tasks": ["6.8", "6.9", "6.10", "6.11", "6.12", "6.13", "6.14", "6.16"] },
    { "id": 8, "tasks": ["8.1", "8.2", "8.2a", "8.2b"] },
    { "id": 9, "tasks": ["8.3", "8.4", "8.5"] },
    { "id": 10, "tasks": ["10.1"] },
    { "id": 11, "tasks": ["10.2", "10.3", "10.4", "10.5", "10.6"] },
    { "id": 12, "tasks": ["10.7", "10.8", "12.1", "12.6"] },
    { "id": 13, "tasks": ["12.2", "12.3", "12.7"] },
    { "id": 14, "tasks": ["12.4", "12.5", "12.8"] },
    { "id": 15, "tasks": ["12.9", "12.10", "14.1"] },
    { "id": 16, "tasks": ["14.2", "14.4"] },
    { "id": 17, "tasks": ["14.3", "14.5", "14.6", "14.7"] },
    { "id": 18, "tasks": ["16.1", "16.2", "16.3"] },
    { "id": 19, "tasks": ["16.4", "16.5", "16.6"] },
    { "id": 20, "tasks": ["18.0", "18.1"] },
    { "id": 21, "tasks": ["18.2", "18.3", "18.7"] },
    { "id": 22, "tasks": ["18.4", "18.5", "18.6"] },
    { "id": 23, "tasks": ["18.8"] }
  ]
}
```
