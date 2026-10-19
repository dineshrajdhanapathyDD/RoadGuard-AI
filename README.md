# RoadGuard AI

> Agentic road-safety inspection platform for the **OpenCV + AWS AI Competition 2026** (Agentic Vision Award focus).

RoadGuard AI ingests a road video, runs a **real OpenCV 5** computer-vision pipeline, and drives a **bounded reasoning agent** whose next action is determined by genuine vision output. Perception decides action: the agent only escalates to a detailed region analysis when the vision stage actually detects damage above threshold.

The full perception-decision-action loop runs **offline** (deterministic Mock_Runtime, no AWS creds) for development and CI, and runs **live on AWS** in production via Amazon Bedrock AgentCore Runtime + ECS Fargate.

---

## Live demo

| Resource | URL |
| --- | --- |
| Dashboard (Amplify Hosting) | https://main.d1mue3zctzzbg9.amplifyapp.com/ |
| REST API (API Gateway) | https://7p1i7iqoyb.execute-api.us-east-1.amazonaws.com/dev/ |

Quick check:

```
curl https://7p1i7iqoyb.execute-api.us-east-1.amazonaws.com/dev/overview
```

Returns live inspection counts and the latest agent decision traces backed by DynamoDB.

> Note: the live stack bills to a personal AWS account and may be torn down after judging. The offline path below reproduces the full agent loop with zero AWS dependency.

---

## Why this is agentic vision

The agent is not a scripted pipeline. Each step the agent takes is chosen from real OpenCV output:

1. `analyze_frame` runs OpenCV 5 detection on a sampled frame.
2. The decider inspects the detection confidence from that frame.
3. **Only if** damage confidence crosses threshold does the agent call `analyze_region` for a detailed bounded-region pass.
4. Every decision is persisted as a trace step (timestamp, tool, inputs, confidence) for auditability.

A verified live trace: `ts=0 analyze_frame (surface_damage 0.754)` to `ts=1 analyze_region (before_conf=0.754)`. The second action exists **because** the first detection fired. Remove the detection and the agent never escalates. That causal link between vision and action is the core of the submission.

---

## Architecture

```
                         +---------------------------+
   road video  ---->     |  ECS Fargate (OpenCV 5)   |   real cv2 pipeline:
                         |  vision/worker_main.py    |   decode -> sample ->
                         +-------------+-------------+   detect -> annotate
                                       |
                                       v
                         +---------------------------+
                         |  Amazon Bedrock AgentCore |   bounded reasoning loop
                         |  Runtime (roadguardagent) |   7 tools, decider picks
                         +-------------+-------------+   next action from vision
                                       |
                 +---------------------+---------------------+
                 v                     v                     v
          +-------------+      +--------------+      +---------------+
          |  DynamoDB   |      |   S3 (evid.) |      |  Lambda x3 +  |
          | single-tbl  |      |   bucket     |      |  API Gateway  |
          +------+------+      +--------------+      +-------+-------+
                 |                                           |
                 +---------------------+---------------------+
                                       v
                         +---------------------------+
                         |  Amplify Hosting (React)  |  dashboard: inspections,
                         |  frontend/                |  traces, confidence charts
                         +---------------------------+
```

---

## Tech stack

- **Vision:** OpenCV 5, Python, NumPy. Classical detector with optional ONNX model path.
- **Agent:** Bounded reasoning loop, 7 tools, pluggable decider (Mock_Runtime offline / Amazon Bedrock online).
- **Runtime:** Amazon Bedrock AgentCore Runtime (managed agent), ECS Fargate (vision worker).
- **Data/API:** DynamoDB single-table (GSI1/2/3), S3, Lambda, API Gateway.
- **Frontend:** React + Vite + TypeScript + Tailwind + Recharts, hosted on AWS Amplify.
- **IaC:** AWS CDK (TypeScript synth) for the backend stack; CloudFormation for AgentCore.
- **Tests:** pytest + Hypothesis (property tests), Vitest for the frontend.

---

## Repository layout

```
roadguard-ai/
  vision/              OpenCV 5 pipeline (decode, sample, detect, annotate)
    verify_opencv.py     credential-free OpenCV-5 smoke check
    worker_main.py       Fargate entrypoint (real cv2 end-to-end)
    samples/             sample-video generator + fetch script
  agent/               bounded loop, 7 tools, decider/store abstractions
  agentcore/           AgentCore Runtime entrypoint + requirements
  frontend/            React + Vite dashboard (live API client)
  infra/               AWS CDK stack (API, Lambda, DynamoDB, S3, Fargate, VPC)
  tests/               pytest + Hypothesis property tests
  amplify.yml          Amplify Hosting build spec
  pyproject.toml       Python project + exact pinned dependencies
```

Authoritative spec: `.kiro/specs/roadguard-ai/` (`requirements.md`, `design.md`, `tasks.md`).

---

## Quick start (offline, no AWS)

```
# 1. install pinned deps
pip install -r requirements.txt -r requirements-dev.txt

# 2. verify OpenCV 5 is wired correctly
python vision/verify_opencv.py

# 3. run the full test suite (32 Python + property tests)
pytest

# 4. run the agent loop against the deterministic Mock_Runtime
python -m agent.run_demo
```

The frontend:

```
cd frontend
npm install
npm run dev      # uses MockApiClient when VITE_API_BASE_URL is unset
npm test         # Vitest
```

---

## Deploy (live AWS)

```
# backend stack (API, Lambda, DynamoDB, S3, Fargate, VPC)
cd infra
npm install
npx aws-cdk@2.1145.0 deploy -c environment=dev

# AgentCore Runtime (managed agent)
cd ../agentcore_app/roadguardagent
agentcore deploy
```

Point the dashboard at the live API by setting `VITE_API_BASE_URL` in `frontend/.env.production`, then deploy via Amplify using the root `amplify.yml`.

**Teardown (stop billing):**

```
cd infra && npx aws-cdk@2.1145.0 destroy -c environment=dev
cd agentcore_app/roadguardagent && agentcore destroy
```

---

## Judging rubric mapping

| Criterion | Where it shows up |
| --- | --- |
| Agentic Vision | Agent action is causally gated on real OpenCV detection confidence; traces prove the link. |
| Real computer vision | OpenCV 5 pipeline runs on ECS Fargate, verified exit 0 end-to-end on real video. |
| Production AWS | Bedrock AgentCore Runtime, Lambda, API Gateway, DynamoDB, S3, Fargate, Amplify - all live. |
| Reproducibility | Offline Mock_Runtime + 32 tests run with zero AWS credentials. |
| Observability | Every agent decision persisted as an auditable trace step in DynamoDB. |

---

## Responsible use

RoadGuard AI is a road-infrastructure inspection aid. It flags candidate surface damage for human review; it is not an authority on road safety and should not replace qualified inspection. Evidence images are stored in a private S3 bucket. The system processes road-surface imagery only and is not designed for identifying people or vehicles.
