# RoadGuard AgentCore agent (Milestone 9, Option A)

The agent REASONING brain, hosted on Amazon Bedrock AgentCore Runtime. It does
NOT bundle OpenCV (verified: the decision layer imports no cv2/numpy) - the heavy
computer-vision work stays on the Fargate Vision_Worker from Milestone 6. This is
the clean agent/vision separation of Option A.

- `agent_entrypoint.py` - BedrockAgentCoreApp with an `@app.entrypoint` that
  receives a decision request (state + a real AnalysisResult from the
  Vision_Worker) and returns the next AgentDecision, constrained to the seven
  RoadGuard tools. Uses the MockDecider by default and the BedrockDecider when
  RUNTIME!=mock and credentials resolve (same DeciderService seam as offline).
- `requirements.txt` - runtime deps (no OpenCV), so the AgentCore image is small.

## Local dev (no AWS)
    agentcore configure --entrypoint agentcore/agent_entrypoint.py
    agentcore dev        # hot-reload local server; POST /invocations with a decide payload

## Deploy to AgentCore Runtime (live; needs credentials + spend)
    agentcore deploy     # builds a small image, pushes, creates the Runtime
    agentcore status
    agentcore invoke '{"action":"ping"}'
    agentcore obs        # AgentCore Observability: spans/traces for agent decisions

## Teardown
    agentcore destroy
