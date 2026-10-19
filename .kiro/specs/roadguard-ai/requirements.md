# Requirements Document

## Introduction

RoadGuard AI is an agentic road-safety intelligence platform built for the OpenCV + AWS AI Competition 2026 (Agentic Vision Award focus, submission deadline October 26, 2026). The platform ingests road video, runs a real OpenCV 5 computer-vision pipeline, and drives a bounded reasoning agent that interprets genuine vision output to decide what to analyze next. The agent perceives (runs vision analysis), reasons (interprets real results), and acts (invokes a constrained set of tools) in a loop with explicit iteration limits and stopping conditions.

The defining characteristic of RoadGuard AI is that real computer-vision output materially changes subsequent agent behavior. A fixed vision result followed by an AI-generated narrative does not satisfy this platform's acceptance criteria. Failed or low-quality vision analysis cannot be promoted into a successful inspection finding.

Phase A (this release) covers static road-surface inspection: processing an uploaded video, detecting primary hazards (potholes and road-surface damage, with cracks and debris or obstructions where the active detection method supports them), escalating uncertain observations for human review, producing a structured inspection report, recording the full agent trace, and reporting evaluation metrics. The vision stack is hybrid: a classical CV path and a pretrained ONNX-model path via the OpenCV DNN module, with the agent selecting the method per frame or region.

Live Amazon Bedrock is the default agent runtime. A deterministic mock fallback runtime also exists so the perception-decision-action loop and the automated tests run without AWS credentials, which is required for reproducible competition deliverables. The primary deliverable is a full AWS deployment defined as infrastructure-as-code.

## Glossary

- **RoadGuard**: The overall platform comprising the vision pipeline, the agent, the backend services, and the dashboard.
- **Vision_Pipeline**: The OpenCV 5 subsystem that decodes video, samples frames, runs detection, and annotates frames. It exposes two detection paths: classical CV and pretrained model.
- **Classical_Path**: The detection path using classical computer-vision techniques (edge detection, contour analysis, morphology, texture analysis).
- **Model_Path**: The detection path using the OpenCV DNN module with a pretrained ONNX road-damage model.
- **Agent**: The bounded reasoning component that interprets Vision_Pipeline output and invokes Agent_Tools to conduct an inspection.
- **Agent_Runtime**: The reasoning backend for the Agent. Either the Bedrock_Runtime or the Mock_Runtime.
- **Bedrock_Runtime**: The default Agent_Runtime that uses Amazon Bedrock for agent reasoning.
- **Mock_Runtime**: The deterministic fallback Agent_Runtime that reproduces the perception-decision-action loop without AWS credentials.
- **Agent_Tool**: One of the seven constrained operations the Agent may invoke: analyze_frame, select_additional_frames, analyze_region, compare_visual_evidence, save_inspection_finding, request_human_review, complete_inspection.
- **Agent_Loop**: The bounded perception-decision-action cycle in which the Agent perceives via the Vision_Pipeline, reasons over real output, and acts by invoking an Agent_Tool.
- **Agent_Trace**: The ordered, persisted record of every perception, reasoning step, tool invocation, tool result, and decision produced during an inspection.
- **Inspection**: One end-to-end run of the Agent_Loop over a single uploaded video.
- **Inspection_Finding**: A recorded hazard observation that references the real Vision_Pipeline evidence that supports it.
- **Inspection_Report**: The structured output summarizing an Inspection, including findings, evidence references, and metrics.
- **Hazard**: A road-safety condition detected by the Vision_Pipeline. Primary hazards are potholes and road-surface damage; cracks and debris or obstructions are detected where the active detection path supports them.
- **Evidence**: Annotated frames, source timestamps, detection outputs, and quality indicators that substantiate an Inspection_Finding.
- **Quality_Flag**: An indicator the Vision_Pipeline attaches to an observation that is uncertain or derived from poor-quality input.
- **Human_Review_Queue**: The store of observations the Agent has escalated for human review.
- **Dashboard**: The React frontend that presents overview statistics, the video inspection view, the Agent_Trace view, the Inspection_Report, and evaluation metrics.
- **Evaluation_Metrics**: The measured quality and operational indicators: precision, recall, F1, IoU, agent action accuracy, completion rate, human-review escalation rate, latency, failure rate, and AWS cost.
- **IaC_Template**: The infrastructure-as-code definition (AWS CDK or CloudFormation) that provisions the AWS resources.
- **Vision_Worker**: The containerized Python and OpenCV 5 process running on ECS or Fargate that executes the Vision_Pipeline.
- **Presigned_Upload**: An S3 upload performed by the client using a time-limited presigned URL issued by the backend.

## Requirements

### Requirement 1: Video Upload and Validation

**User Story:** As a road-safety inspector, I want to upload a road video and have it validated before processing, so that only usable input enters the inspection pipeline.

#### Acceptance Criteria

1. WHEN an inspector initiates a video upload, THE RoadGuard SHALL issue a Presigned_Upload URL scoped to a single destination object in a private S3 location.
2. WHEN a video upload completes, THE RoadGuard SHALL validate the video container format, codec, duration, and file size against the configured accepted ranges before starting an Inspection.
3. IF an uploaded video fails validation, THEN THE RoadGuard SHALL reject the video, record the specific validation failure reason, and SHALL NOT start an Inspection.
4. WHEN video validation succeeds, THE RoadGuard SHALL record the source filename, duration, resolution, frame rate, and upload timestamp.
5. WHILE an Inspection is processing, THE RoadGuard SHALL report upload and processing progress as a percentage from 0 to 100.

### Requirement 2: OpenCV 5 Vision Pipeline

**User Story:** As a competition reviewer, I want the platform to run a real OpenCV 5 pipeline that produces genuine analysis output, so that the agentic behavior is grounded in actual computer vision.

#### Acceptance Criteria

1. WHEN an Inspection starts, THE Vision_Pipeline SHALL decode the validated video using OpenCV 5 and sample frames for analysis.
2. WHEN the Vision_Pipeline analyzes a frame, THE Vision_Pipeline SHALL produce detection output containing detected regions, per-detection confidence values, and the source frame timestamp.
3. WHEN the Vision_Pipeline produces a detection, THE Vision_Pipeline SHALL generate an annotated frame marking the detected region and SHALL store the annotated frame as Evidence.
4. WHEN the Vision_Pipeline records a detection, THE Vision_Pipeline SHALL preserve the source frame timestamp that links the detection to its position in the source video.
5. IF the Vision_Pipeline cannot decode a frame or completes analysis with low confidence or poor input quality, THEN THE Vision_Pipeline SHALL attach a Quality_Flag describing the uncertainty or failure.
6. IF a Vision_Pipeline analysis fails, THEN THE RoadGuard SHALL mark the analysis result as failed and SHALL NOT allow the failed result to be recorded as a successful Inspection_Finding.

### Requirement 3: Responsible-Use Vision Safeguards

**User Story:** As a platform owner, I want the vision output to avoid unsupported physical claims, so that the platform reports honestly about what it detected.

#### Acceptance Criteria

1. THE Vision_Pipeline SHALL report hazard measurements in pixel-based units and SHALL label those measurements as pixel-based.
2. IF physical dimensions or severity are requested without a calibration reference, THEN THE Vision_Pipeline SHALL report that physical measurement is unavailable rather than converting pixel area into physical dimensions.
3. WHERE a data value is unavailable, THE RoadGuard SHALL record an explicit unavailable indicator and SHALL NOT fabricate a confidence value, a measurement, or a detection.

### Requirement 4: Hybrid Detection with Agent-Selected Method

**User Story:** As an inspector, I want the platform to choose between classical and model-based vision analysis, so that each frame or region is analyzed with the more suitable method.

#### Acceptance Criteria

1. THE Vision_Pipeline SHALL provide a Classical_Path using edge detection, contour analysis, morphology, and texture analysis.
2. THE Vision_Pipeline SHALL provide a Model_Path using the OpenCV DNN module with a pretrained ONNX road-damage model.
3. WHEN the Agent requests analysis of a frame or region, THE Agent SHALL select the Classical_Path or the Model_Path and SHALL record the selected method in the Agent_Trace.
4. THE RoadGuard SHALL pin the ONNX road-damage model to a specific version and SHALL record the model source and verified license in the repository.
5. IF the Model_Path cannot load the pinned model, THEN THE Vision_Pipeline SHALL report the load failure and THE Agent SHALL fall back to the Classical_Path for that analysis.
6. THE Vision_Pipeline SHALL detect potholes and road-surface damage on both the Classical_Path and the Model_Path, and SHALL detect cracks and debris or obstructions where the active detection path supports those hazard types.

### Requirement 5: Bounded Agent Loop

**User Story:** As a platform owner, I want the agent to operate within explicit limits, so that inspections terminate predictably and the agent cannot take unsafe actions.

#### Acceptance Criteria

1. WHEN an Inspection starts, THE Agent SHALL run the Agent_Loop by perceiving Vision_Pipeline output, reasoning over the real output, and invoking exactly one Agent_Tool per iteration.
2. THE Agent SHALL enforce a configured maximum iteration count for each Inspection and SHALL enforce a configured maximum retry count for each Agent_Tool invocation.
3. WHEN the Agent reaches the maximum iteration count, THE Agent SHALL stop the Agent_Loop and SHALL invoke complete_inspection.
4. THE Agent SHALL stop the Agent_Loop when the Agent invokes complete_inspection, when the maximum iteration count is reached, or when an unrecoverable failure is detected.
5. THE Agent SHALL invoke only the seven defined Agent_Tools and SHALL NOT execute arbitrary shell commands or call AWS APIs outside the defined Agent_Tools.
6. WHEN real Vision_Pipeline output is returned to the Agent, THE Agent SHALL use that output to determine the next Agent_Tool invocation, such that the vision output changes the subsequent action.
7. IF an Agent_Tool invocation fails after reaching the maximum retry count, THEN THE Agent SHALL record the failure in the Agent_Trace and SHALL either escalate via request_human_review or invoke complete_inspection.

### Requirement 6: Agent Tool - analyze_frame

**User Story:** As the agent, I want to analyze a specific frame, so that I obtain real detection output to reason over.

#### Acceptance Criteria

1. WHEN analyze_frame is invoked, THE Agent SHALL accept an input of a frame reference and a selected detection method.
2. WHEN analyze_frame completes, THE Agent SHALL return detected regions, per-detection confidence values, the frame timestamp, any Quality_Flag, and the detection method used.
3. IF analyze_frame fails, THEN THE Agent SHALL return a failure status with a reason and SHALL NOT return fabricated detections.

### Requirement 7: Agent Tool - select_additional_frames

**User Story:** As the agent, I want to select additional frames to analyze, so that I can investigate observations that warrant more evidence.

#### Acceptance Criteria

1. WHEN select_additional_frames is invoked, THE Agent SHALL accept an input describing selection criteria and the number of frames requested.
2. WHEN select_additional_frames completes, THE Agent SHALL return an ordered list of frame references with their source timestamps.
3. IF no frames match the selection criteria, THEN THE Agent SHALL return an empty list and a status indicating no matching frames.

### Requirement 8: Agent Tool - analyze_region

**User Story:** As the agent, I want to analyze a specific region within a frame, so that I can obtain focused detection output for an area of interest.

#### Acceptance Criteria

1. WHEN analyze_region is invoked, THE Agent SHALL accept an input of a frame reference, a region boundary, and a selected detection method.
2. WHEN analyze_region completes, THE Agent SHALL return detection output for the specified region, per-detection confidence values, and any Quality_Flag.
3. IF the region boundary falls outside the frame bounds, THEN THE Agent SHALL return a validation failure and SHALL NOT analyze the region.

### Requirement 9: Agent Tool - compare_visual_evidence

**User Story:** As the agent, I want to compare visual evidence across frames or regions, so that I can corroborate or reconcile observations.

#### Acceptance Criteria

1. WHEN compare_visual_evidence is invoked, THE Agent SHALL accept an input referencing two or more Evidence items.
2. WHEN compare_visual_evidence completes, THE Agent SHALL return a comparison result describing agreement or disagreement across the referenced Evidence.
3. IF fewer than two valid Evidence items are referenced, THEN THE Agent SHALL return a validation failure.

### Requirement 10: Agent Tool - save_inspection_finding

**User Story:** As the agent, I want to save a validated finding, so that confirmed hazards are recorded with supporting evidence.

#### Acceptance Criteria

1. WHEN save_inspection_finding is invoked, THE Agent SHALL accept an input containing the hazard type, the referenced Evidence, the confidence value, and the source timestamp.
2. IF the referenced Evidence is a failed or Quality_Flag-rejected Vision_Pipeline result, THEN THE Agent SHALL reject the save and SHALL NOT record an Inspection_Finding.
3. WHEN save_inspection_finding succeeds, THE Agent SHALL persist the Inspection_Finding with references to the real Evidence that supports it.

### Requirement 11: Agent Tool - request_human_review

**User Story:** As the agent, I want to escalate uncertain observations, so that a human can adjudicate cases the agent cannot confidently resolve.

#### Acceptance Criteria

1. WHEN request_human_review is invoked, THE Agent SHALL accept an input containing the observation reference, the referenced Evidence, and the escalation reason.
2. WHEN request_human_review completes, THE Agent SHALL add the observation to the Human_Review_Queue with its Evidence and escalation reason.
3. WHILE an observation carries a Quality_Flag and the Agent cannot resolve the observation within the retry limit, THE Agent SHALL invoke request_human_review rather than save_inspection_finding.

### Requirement 12: Agent Tool - complete_inspection

**User Story:** As the agent, I want to complete an inspection, so that the loop terminates and the report is finalized.

#### Acceptance Criteria

1. WHEN complete_inspection is invoked, THE Agent SHALL finalize the Inspection and generate the Inspection_Report.
2. WHEN complete_inspection completes, THE Agent SHALL persist the Inspection_Report and the full Agent_Trace.
3. IF complete_inspection is invoked when no Inspection_Finding and no escalated observation exist, THEN THE Agent SHALL generate an Inspection_Report that explicitly records zero findings.

### Requirement 13: Inspection Report

**User Story:** As an inspector, I want a structured inspection report, so that I can review hazards, evidence, and outcomes for a video.

#### Acceptance Criteria

1. WHEN an Inspection completes, THE RoadGuard SHALL generate an Inspection_Report containing each Inspection_Finding, its hazard type, its referenced Evidence, its confidence value, and its source timestamp.
2. THE Inspection_Report SHALL list each escalated observation in the Human_Review_Queue with its escalation reason.
3. THE RoadGuard SHALL persist the Inspection_Report and SHALL make the Inspection_Report retrievable by Inspection identifier.
4. WHERE a reported value is unavailable, THE Inspection_Report SHALL display an explicit unavailable indicator.

### Requirement 14: Agent Trace

**User Story:** As a competition reviewer, I want to see the full agent trace, so that I can verify that real vision output drives the agent's decisions.

#### Acceptance Criteria

1. WHEN the Agent runs an iteration, THE RoadGuard SHALL append to the Agent_Trace the perception input, the reasoning step, the invoked Agent_Tool, the tool result, and the resulting decision in order.
2. THE Agent_Trace SHALL present, for a traced step, the initial OpenCV output, the reasoning over that output, the tool invoked, the subsequent output, and the final decision.
3. THE RoadGuard SHALL persist the Agent_Trace and SHALL make the Agent_Trace retrievable by Inspection identifier.

### Requirement 15: Evaluation Metrics

**User Story:** As a competition reviewer, I want measured evaluation metrics, so that I can assess detection quality and agent operation.

#### Acceptance Criteria

1. WHEN an evaluation dataset with ground truth is processed, THE RoadGuard SHALL compute precision, recall, F1, and IoU for detections.
2. THE RoadGuard SHALL compute agent action accuracy, completion rate, and human-review escalation rate across Inspections.
3. THE RoadGuard SHALL record per-Inspection latency, failure rate, and measured AWS cost.
4. THE Dashboard SHALL display the Evaluation_Metrics.

### Requirement 16: Dashboard

**User Story:** As an inspector, I want a dashboard, so that I can upload videos, follow inspections, and review results.

#### Acceptance Criteria

1. THE Dashboard SHALL present an overview view displaying aggregate statistics across Inspections.
2. THE Dashboard SHALL present a video inspection view supporting upload, validation feedback, processing progress, and display of annotated frames.
3. THE Dashboard SHALL present an Agent_Trace view showing the ordered perception, reasoning, tool invocation, subsequent output, and final decision for an Inspection.
4. THE Dashboard SHALL present the Inspection_Report view and the Evaluation_Metrics view.

### Requirement 17: AWS Deployment and Infrastructure-as-Code

**User Story:** As a platform owner, I want the AWS environment defined as infrastructure-as-code, so that the deployment is reproducible and reviewable.

#### Acceptance Criteria

1. THE RoadGuard SHALL define its AWS infrastructure as an IaC_Template using AWS CDK or CloudFormation.
2. THE IaC_Template SHALL provision S3, API Gateway, Lambda, DynamoDB, CloudWatch, Amazon Bedrock access, and the ECS or Fargate Vision_Worker.
3. THE RoadGuard SHALL run the Vision_Pipeline in the containerized Vision_Worker on ECS or Fargate.
4. THE RoadGuard SHALL provide deployment, AWS cost, security, and resource-cleanup documentation covering the provisioned resources.
5. WHEN the cleanup procedure is followed, THE IaC_Template SHALL support removal of the provisioned resources.

### Requirement 18: Security Controls

**User Story:** As a platform owner, I want enforced security controls, so that data and credentials are protected.

#### Acceptance Criteria

1. THE RoadGuard SHALL grant each component least-privilege IAM permissions scoped to the resources that component uses.
2. THE RoadGuard SHALL store all S3 buckets as private with public access blocked.
3. THE RoadGuard SHALL encrypt stored data at rest and SHALL transmit data over encrypted connections in transit.
4. THE RoadGuard SHALL accept uploads only through validated Presigned_Upload URLs.
5. THE RoadGuard SHALL retrieve credentials and secrets from a managed secret source at runtime and SHALL NOT store hardcoded credentials or secrets in the repository.
6. THE RoadGuard SHALL emit logs and metrics to CloudWatch for Inspections and Agent_Tool invocations.
7. IF an AWS operation fails, THEN THE RoadGuard SHALL apply bounded retries with timeouts and SHALL record the failure.

### Requirement 19: Local Mock Fallback Runtime

**User Story:** As a developer and competition reviewer, I want a deterministic mock runtime, so that the agent loop and tests run reproducibly without AWS credentials.

#### Acceptance Criteria

1. THE RoadGuard SHALL use the Bedrock_Runtime as the default Agent_Runtime.
2. WHERE AWS credentials are unavailable or the Mock_Runtime is selected, THE RoadGuard SHALL run the Agent_Loop using the Mock_Runtime.
3. WHEN the Mock_Runtime runs with the same input, THE Mock_Runtime SHALL produce deterministic agent decisions.
4. THE RoadGuard SHALL execute its automated tests of the Agent_Loop using the Mock_Runtime without requiring AWS credentials.
5. WHILE the Mock_Runtime is active, THE RoadGuard SHALL drive the Agent_Loop from real Vision_Pipeline output rather than from fixed detection results.

## Competition Requirements

**User Story:** As a competition entrant, I want the submission to meet the OpenCV + AWS AI Competition 2026 Agentic Vision criteria, so that RoadGuard AI qualifies and demonstrates genuine agentic vision.

#### Acceptance Criteria

1. THE RoadGuard SHALL demonstrate an Inspection in which real OpenCV 5 output changes a subsequent Agent_Tool invocation or action.
2. IF a Vision_Pipeline analysis is fixed and followed only by an AI-generated explanation, THEN THE RoadGuard SHALL treat that sequence as not satisfying the agentic demonstration criterion.
3. THE RoadGuard SHALL prevent a failed Vision_Pipeline analysis from being recorded as a successful Inspection_Finding.
4. THE RoadGuard SHALL provide reproducible competition deliverables that run the Agent_Loop and automated tests using the Mock_Runtime without AWS credentials.
5. THE RoadGuard SHALL target submission readiness for the competition deadline of October 26, 2026.

## Out of Scope

The following are explicitly out of scope for this initial release (Phase A):

1. THE RoadGuard SHALL NOT include rash-driving or unsafe-driving analysis (deferred to Phase B).
2. THE RoadGuard SHALL NOT include AccessLens functionality.
3. THE RoadGuard SHALL NOT include live surveillance or continuous monitoring.
4. THE RoadGuard SHALL NOT include facial recognition or number-plate identification.
5. THE RoadGuard SHALL NOT include OpenCV COOL functionality (deferred to Phase C).
