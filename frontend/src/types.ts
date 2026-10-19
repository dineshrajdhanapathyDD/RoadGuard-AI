/**
 * Shared TypeScript types mirroring the Python `vision.result_schema`,
 * `agent.trace`, `agent.store`, and `agent.tools.completion` shapes
 * (RoadGuard AI, task 10.1). These are the read-model the dashboard renders.
 *
 * Responsible-use: any value the backend could not compute is the explicit
 * string sentinel `"unavailable"` (never a fabricated number), so the UI must
 * treat `number | Unavailable` fields accordingly and show an explicit
 * "unavailable" indicator rather than 0.
 */

export const UNAVAILABLE = "unavailable" as const;
export type Unavailable = typeof UNAVAILABLE;

/** A numeric value that may be explicitly unavailable. */
export type MaybeNumber = number | Unavailable;

export type HazardType = "pothole" | "surface_damage" | "crack" | "debris";
export type Method = "classical" | "model";
export type Status = "success" | "failed";
export type QualityCode = "decode_failure" | "low_confidence" | "poor_input_quality";

export interface Measurement {
  unit: "pixels";
  area_px: number;
  width_px: number;
  height_px: number;
  physical: "unavailable";
}

export interface QualityFlag {
  code: QualityCode;
  detail: string;
}

export interface Detection {
  hazard_type: HazardType;
  bbox_px: [number, number, number, number];
  confidence: number;
  measurement: Measurement;
}

export interface AnalysisResult {
  status: Status;
  method: Method;
  frame_ref: string;
  source_timestamp_ms: number;
  detections: Detection[];
  quality_flags: QualityFlag[];
  failure_reason: string | null;
  evidence_ref: string | null;
}

export interface ComparisonResult {
  status: Status;
  agreement: "agreement" | "disagreement" | "unavailable";
  evidence_refs: string[];
  detail: string;
  failure_reason: string | null;
}

/** One side of a trace step's vision output: a real result or the sentinel. */
export type TraceOutput = AnalysisResult | ComparisonResult | Unavailable;

/** One ordered agent-trace entry (agent.trace.TraceEntry). */
export interface TraceEntry {
  perception: string;
  reasoning: string;
  tool: string;
  before_output: TraceOutput;
  after_output: TraceOutput;
  decision: string;
  ts: number;
}

export interface InspectionFinding {
  hazard_type: string;
  evidence_ref: string;
  confidence: number;
  source_timestamp_ms: number;
  method: string;
}

export interface HumanReviewItem {
  observation_ref: string;
  evidence_ref: string;
  reason: string;
  status: "open" | "resolved";
}

export interface EvaluationMetrics {
  precision: MaybeNumber;
  recall: MaybeNumber;
  f1: MaybeNumber;
  iou: MaybeNumber;
  action_accuracy: MaybeNumber;
  completion: MaybeNumber;
  escalation_rate: MaybeNumber;
  latency_ms: MaybeNumber;
  failure_rate: MaybeNumber;
  aws_cost: MaybeNumber;
}

export type InspectionStatus = "pending" | "processing" | "completed" | "failed";

export interface InspectionRecord {
  inspection_id: string;
  status: InspectionStatus;
  filename: string;
  duration_ms: MaybeNumber;
  resolution: string;
  fps: MaybeNumber;
  upload_ts: string;
  progress: number;
  stop_reason: string;
}

/** The structured report (agent.tools.completion.InspectionReport). */
export interface InspectionReport {
  inspection_id: string;
  status: string;
  stop_reason: string;
  findings: InspectionFinding[];
  escalations: HumanReviewItem[];
  finding_count: number;
  escalation_count: number;
  zero_findings: boolean;
  trace_length: number;
  metrics: EvaluationMetrics | Unavailable;
}

/** Overview aggregate across inspections (dashboard overview view). */
export interface OverviewStats {
  total_inspections: number;
  completed: number;
  failed: number;
  potential_findings: number;
  awaiting_review: number;
  recent: InspectionRecord[];
}

/** Type guards for the "unavailable" sentinel. */
export function isUnavailable(v: unknown): v is Unavailable {
  return v === UNAVAILABLE;
}

export function isAnalysisResult(v: TraceOutput): v is AnalysisResult {
  return typeof v === "object" && v !== null && "detections" in v;
}

export function isComparisonResult(v: TraceOutput): v is ComparisonResult {
  return typeof v === "object" && v !== null && "agreement" in v;
}

/** Format a MaybeNumber for display, honouring the unavailable sentinel. */
export function fmtNumber(v: MaybeNumber, digits = 2): string {
  return isUnavailable(v) ? "unavailable" : v.toFixed(digits);
}

/** Format a [0,1] rate as a percentage, honouring the sentinel. */
export function fmtPct(v: MaybeNumber): string {
  return isUnavailable(v) ? "unavailable" : `${(v * 100).toFixed(1)}%`;
}
