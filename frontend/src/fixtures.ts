/**
 * Deterministic offline fixtures for the RoadGuard dashboard (task 10.1).
 *
 * These mirror a real Mock_Runtime inspection over the pothole sample: a first
 * real OpenCV analyze_frame (mid-confidence surface_damage ~0.75) that drives a
 * second analyze_region call, i.e. the critical criterion. They let the UI and
 * its tests run with NO backend and NO AWS (requirement 19.4 / Competition 4).
 * The real API client swaps these for live read-endpoint responses in M6.
 *
 * Clearly labelled as SAMPLE/demonstration data so the UI never presents
 * fixtures as measured results (responsible-use).
 */
import type {
  AnalysisResult,
  InspectionRecord,
  InspectionReport,
  OverviewStats,
  TraceEntry,
} from "./types";

const FRAME = "sample_road_pothole.mp4#frame_00000000";

function analysis(
  method: "classical" | "model",
  conf: number,
  frame = FRAME,
  ts = 0,
): AnalysisResult {
  return {
    status: "success",
    method,
    frame_ref: frame,
    source_timestamp_ms: ts,
    detections: [
      {
        hazard_type: "surface_damage",
        bbox_px: [69, 119, 9, 21],
        confidence: conf,
        measurement: { unit: "pixels", area_px: 189, width_px: 9, height_px: 21, physical: "unavailable" },
      },
    ],
    quality_flags: [],
    failure_reason: null,
    evidence_ref: `evidence/demo-001/${frame.replace(/[^A-Za-z0-9._-]/g, "_")}.png`,
  };
}

const low = analysis("classical", 0.75);
const region = analysis("classical", 0.72);

export const SAMPLE_TRACE: TraceEntry[] = [
  {
    ts: 0,
    perception: FRAME,
    reasoning: "initial perception: the first real OpenCV analysis the first agent decision is reasoned over (5.6)",
    tool: "analyze_frame",
    before_output: "unavailable",
    after_output: low,
    decision: "initial analysis detected 'surface_damage' at confidence 0.75; evaluating whether to corroborate or save",
  },
  {
    ts: 1,
    perception: FRAME,
    reasoning: "Best detection is moderately confident (0.75 < 0.80); corroborating with a higher-fidelity region analysis before saving a finding.",
    tool: "analyze_region",
    before_output: low,
    after_output: region,
    decision: "investigated region at higher fidelity to corroborate the signal",
  },
  {
    ts: 2,
    perception: FRAME,
    reasoning: "region corroboration inconclusive; widening the search to additional frames",
    tool: "select_additional_frames",
    before_output: region,
    after_output: "unavailable",
    decision: "selected additional frames to widen the inspection",
  },
  {
    ts: 3,
    perception: FRAME,
    reasoning: "no confident, corroborated hazard resolved; completing the inspection",
    tool: "complete_inspection",
    before_output: "unavailable",
    after_output: "unavailable",
    decision: "completed the inspection (complete_inspection)",
  },
];

export const SAMPLE_REPORT: InspectionReport = {
  inspection_id: "demo-001",
  status: "completed",
  stop_reason: "complete_inspection",
  findings: [],
  escalations: [],
  finding_count: 0,
  escalation_count: 0,
  zero_findings: true,
  trace_length: SAMPLE_TRACE.length,
  metrics: {
    precision: "unavailable",
    recall: "unavailable",
    f1: "unavailable",
    iou: "unavailable",
    action_accuracy: 1.0,
    completion: 1.0,
    escalation_rate: 0.0,
    latency_ms: 1840,
    failure_rate: 0.0,
    aws_cost: "unavailable",
  },
};

export const SAMPLE_INSPECTION: InspectionRecord = {
  inspection_id: "demo-001",
  status: "completed",
  filename: "sample_road_pothole.mp4",
  duration_ms: 2000,
  resolution: "320x240",
  fps: 15,
  upload_ts: "2026-01-01T00:00:00Z",
  progress: 100,
  stop_reason: "complete_inspection",
};

export const SAMPLE_OVERVIEW: OverviewStats = {
  total_inspections: 1,
  completed: 1,
  failed: 0,
  potential_findings: 0,
  awaiting_review: 0,
  recent: [SAMPLE_INSPECTION],
};

/** True for the fixtures module so the UI can badge SAMPLE data honestly. */
export const IS_SAMPLE_DATA = true;
