/**
 * Agent-trace view (task 10.4, req 16.3, 14.2).
 *
 * Renders the ordered before-output / reasoning / tool / after-output /
 * decision sequence that evidences the critical criterion: a real OpenCV
 * output changing a subsequent tool call. Each step shows the real vision
 * output on each side (or the explicit "unavailable" sentinel), so a reviewer
 * can see the causal chain rather than an AI-generated narrative.
 */
import type { TraceEntry, TraceOutput } from "../types";
import { isAnalysisResult, isComparisonResult, isUnavailable } from "../types";
import { Card, Badge } from "./ui";

function OutputSummary({ label, output }: { label: string; output: TraceOutput }) {
  let body: React.ReactNode;
  if (isUnavailable(output)) {
    body = <span className="italic text-slate-400">unavailable</span>;
  } else if (isAnalysisResult(output)) {
    const best = output.detections.reduce<number | null>((m, d) => (m === null || d.confidence > m ? d.confidence : m), null);
    body = (
      <span className="tabular-nums">
        <Badge tone={output.status === "failed" ? "rose" : "emerald"}>{output.status}</Badge>{" "}
        <span className="text-slate-500">method</span> {output.method} ·{" "}
        <span className="text-slate-500">dets</span> {output.detections.length}
        {best !== null ? <> · <span className="text-slate-500">best</span> {best.toFixed(2)}</> : null}
      </span>
    );
  } else if (isComparisonResult(output)) {
    body = <span>comparison: <strong>{output.agreement}</strong></span>;
  } else {
    body = <span className="italic text-slate-400">—</span>;
  }
  return (
    <div className="min-w-0">
      <div className="text-xs uppercase tracking-wide text-slate-400">{label}</div>
      <div className="truncate text-sm text-slate-700">{body}</div>
    </div>
  );
}

export function AgentTraceView({ trace }: { trace: TraceEntry[] }) {
  if (trace.length === 0) {
    return <Card title="Agent trace"><p className="text-sm text-slate-500">No trace recorded.</p></Card>;
  }
  return (
    <Card title="Agent trace — perception → reasoning → tool → subsequent output → decision">
      <ol className="space-y-3">
        {trace.map((e) => (
          <li key={e.ts} className="rounded-lg border border-slate-200 p-3">
            <div className="mb-2 flex items-center gap-2">
              <span className="inline-flex h-6 w-6 items-center justify-center rounded-full bg-slate-800 text-xs font-bold text-white tabular-nums">{e.ts}</span>
              <Badge tone="sky">{e.tool}</Badge>
              <span className="truncate text-xs text-slate-500">{e.perception}</span>
            </div>
            <p className="mb-2 text-sm text-slate-700"><span className="font-medium text-slate-500">reasoning: </span>{e.reasoning}</p>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              <OutputSummary label="before (initial output)" output={e.before_output} />
              <OutputSummary label="after (subsequent output)" output={e.after_output} />
            </div>
            <p className="mt-2 text-sm"><span className="font-medium text-slate-500">decision: </span><span className="text-slate-900">{e.decision}</span></p>
          </li>
        ))}
      </ol>
    </Card>
  );
}
