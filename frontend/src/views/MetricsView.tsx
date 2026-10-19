/** Evaluation metrics view with Recharts (task 10.6, req 16.4, 15.4). */
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";
import type { EvaluationMetrics, MaybeNumber } from "../types";
import { fmtNumber, fmtPct, isUnavailable } from "../types";
import { Card, Stat } from "./ui";

function num(v: MaybeNumber): number | null {
  return isUnavailable(v) ? null : v;
}

export function MetricsView({ metrics }: { metrics: EvaluationMetrics | "unavailable" }) {
  if (isUnavailable(metrics)) {
    return <Card title="Evaluation metrics"><p className="text-sm text-slate-500">No metrics computed yet.</p></Card>;
  }

  // Detection-quality metrics (only plot the ones that are actually measured;
  // unavailable values are omitted from the chart rather than shown as 0).
  const detection = [
    { name: "precision", value: num(metrics.precision) },
    { name: "recall", value: num(metrics.recall) },
    { name: "F1", value: num(metrics.f1) },
    { name: "IoU", value: num(metrics.iou) },
  ].filter((d): d is { name: string; value: number } => d.value !== null);

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        <Stat label="Action acc." value={fmtPct(metrics.action_accuracy)} />
        <Stat label="Completion" value={fmtPct(metrics.completion)} />
        <Stat label="Escalation" value={fmtPct(metrics.escalation_rate)} />
        <Stat label="Failure" value={fmtPct(metrics.failure_rate)} tone="default" />
        <Stat label="Latency" value={isUnavailable(metrics.latency_ms) ? "unavailable" : `${metrics.latency_ms} ms`} />
        <Stat label="AWS cost" value={isUnavailable(metrics.aws_cost) ? "unavailable" : `$${fmtNumber(metrics.aws_cost, 4)}`} />
      </div>

      <Card title="Detection quality (measured)">
        {detection.length === 0 ? (
          <p className="text-sm text-slate-500">Detection metrics are unavailable (no labeled ground truth evaluated yet).</p>
        ) : (
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={detection} margin={{ top: 8, right: 8, bottom: 8, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                <YAxis domain={[0, 1]} tick={{ fontSize: 12 }} />
                <Tooltip />
                <Bar dataKey="value" fill="#0ea5e9" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
        <p className="mt-2 text-xs text-slate-400">Detection quality and agent-decision quality are measured and reported separately.</p>
      </Card>
    </div>
  );
}
