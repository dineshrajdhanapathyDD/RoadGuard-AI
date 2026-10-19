/** Inspection report view (task 10.5, req 16.4, 13.1, 13.2, 13.4, 12.3). */
import type { InspectionReport } from "../types";
import { isUnavailable } from "../types";
import { Card, Badge, Stat } from "./ui";

export function ReportView({ report }: { report: InspectionReport }) {
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat label="Findings" value={report.finding_count} tone={report.finding_count ? "warn" : "default"} />
        <Stat label="Escalations" value={report.escalation_count} tone={report.escalation_count ? "warn" : "default"} />
        <Stat label="Trace steps" value={report.trace_length} />
        <Stat label="Status" value={report.status} tone={report.status === "completed" ? "ok" : "default"} />
      </div>

      <Card title={`Report · ${report.inspection_id}`}>
        <div className="mb-3 text-sm text-slate-600">Stop reason: <strong>{report.stop_reason}</strong></div>

        {report.zero_findings ? (
          <div className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-600">
            This inspection explicitly recorded <strong>zero findings</strong> and zero escalations.
          </div>
        ) : (
          <>
            {report.findings.length > 0 && (
              <div className="mb-4">
                <h4 className="mb-2 text-sm font-semibold text-slate-700">Findings</h4>
                <ul className="divide-y divide-slate-100">
                  {report.findings.map((f, i) => (
                    <li key={i} className="flex items-center justify-between py-2 text-sm">
                      <div>
                        <Badge tone="amber">{f.hazard_type}</Badge>{" "}
                        <span className="text-slate-500">@ {f.source_timestamp_ms} ms</span>
                        <div className="truncate text-xs text-slate-400">{f.evidence_ref}</div>
                      </div>
                      <div className="tabular-nums text-slate-700">conf {f.confidence.toFixed(2)} · {f.method}</div>
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {report.escalations.length > 0 && (
              <div>
                <h4 className="mb-2 text-sm font-semibold text-slate-700">Awaiting human review</h4>
                <ul className="divide-y divide-slate-100">
                  {report.escalations.map((e, i) => (
                    <li key={i} className="py-2 text-sm">
                      <span className="font-medium text-slate-800">{e.observation_ref}</span>{" "}
                      <Badge tone={e.status === "open" ? "amber" : "emerald"}>{e.status}</Badge>
                      <div className="text-xs text-slate-500">reason: {e.reason}</div>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </>
        )}

        <p className="mt-4 text-xs text-slate-400">
          Pixel-based measurements only; physical dimensions are reported as <em>unavailable</em> without calibration.
          {isUnavailable(report.metrics) ? " Metrics not yet computed." : ""}
        </p>
      </Card>
    </div>
  );
}
