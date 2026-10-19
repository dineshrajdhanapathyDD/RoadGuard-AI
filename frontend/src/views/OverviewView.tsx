/** Overview view: aggregate statistics across inspections (task 10.2, req 16.1). */
import type { OverviewStats } from "../types";
import { Card, Stat, Badge } from "./ui";

export function OverviewView({ stats }: { stats: OverviewStats }) {
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
        <Stat label="Inspections" value={stats.total_inspections} />
        <Stat label="Completed" value={stats.completed} tone="ok" />
        <Stat label="Failed" value={stats.failed} tone={stats.failed ? "danger" : "default"} />
        <Stat label="Potential findings" value={stats.potential_findings} tone={stats.potential_findings ? "warn" : "default"} />
        <Stat label="Awaiting review" value={stats.awaiting_review} tone={stats.awaiting_review ? "warn" : "default"} />
      </div>
      <Card title="Recent activity">
        {stats.recent.length === 0 ? (
          <p className="text-sm text-slate-500">No inspections yet.</p>
        ) : (
          <ul className="divide-y divide-slate-100">
            {stats.recent.map((r) => (
              <li key={r.inspection_id} className="flex items-center justify-between py-2">
                <div>
                  <div className="font-medium text-slate-800">{r.filename}</div>
                  <div className="text-xs text-slate-500">{r.inspection_id} · {r.upload_ts}</div>
                </div>
                <Badge tone={r.status === "completed" ? "emerald" : r.status === "failed" ? "rose" : "sky"}>{r.status}</Badge>
              </li>
            ))}
          </ul>
        )}
      </Card>
    </div>
  );
}
