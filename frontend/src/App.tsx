/**
 * RoadGuard AI dashboard root (Milestone 5).
 *
 * Hosts the five views - overview, inspection, agent trace, report, metrics -
 * behind simple tab navigation, loading data from the resolved API client
 * (offline fixtures by default, so the dashboard runs with no backend / no AWS).
 */
import { useEffect, useMemo, useState } from "react";
import { resolveApiClient } from "./api";
import type { InspectionRecord, InspectionReport, OverviewStats, TraceEntry } from "./types";
import { OverviewView } from "./views/OverviewView";
import { InspectionView } from "./views/InspectionView";
import { AgentTraceView } from "./views/AgentTraceView";
import { ReportView } from "./views/ReportView";
import { MetricsView } from "./views/MetricsView";
import { SampleBanner } from "./views/ui";

type Tab = "overview" | "inspection" | "trace" | "report" | "metrics";
const TABS: { id: Tab; label: string }[] = [
  { id: "overview", label: "Overview" },
  { id: "inspection", label: "Inspection" },
  { id: "trace", label: "Agent trace" },
  { id: "report", label: "Report" },
  { id: "metrics", label: "Metrics" },
];

interface DashboardData {
  overview: OverviewStats;
  inspection: InspectionRecord;
  report: InspectionReport;
  trace: TraceEntry[];
}

export function App() {
  const client = useMemo(() => resolveApiClient(), []);
  const [tab, setTab] = useState<Tab>("overview");
  const [data, setData] = useState<DashboardData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const overview = await client.getOverview();
        const id = overview.recent[0]?.inspection_id ?? "demo-001";
        const [inspection, report, trace] = await Promise.all([
          client.getInspection(id),
          client.getReport(id),
          client.getTrace(id),
        ]);
        if (active) setData({ overview, inspection, report, trace });
      } catch (e) {
        if (active) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      active = false;
    };
  }, [client]);

  return (
    <div className="min-h-screen bg-slate-100 text-slate-900">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-5xl items-center justify-between px-4 py-4">
          <div>
            <h1 className="text-xl font-bold">RoadGuard AI</h1>
            <p className="text-xs text-slate-500">Agentic road-safety inspection · OpenCV 5 + bounded agent</p>
          </div>
        </div>
        <nav className="mx-auto flex max-w-5xl gap-1 px-4" role="tablist">
          {TABS.map((t) => (
            <button
              key={t.id}
              role="tab"
              aria-selected={tab === t.id}
              onClick={() => setTab(t.id)}
              className={`rounded-t-lg px-3 py-2 text-sm font-medium ${tab === t.id ? "bg-slate-100 text-sky-700" : "text-slate-500 hover:text-slate-700"}`}
            >
              {t.label}
            </button>
          ))}
        </nav>
      </header>

      <main className="mx-auto max-w-5xl px-4 py-6">
        {client.isSample ? <SampleBanner /> : null}
        {error ? (
          <div className="rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">Failed to load: {error}</div>
        ) : !data ? (
          <p className="text-sm text-slate-500">Loading…</p>
        ) : (
          <>
            {tab === "overview" && <OverviewView stats={data.overview} />}
            {tab === "inspection" && <InspectionView inspection={data.inspection} />}
            {tab === "trace" && <AgentTraceView trace={data.trace} />}
            {tab === "report" && <ReportView report={data.report} />}
            {tab === "metrics" && <MetricsView metrics={data.report.metrics} />}
          </>
        )}
      </main>
    </div>
  );
}
