/** Small shared UI primitives for the RoadGuard dashboard (taste-informed). */
import type { ReactNode } from "react";

export function Card({ title, children, className = "" }: { title?: string; children: ReactNode; className?: string }) {
  return (
    <section className={`rounded-xl border border-slate-200 bg-white shadow-sm ${className}`}>
      {title ? <h3 className="border-b border-slate-100 px-4 py-3 text-sm font-semibold text-slate-700">{title}</h3> : null}
      <div className="p-4">{children}</div>
    </section>
  );
}

export function Stat({ label, value, tone = "default" }: { label: string; value: ReactNode; tone?: "default" | "warn" | "danger" | "ok" }) {
  const toneCls =
    tone === "danger" ? "text-rose-600" : tone === "warn" ? "text-amber-600" : tone === "ok" ? "text-emerald-600" : "text-slate-900";
  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3">
      <div className="text-xs uppercase tracking-wide text-slate-500">{label}</div>
      <div className={`mt-1 text-2xl font-bold tabular-nums ${toneCls}`}>{value}</div>
    </div>
  );
}

export function Badge({ children, tone = "slate" }: { children: ReactNode; tone?: "slate" | "amber" | "rose" | "emerald" | "sky" }) {
  const cls = {
    slate: "bg-slate-100 text-slate-700",
    amber: "bg-amber-100 text-amber-800",
    rose: "bg-rose-100 text-rose-800",
    emerald: "bg-emerald-100 text-emerald-800",
    sky: "bg-sky-100 text-sky-800",
  }[tone];
  return <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${cls}`}>{children}</span>;
}

/** Honest "sample data" banner so demonstration fixtures are never mistaken for measured results. */
export function SampleBanner() {
  return (
    <div className="mb-4 rounded-lg border border-amber-200 bg-amber-50 px-4 py-2 text-sm text-amber-800">
      Showing <strong>sample demonstration data</strong> (offline Mock_Runtime). Live measurements appear once a backend is configured.
    </div>
  );
}
