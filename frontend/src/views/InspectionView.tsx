/** Video inspection view: upload, validation, progress, annotated frames (task 10.3, req 16.2, 1.5). */
import { useState } from "react";
import type { InspectionRecord } from "../types";
import { fmtNumber } from "../types";
import { Card, Badge } from "./ui";

const ACCEPT = ["video/mp4", "video/quicktime", "video/x-msvideo"];

export function InspectionView({ inspection }: { inspection: InspectionRecord }) {
  const [validation, setValidation] = useState<string | null>(null);

  function onPick(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;
    // Client-side pre-validation mirrors the backend's accepted ranges (req 1.2);
    // the authoritative validation runs server-side on upload (Milestone 6).
    if (!ACCEPT.includes(f.type)) {
      setValidation(`Rejected: unsupported type ${f.type || "unknown"} (accepts mp4/mov/avi).`);
      return;
    }
    setValidation(`Ready: ${f.name} (${(f.size / 1_000_000).toFixed(1)} MB) passed client checks.`);
  }

  const pct = Math.max(0, Math.min(100, inspection.progress));

  return (
    <div className="space-y-4">
      <Card title="Upload a road video">
        <label className="flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-slate-300 bg-slate-50 px-6 py-8 text-center hover:border-sky-400">
          <span className="text-sm font-medium text-slate-700">Choose a video (mp4, mov, avi)</span>
          <span className="mt-1 text-xs text-slate-500">Validated on upload; processed by the OpenCV 5 pipeline.</span>
          <input type="file" accept="video/*" className="hidden" onChange={onPick} />
        </label>
        {validation ? (
          <p className={`mt-3 text-sm ${validation.startsWith("Rejected") ? "text-rose-600" : "text-emerald-600"}`}>{validation}</p>
        ) : null}
      </Card>

      <Card title="Processing">
        <div className="mb-2 flex items-center justify-between text-sm">
          <span className="font-medium text-slate-700">{inspection.filename}</span>
          <Badge tone={inspection.status === "completed" ? "emerald" : inspection.status === "failed" ? "rose" : "sky"}>
            {inspection.status}
          </Badge>
        </div>
        <div className="h-3 w-full overflow-hidden rounded-full bg-slate-200" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
          <div className="h-full rounded-full bg-sky-500 transition-all" style={{ width: `${pct}%` }} />
        </div>
        <div className="mt-1 text-right text-xs text-slate-500">{pct}%</div>
        <dl className="mt-4 grid grid-cols-2 gap-x-6 gap-y-1 text-sm sm:grid-cols-4">
          <div><dt className="text-slate-500">Resolution</dt><dd className="font-medium">{inspection.resolution}</dd></div>
          <div><dt className="text-slate-500">FPS</dt><dd className="font-medium">{fmtNumber(inspection.fps, 0)}</dd></div>
          <div><dt className="text-slate-500">Duration</dt><dd className="font-medium">{fmtNumber(inspection.duration_ms, 0)} ms</dd></div>
          <div><dt className="text-slate-500">Stop reason</dt><dd className="font-medium">{inspection.stop_reason}</dd></div>
        </dl>
      </Card>

      <Card title="Annotated evidence">
        <p className="text-sm text-slate-500">
          Annotated frames (bounding boxes + hazard labels) render here from the evidence store once the backend serves
          evidence URLs (Milestone 6). The trace tab shows which frames were analyzed.
        </p>
      </Card>
    </div>
  );
}
