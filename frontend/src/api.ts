/**
 * Typed API client for the RoadGuard read endpoints (task 10.1).
 *
 * The dashboard reads inspection metadata, the inspection report, the agent
 * trace, aggregate metrics, and overview stats. Until the backend lands
 * (Milestone 6) the default client serves the deterministic offline fixtures,
 * so the UI runs with no backend and no AWS (19.4 / Competition 4). A live
 * client (`HttpApiClient`) is provided for M6 and targets the same shapes.
 */
import {
  SAMPLE_INSPECTION,
  SAMPLE_OVERVIEW,
  SAMPLE_REPORT,
  SAMPLE_TRACE,
} from "./fixtures";
import type {
  InspectionRecord,
  InspectionReport,
  OverviewStats,
  TraceEntry,
} from "./types";

export interface ApiClient {
  /** Whether this client serves SAMPLE/demonstration data (for honest UI badging). */
  readonly isSample: boolean;
  getOverview(): Promise<OverviewStats>;
  getInspection(id: string): Promise<InspectionRecord>;
  getReport(id: string): Promise<InspectionReport>;
  getTrace(id: string): Promise<TraceEntry[]>;
  getMetrics(): Promise<OverviewStats>;
}

/** Offline client backed by the deterministic fixtures (default). */
export class MockApiClient implements ApiClient {
  readonly isSample = true;
  async getOverview(): Promise<OverviewStats> {
    return SAMPLE_OVERVIEW;
  }
  async getInspection(_id: string): Promise<InspectionRecord> {
    return SAMPLE_INSPECTION;
  }
  async getReport(_id: string): Promise<InspectionReport> {
    return SAMPLE_REPORT;
  }
  async getTrace(_id: string): Promise<TraceEntry[]> {
    return SAMPLE_TRACE;
  }
  async getMetrics(): Promise<OverviewStats> {
    return SAMPLE_OVERVIEW;
  }
}

/** Live client for Milestone 6 (API Gateway read endpoints). */
export class HttpApiClient implements ApiClient {
  readonly isSample = false;
  private readonly fetchImpl: typeof fetch;
  // Default to a window-bound fetch: an unbound `fetch` reference called as
  // `this.fetchImpl(...)` throws "Illegal invocation" in browsers because fetch
  // must run bound to the global (window). Bind it here; still injectable for tests.
  constructor(private readonly baseUrl: string, fetchImpl?: typeof fetch) {
    this.fetchImpl =
      fetchImpl ??
      (typeof window !== "undefined"
        ? window.fetch.bind(window)
        : (globalThis.fetch?.bind(globalThis) as typeof fetch));
  }

  private async get<T>(path: string): Promise<T> {
    const res = await this.fetchImpl(`${this.baseUrl}${path}`, {
      headers: { accept: "application/json" },
    });
    if (!res.ok) {
      throw new Error(`GET ${path} failed: ${res.status} ${res.statusText}`);
    }
    return (await res.json()) as T;
  }

  getOverview() {
    return this.get<OverviewStats>("/overview");
  }
  getInspection(id: string) {
    return this.get<InspectionRecord>(`/inspections/${encodeURIComponent(id)}`);
  }
  getReport(id: string) {
    return this.get<InspectionReport>(`/inspections/${encodeURIComponent(id)}/report`);
  }
  getTrace(id: string) {
    return this.get<TraceEntry[]>(`/inspections/${encodeURIComponent(id)}/trace`);
  }
  getMetrics() {
    return this.get<OverviewStats>("/metrics");
  }
}

/**
 * Resolve the client for this build. Uses the live HTTP client only when a
 * base URL is configured (VITE_API_BASE_URL); otherwise the offline fixtures
 * client, so local dev/tests need no backend.
 */
export function resolveApiClient(): ApiClient {
  const base = import.meta.env?.VITE_API_BASE_URL as string | undefined;
  return base ? new HttpApiClient(base) : new MockApiClient();
}
