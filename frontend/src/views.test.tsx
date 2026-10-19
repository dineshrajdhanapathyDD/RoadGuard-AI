import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { App } from "./App";

/**
 * Render tests for the five dashboard views (task 10.8, req 16.1-16.4, 14.2).
 * Uses the default offline fixtures client (no backend, no AWS). Verifies each
 * tab renders and, critically, that the agent-trace view surfaces the
 * before/reasoning/tool/after/decision sequence that evidences Competition 1.
 */

async function switchTo(label: string) {
  fireEvent.click(screen.getByRole("tab", { name: label }));
}

it("overview view renders aggregate stats and recent activity", async () => {
  render(<App />);
  await waitFor(() => expect(screen.getByText("Recent activity")).toBeDefined());
  expect(screen.getByText("Inspections")).toBeDefined();
  expect(screen.getByText("sample_road_pothole.mp4")).toBeDefined();
});

it("inspection view shows a bounded progress bar", async () => {
  render(<App />);
  await waitFor(() => expect(screen.getByText("Recent activity")).toBeDefined());
  await switchTo("Inspection");
  const bar = await screen.findByRole("progressbar");
  const now = Number(bar.getAttribute("aria-valuenow"));
  expect(now).toBeGreaterThanOrEqual(0);
  expect(now).toBeLessThanOrEqual(100);
});

it("agent-trace view renders the before/after/decision sequence (critical criterion)", async () => {
  render(<App />);
  await waitFor(() => expect(screen.getByText("Recent activity")).toBeDefined());
  await switchTo("Agent trace");
  // the first real analysis and the evidence-driven second vision call are both shown
  expect(await screen.findByText(/perception . reasoning . tool/i)).toBeDefined();
  expect(screen.getAllByText("analyze_frame").length).toBeGreaterThan(0);
  expect(screen.getByText("analyze_region")).toBeDefined();
  // before/after labels present so the causal chain is visible
  expect(screen.getAllByText(/before \(initial output\)/i).length).toBeGreaterThan(0);
  expect(screen.getAllByText(/after \(subsequent output\)/i).length).toBeGreaterThan(0);
});

it("report view records explicit zero findings when there are none", async () => {
  render(<App />);
  await waitFor(() => expect(screen.getByText("Recent activity")).toBeDefined());
  await switchTo("Report");
  expect(await screen.findByText(/zero findings/i)).toBeDefined();
});

it("metrics view shows operational stats and honours unavailable detection metrics", async () => {
  render(<App />);
  await waitFor(() => expect(screen.getByText("Recent activity")).toBeDefined());
  await switchTo("Metrics");
  // detection metrics are unavailable in the fixtures -> explicit message, not fake zeros
  expect(await screen.findByText(/Detection metrics are unavailable/i)).toBeDefined();
  // operational stat labels are present
  expect(screen.getByText("Completion")).toBeDefined();
});
