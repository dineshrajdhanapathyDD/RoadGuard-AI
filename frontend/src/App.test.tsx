import { render, screen } from "@testing-library/react";
import { App } from "./App";

// Smoke test so the pinned Vitest + Testing Library toolchain is wired and runs.
it("renders the app title", () => {
  render(<App />);
  expect(screen.getByText("RoadGuard AI")).toBeDefined();
});
