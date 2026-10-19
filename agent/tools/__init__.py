"""RoadGuard agent tools package (the seven constrained Agent_Tools).

Each tool is a thin, well-typed boundary over the :class:`vision.pipeline.VisionPipeline`
façade (and, for the persistence tools, the :class:`agent.store.Store`). A tool:

* declares a clear **input schema** (validated from the free-form
  :attr:`agent.state.AgentDecision.arguments` mapping the decider produces),
* returns a structured **output schema** from ``vision.result_schema`` (or the
  persistence records), and
* has a **defined failure behavior** -- bad input is returned as an honest
  failed result with a reason, never a fabricated success and never an
  uncaught exception (3.3, 6.3).

This keeps the tool boundary clean so the bounded agent loop (task 6.8) can
dispatch an :class:`~agent.state.AgentDecision` into exactly one tool per
iteration (5.1) by name (5.5).

The modules are split by task so parallel tool tasks do not collide:

* ``analysis`` -- :func:`analyze_frame`, :func:`analyze_region` (task 6.4).
* ``selection`` (task 6.5), ``recording`` (task 6.6), ``completion`` (task 6.7)
  land their own tools and extend this package's exports.

This module performs no OpenCV, network, or AWS work on import and is importable
with no AWS credentials (Requirement 19.4, Competition 4).
"""

from __future__ import annotations

from agent.tools.analysis import (
    AnalyzeFrameInput,
    AnalyzeRegionInput,
    ToolInputError,
    analyze_frame,
    analyze_region,
)

__all__ = [
    "ToolInputError",
    "AnalyzeFrameInput",
    "AnalyzeRegionInput",
    "analyze_frame",
    "analyze_region",
]
