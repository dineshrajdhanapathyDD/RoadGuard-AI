"""RoadGuard AI CDK app entry (Milestone 6, task 12.1).

Instantiates the single :class:`RoadGuardStack` and applies the standard AWS
resource **tag set** to every resource in the app (the tagging the user asked to
be present whenever this is deployed). Tags are applied at the ``App`` scope via
``Tags.of(app)`` so they propagate to all taggable resources in the stack.

Synth-only by default: ``python app.py`` / ``cdk synth`` require no AWS
credentials. A live ``cdk deploy`` is gated on an explicit Region + spend
go-ahead and is not part of this authoring step.
"""

from __future__ import annotations

import os

import aws_cdk as cdk

from roadguard_stack import RoadGuardStack

app = cdk.App()

# Environment name drives the Environment tag and resource naming. Override with
# `-c environment=prod`. Defaults to "dev" so synth works with no extra context.
environment = app.node.try_get_context("environment") or "dev"

# Resolve the deploy target from standard CDK env vars when present; left
# unresolved (None) for synth so no account/region is required to author.
account = os.environ.get("CDK_DEFAULT_ACCOUNT")
region = os.environ.get("CDK_DEFAULT_REGION")
env = cdk.Environment(account=account, region=region) if account and region else None

RoadGuardStack(
    app,
    f"RoadGuard-{environment}",
    environment_name=environment,
    env=env,
)

# --- Standard AWS resource tag set (applied to EVERY resource) -------------- #
# Cost-allocation + ownership tags so deployed resources are attributable and
# teardown/cost tracking is straightforward. Requested by the user for all AWS
# things. These propagate to all taggable resources via the App scope.
cdk.Tags.of(app).add("Project", "roadguard-ai")
cdk.Tags.of(app).add("Environment", environment)
cdk.Tags.of(app).add("ManagedBy", "cdk")
cdk.Tags.of(app).add("Component", "roadguard-ai-backend")
cdk.Tags.of(app).add("CostCenter", "roadguard-ai")

app.synth()
