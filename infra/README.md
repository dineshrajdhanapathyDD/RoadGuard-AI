# RoadGuard AI - AWS infrastructure (CDK, Python)

Infrastructure-as-code for Milestone 6. Authored and synth-verified locally; a
live `cdk deploy` is gated on an explicit Region + spend go-ahead (not run yet).

## Layout
- `app.py` - CDK app entry; applies the standard AWS resource tag set to everything.
- `roadguard_stack.py` - the single stack: S3 (private, BPA, SSE, destroyable),
  DynamoDB single table + GSI1/2/3 (SSE, PITR), API Gateway (TLS-only, request
  validation), Lambda read/upload/inspection handlers, CloudWatch, Bedrock access
  (least privilege), and the Fargate Vision_Worker.
- `handlers/` - Lambda handler code (Python).

## Resource tags (applied to every resource)
Project=roadguard-ai, Environment=<env context, default "dev">, ManagedBy=cdk,
Component=roadguard-ai-backend, CostCenter=roadguard-ai. Override Environment with
`-c environment=prod`.

## Verify (no deploy, no AWS creds)
    pip install -r requirements.txt
    npx aws-cdk@2.1145.0 synth --strict

## Deploy (requires Region + credentials + explicit spend approval - NOT run yet)
    npx aws-cdk@2.1145.0 bootstrap aws://<account>/<region>
    npx aws-cdk@2.1145.0 deploy -c environment=<env>

## Teardown
    npx aws-cdk@2.1145.0 destroy
S3 buckets use removalPolicy=DESTROY + autoDeleteObjects so they are removed cleanly.
