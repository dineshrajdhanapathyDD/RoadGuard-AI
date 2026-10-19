# RoadGuard AI - AWS deployment, cost, security, and cleanup (task 12.10)

Status: **authored and synth-verified locally; NOT deployed.** A live deploy is
gated on an explicit Region + credentials + spend approval.

## Provisioned resources (RoadGuard-<env> stack)
| Resource | Purpose | Notable config |
| --- | --- | --- |
| S3 `EvidenceBucket` | video uploads + annotated-frame evidence | Block Public Access (all), SSE (S3-managed), TLS-only bucket policy, `uploads/` 7-day expiry, DESTROY + autoDeleteObjects |
| DynamoDB `RoadGuard-<env>` | inspections, findings, trace, review, metrics | single table + GSI1/2/3, SSE, PITR, PAY_PER_REQUEST, DESTROY |
| Lambda `UploadHandler` | presigned PUT + upload validation | 256 MB, 29s, least-privilege (S3 put `uploads/*`, DDB write) |
| Lambda `InspectionHandler` | start/status; launches Fargate worker | DDB read/write |
| Lambda `ReadHandler` | overview/report/trace/metrics | DDB read, S3 read |
| API Gateway `roadguard-<env>` | REST API (HTTPS/TLS-only) | access logging, throttling (50 rps / 20 burst), metrics |
| ECS Fargate Vision_Worker | real OpenCV pipeline + agent off the Lambda path | 1 vCPU / 2 GB, CloudWatch logs, least-privilege task role + scoped Bedrock invoke |
| VPC (2 AZ, 1 NAT) | worker networking | - |
| CloudWatch log groups | Lambda + API + worker logs | 1-month retention |

## Resource tags (applied to EVERY resource via `Tags.of(app)`)
`Project=roadguard-ai`, `Environment=<env>` (default `dev`), `ManagedBy=cdk`,
`Component=roadguard-ai-backend`, `CostCenter=roadguard-ai`. Verified in synth:
present on 31 resources. Use these for cost allocation and ownership. Override
the environment with `-c environment=prod`.

## Expected cost categories (no always-on compute except NAT)
- **DynamoDB**: on-demand - pay per request; near-zero idle.
- **S3**: storage + requests; raw uploads auto-expire after 7 days.
- **Lambda / API Gateway**: per-invocation; negligible at demo volume.
- **Fargate**: per-task vCPU/GB-seconds only while processing a video.
- **NAT Gateway**: ~hourly + data processing - the main idle cost; remove or use a
  VPC endpoint strategy if cost-sensitive.
- **Bedrock**: per-token when the live runtime is enabled (mock by default -> $0).
- **CloudWatch**: logs ingestion/storage (1-month retention).

## Security controls (map to Requirement 18)
- Least-privilege IAM via `grant*()` per component (18.1).
- Private S3 + Block Public Access on every bucket (18.2).
- Encryption at rest (S3 SSE, DynamoDB SSE) + in transit (API GW HTTPS, S3
  `aws:SecureTransport` deny) (18.3).
- Presigned-only uploads, scoped to one `uploads/` key (18.4).
- No hardcoded secrets; task/exec roles supply runtime access (18.5).
- CloudWatch logs + API/ECS metrics (18.6).
- Bounded retries + timeouts on DynamoDB calls (`dynamo_store._with_retries`) (18.7).

## Deploy (when approved)
    cd infra
    pip install -r requirements.txt
    npx aws-cdk@2.1145.0 bootstrap aws://<account>/<region>
    CDK_DEFAULT_ACCOUNT=<account> CDK_DEFAULT_REGION=<region> \
      npx aws-cdk@2.1145.0 deploy -c environment=<env>
Pick a Region where all services (incl. Bedrock/AgentCore if used later) are
available; verify before deploying.

## Cleanup / teardown
    npx aws-cdk@2.1145.0 destroy -c environment=<env>
All stateful resources use `removalPolicy=DESTROY` (S3 also `autoDeleteObjects`,
DynamoDB PITR does not block delete), so `destroy` removes everything with no
orphaned buckets/tables. Confirm in the console afterwards and delete any
CloudWatch log groups if retention outlives the stack.
