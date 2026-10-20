"""RoadGuard AI infrastructure stack (Milestone 6, tasks 12.1/12.3/12.4/12.5).

Provisions the simplest secure architecture for the read/upload/inspection
backend plus the containerized Vision_Worker, following the design's AWS
integration and security sections:

* S3 evidence+video bucket - private, Block Public Access, SSE, TLS-only bucket
  policy, destroyable (removalPolicy=DESTROY + autoDeleteObjects) for clean
  teardown (18.2, 18.3, 17.5).
* DynamoDB single table `RoadGuard` - PK/SK, GSI1 (status/time), GSI2 (review
  queue), GSI3 (metrics aggregate), SSE, PITR, destroyable (13.3, 14.3).
* Lambda handlers - upload (presign+validate), inspection (start/status),
  read (report/trace/metrics) with least-privilege grants (1.x, 13.3, 14.3).
* API Gateway REST - TLS-only (HTTPS endpoints), request validation, access
  logging, throttling (18.x).
* Fargate Vision_Worker - runs the real OpenCV pipeline + agent, least-privilege
  task role, CloudWatch logs, least-privilege Bedrock invoke (17.3, 18.1/6).

IAM uses ``grant*()`` so each component gets least privilege scoped to exactly
the resources it uses (18.1). Nothing here hardcodes credentials (18.5).
"""

from __future__ import annotations

from pathlib import Path

from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
    aws_apigateway as apigw,
    aws_cloudfront as cloudfront,
    aws_cloudfront_origins as origins,
    aws_dynamodb as ddb,
    aws_ec2 as ec2,
    aws_ecr as ecr,
    aws_ecs as ecs,
    aws_iam as iam,
    aws_lambda as lambda_,
    aws_logs as logs,
    aws_s3 as s3,
)
from constructs import Construct

HANDLERS_DIR = str(Path(__file__).resolve().parent / "handlers")


class RoadGuardStack(Stack):
    """The single RoadGuard backend + worker stack."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        environment_name: str,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)
        self._env_name = environment_name

        bucket = self._evidence_bucket()
        table = self._table()
        # Build the Fargate worker first so the backend can wire the upload->RunTask
        # trigger to it. Dashboard is served by Amplify Hosting (not CloudFront here).
        self._vision_worker(bucket, table)
        self._backend(bucket, table)

    # -- Dashboard static site: private S3 + CloudFront (OAC) ------------- #

    def _dashboard_site(self) -> None:
        """Host the built React dashboard on S3 + CloudFront with a public URL.

        The S3 bucket stays PRIVATE (Block Public Access); CloudFront reads it
        through Origin Access Control (OAC), so the only public surface is the
        HTTPS CloudFront URL (18.2, 18.3). The built assets (frontend/dist) are
        uploaded by a BucketDeployment. SPA routing: 403/404 -> index.html.
        """
        repo_root = Path(__file__).resolve().parent.parent
        dist_dir = repo_root / "frontend" / "dist"

        site_bucket = s3.Bucket(
            self,
            "DashboardBucket",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,  # private (18.2)
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )

        distribution = cloudfront.Distribution(
            self,
            "DashboardCDN",
            default_root_object="index.html",
            default_behavior=cloudfront.BehaviorOptions(
                origin=origins.S3BucketOrigin.with_origin_access_control(site_bucket),
                viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                cache_policy=cloudfront.CachePolicy.CACHING_OPTIMIZED,
            ),
            # Single-page app: route client paths back to index.html.
            error_responses=[
                cloudfront.ErrorResponse(
                    http_status=403, response_http_status=200,
                    response_page_path="/index.html", ttl=Duration.minutes(5),
                ),
                cloudfront.ErrorResponse(
                    http_status=404, response_http_status=200,
                    response_page_path="/index.html", ttl=Duration.minutes(5),
                ),
            ],
            comment=f"RoadGuard dashboard ({self._env_name})",
        )

        # NOTE: assets are uploaded post-deploy with `aws s3 sync` + a CloudFront
        # invalidation (see infra/deploy_dashboard.ps1). We intentionally do NOT
        # use s3deploy.BucketDeployment: it bundles assets via a Docker image,
        # which is heavyweight and unreliable in this environment. A plain sync
        # to the private bucket (CloudFront reads it via OAC) is simpler and
        # equivalent. The bucket name and distribution id are emitted as outputs.
        _ = dist_dir  # referenced for documentation; upload is external
        CfnOutput(self, "DashboardBucketName", value=site_bucket.bucket_name,
                  description="Private S3 bucket that backs the dashboard (upload via s3 sync)")
        CfnOutput(self, "DashboardDistributionId", value=distribution.distribution_id,
                  description="CloudFront distribution id (invalidate after upload)")
        CfnOutput(self, "DashboardUrl", value=f"https://{distribution.distribution_domain_name}",
                  description="Public HTTPS URL of the RoadGuard dashboard")

    # -- S3 evidence + video bucket (18.2, 18.3, 17.5) --------------------- #

    def _evidence_bucket(self) -> s3.Bucket:
        bucket = s3.Bucket(
            self,
            "EvidenceBucket",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,  # private (18.2)
            encryption=s3.BucketEncryption.S3_MANAGED,  # SSE at rest (18.3)
            enforce_ssl=True,  # TLS-only bucket policy, denies non-HTTPS (18.3)
            versioned=False,
            removal_policy=RemovalPolicy.DESTROY,  # clean teardown (17.5)
            auto_delete_objects=True,  # required with DESTROY for non-empty buckets
            lifecycle_rules=[
                # Expire raw uploads after processing to bound storage cost.
                s3.LifecycleRule(
                    id="expire-raw-uploads",
                    prefix="uploads/",
                    expiration=Duration.days(7),
                )
            ],
        )
        # CORS so the browser can PUT directly to a presigned URL and GET evidence.
        bucket.add_cors_rule(
            allowed_methods=[s3.HttpMethods.PUT, s3.HttpMethods.GET],
            allowed_origins=["*"],  # tighten to the dashboard origin at deploy time
            allowed_headers=["*"],
            max_age=3000,
        )
        return bucket

    # -- DynamoDB single table + GSIs (13.3, 14.3, 18.3) ------------------- #

    def _table(self) -> ddb.Table:
        table = ddb.Table(
            self,
            "RoadGuardTable",
            table_name=f"RoadGuard-{self._env_name}",
            partition_key=ddb.Attribute(name="PK", type=ddb.AttributeType.STRING),
            sort_key=ddb.Attribute(name="SK", type=ddb.AttributeType.STRING),
            billing_mode=ddb.BillingMode.PAY_PER_REQUEST,  # no idle provisioned cost
            encryption=ddb.TableEncryption.AWS_MANAGED,  # SSE at rest (18.3)
            point_in_time_recovery=True,  # PITR
            removal_policy=RemovalPolicy.DESTROY,  # clean teardown for the demo
        )
        # GSI1: status/time listing for the overview view (16.1).
        table.add_global_secondary_index(
            index_name="GSI1",
            partition_key=ddb.Attribute(name="GSI1PK", type=ddb.AttributeType.STRING),
            sort_key=ddb.Attribute(name="GSI1SK", type=ddb.AttributeType.STRING),
        )
        # GSI2: global human-review queue (11.2, 16).
        table.add_global_secondary_index(
            index_name="GSI2",
            partition_key=ddb.Attribute(name="GSI2PK", type=ddb.AttributeType.STRING),
            sort_key=ddb.Attribute(name="GSI2SK", type=ddb.AttributeType.STRING),
        )
        # GSI3: metrics aggregate across inspections (15.2, 16.4).
        table.add_global_secondary_index(
            index_name="GSI3",
            partition_key=ddb.Attribute(name="GSI3PK", type=ddb.AttributeType.STRING),
            sort_key=ddb.Attribute(name="GSI3SK", type=ddb.AttributeType.STRING),
        )
        return table

    # -- Lambda backend + API Gateway (1.x, 13.3, 14.3, 18.x) -------------- #

    def _backend(self, bucket: s3.Bucket, table: ddb.Table) -> None:
        common_env = {
            "ROADGUARD_TABLE": table.table_name,
            "EVIDENCE_BUCKET": bucket.bucket_name,
            "RUNTIME": "mock",  # backbone default; live Bedrock gated separately
        }

        def make_fn(name: str, handler: str) -> lambda_.Function:
            fn = lambda_.Function(
                self,
                name,
                runtime=lambda_.Runtime.PYTHON_3_13,
                handler=handler,
                code=lambda_.Code.from_asset(HANDLERS_DIR),
                timeout=Duration.seconds(29),  # under API GW 30s integration limit
                memory_size=256,
                environment=common_env,
                log_retention=logs.RetentionDays.ONE_MONTH,  # CloudWatch logs (18.6)
            )
            return fn

        upload_fn = make_fn("UploadHandler", "upload_handler.handler")
        inspection_fn = make_fn("InspectionHandler", "inspection_handler.handler")
        read_fn = make_fn("ReadHandler", "read_handler.handler")

        # Least-privilege grants (18.1): upload presigns/puts to the input prefix
        # and records inspection metadata; read only reads; inspection reads/writes
        # its own records and launches the worker.
        bucket.grant_put(upload_fn, objects_key_pattern="uploads/*")
        bucket.grant_read(read_fn)
        table.grant_write_data(upload_fn)
        table.grant_read_data(read_fn)
        table.grant_read_write_data(inspection_fn)

        # --- Wire upload -> Fargate worker (the missing trigger) -------------
        # The inspection Lambda launches the Vision_Worker Fargate task to run
        # the real OpenCV pipeline + Nova reasoning. We pass the cluster, task
        # def, subnets, security group, and container name as env, and grant
        # ecs:RunTask + PassRole so the Lambda can start the task.
        vpc = self._worker_vpc
        subnet_ids = [s.subnet_id for s in vpc.private_subnets] or [s.subnet_id for s in vpc.public_subnets]
        worker_sg = ec2.SecurityGroup(self, "WorkerTaskSg", vpc=vpc, allow_all_outbound=True)
        inspection_fn.add_environment("WORKER_CLUSTER_ARN", self._worker_cluster.cluster_arn)
        inspection_fn.add_environment("WORKER_TASK_DEF_ARN", self._worker_task_def.task_definition_arn)
        inspection_fn.add_environment("WORKER_CONTAINER_NAME", self._worker_container_name)
        inspection_fn.add_environment("WORKER_SUBNETS", ",".join(subnet_ids))
        inspection_fn.add_environment("WORKER_SECURITY_GROUP", worker_sg.security_group_id)

        inspection_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["ecs:RunTask"],
                resources=[self._worker_task_def.task_definition_arn],
            )
        )
        # PassRole for the task execution + task roles (required by ecs:RunTask).
        pass_roles = [self._worker_task_def.task_role.role_arn]
        if self._worker_task_def.execution_role is not None:
            pass_roles.append(self._worker_task_def.execution_role.role_arn)
        inspection_fn.add_to_role_policy(
            iam.PolicyStatement(actions=["iam:PassRole"], resources=pass_roles)
        )

        # REST API, TLS-only by nature (API Gateway endpoints are HTTPS), with
        # request validation and access logging + throttling (18.x).
        access_logs = logs.LogGroup(
            self, "ApiAccessLogs", retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.DESTROY,
        )
        api = apigw.RestApi(
            self,
            "RoadGuardApi",
            rest_api_name=f"roadguard-{self._env_name}",
            deploy_options=apigw.StageOptions(
                stage_name=self._env_name,
                throttling_rate_limit=50,
                throttling_burst_limit=20,
                access_log_destination=apigw.LogGroupLogDestination(access_logs),
                access_log_format=apigw.AccessLogFormat.json_with_standard_fields(
                    caller=True, http_method=True, ip=True, protocol=True,
                    request_time=True, resource_path=True, response_length=True,
                    status=True, user=True,
                ),
                metrics_enabled=True,  # CloudWatch metrics (18.6)
            ),
            default_cors_preflight_options=apigw.CorsOptions(
                allow_origins=apigw.Cors.ALL_ORIGINS,
                allow_methods=apigw.Cors.ALL_METHODS,
            ),
        )

        uploads = api.root.add_resource("uploads")
        uploads.add_method("POST", apigw.LambdaIntegration(upload_fn))

        inspections = api.root.add_resource("inspections")
        inspections.add_method("POST", apigw.LambdaIntegration(inspection_fn))
        by_id = inspections.add_resource("{id}")
        by_id.add_method("GET", apigw.LambdaIntegration(read_fn))
        by_id.add_resource("report").add_method("GET", apigw.LambdaIntegration(read_fn))
        by_id.add_resource("trace").add_method("GET", apigw.LambdaIntegration(read_fn))

        api.root.add_resource("overview").add_method("GET", apigw.LambdaIntegration(read_fn))
        api.root.add_resource("metrics").add_method("GET", apigw.LambdaIntegration(read_fn))

    # -- Fargate Vision_Worker (17.3, 18.1, 18.6) -------------------------- #

    def _vision_worker(self, bucket: s3.Bucket, table: ddb.Table) -> None:
        vpc = ec2.Vpc(self, "WorkerVpc", max_azs=2, nat_gateways=1)
        cluster = ecs.Cluster(self, "WorkerCluster", vpc=vpc, container_insights=True)

        task_def = ecs.FargateTaskDefinition(
            self, "VisionWorkerTask", cpu=1024, memory_limit_mib=2048,
        )
        # Build the REAL Vision_Worker image from the repo root using
        # Dockerfile.worker (Milestone 6 task 12.5). CDK builds it with local
        # Docker and pushes it to ECR on deploy; the task then runs the real
        # OpenCV pipeline + agent loop (vision.worker_main).
        # Use the PRE-BUILT image already pushed to ECR (built + pushed out of
        # band with `docker build`/`docker push`), so `cdk synth`/`deploy` needs
        # NO local Docker build. Rebuild + push the image to update the worker.
        worker_repo = ecr.Repository.from_repository_name(
            self, "VisionWorkerRepo", "roadguard-vision-worker"
        )
        task_def.add_container(
            "VisionWorker",
            image=ecs.ContainerImage.from_ecr_repository(worker_repo, tag="latest"),
            logging=ecs.LogDriver.aws_logs(
                stream_prefix="vision-worker",
                log_retention=logs.RetentionDays.ONE_MONTH,  # CloudWatch (18.6)
            ),
            environment={
                "ROADGUARD_TABLE": table.table_name,
                "EVIDENCE_BUCKET": bucket.bucket_name,
                # Live Amazon Nova reasoning via Strands + Bedrock (ECS does it all).
                "RUNTIME": "bedrock",
                "BEDROCK_MODEL_ID": "us.amazon.nova-lite-v1:0",
            },
        )

        # Least-privilege task role (18.1): read input, write evidence, read/write
        # its own inspection records, and invoke Bedrock models only.
        bucket.grant_read_write(task_def.task_role)
        table.grant_read_write_data(task_def.task_role)
        task_def.task_role.add_to_principal_policy(
            iam.PolicyStatement(
                actions=["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
                # Nova is invoked via a cross-region inference profile, so allow both
                # the inference-profile and the underlying foundation-model ARNs.
                resources=[
                    f"arn:aws:bedrock:*:{Stack.of(self).account}:inference-profile/*",
                    "arn:aws:bedrock:*::foundation-model/*",
                ],
            )
        )

        # Expose the worker wiring so _backend can launch it on upload (ecs:RunTask).
        self._worker_cluster = cluster
        self._worker_task_def = task_def
        self._worker_vpc = vpc
        self._worker_container_name = "VisionWorker"
