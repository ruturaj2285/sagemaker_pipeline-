import os
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.parameters import ParameterString
from sagemaker.processing import Processor
from sagemaker.estimator import Estimator
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.pipeline_context import PipelineSession

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
region = "ap-northeast-1"
role = "arn:aws:iam::227295996532:role/sagemaker-service-role"
bucket = "ml-demo-bucket2286"

# ------------------------------------------------------------------
# Image URIs (OPTIONAL – step created only if present)
# ------------------------------------------------------------------
PREPROCESS_IMAGE_URI = os.getenv(
    "MDL_PRE_PROCESSING_IMAGE",
    "227295996532.dkr.ecr.ap-northeast-1.amazonaws.com/"
    "jaime-dev-mdl-data-collection:79773e7",
)

TRAIN_IMAGE_URI = os.getenv(
    "MDL_TRAINING_IMAGE",
    sagemaker.image_uris.retrieve(
        "xgboost", region=region, version="1.5-1"
    ),
)

# ------------------------------------------------------------------
# Session
# ------------------------------------------------------------------
session = PipelineSession(default_bucket=bucket)

# ------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------
input_data = ParameterString(
    name="InputData",
    default_value=f"s3://{bucket}/data/iris.csv"
)

# ------------------------------------------------------------------
# Pipeline factory
# ------------------------------------------------------------------
def get_pipeline():
    # ==============================================================
    # Step 1: Preprocessing (OPTIONAL)
    # ==============================================================
    step_process = None
    if PREPROCESS_IMAGE_URI:
        processor = Processor(
            image_uri=PREPROCESS_IMAGE_URI,
            role=role,
            instance_type="ml.m5.large",
            instance_count=1,
            sagemaker_session=session,
        )

        step_process = ProcessingStep(
            name="PreprocessData",
            processor=processor,
            inputs=[
                sagemaker.processing.ProcessingInput(
                    source=input_data,
                    destination="/opt/ml/processing/input",
                )
            ],
            outputs=[
                sagemaker.processing.ProcessingOutput(
                    output_name="train_data",
                    source="/opt/ml/processing/output",
                )
            ],
        )

    # ==============================================================
    # Step 2: Training (OPTIONAL)
    # ==============================================================
    step_train = None
    estimator = None
    if TRAIN_IMAGE_URI and step_process:
        estimator = Estimator(
            image_uri=TRAIN_IMAGE_URI,
            role=role,
            instance_type="ml.m5.large",
            instance_count=1,
            output_path=f"s3://{bucket}/output/",
            sagemaker_session=session,
        )

        step_train = TrainingStep(
            name="TrainModel",
            estimator=estimator,
            inputs={
                "train": step_process.properties
                .ProcessingOutputConfig.Outputs["train_data"]
                .S3Output.S3Uri
            },
        )

    # ==============================================================
    # Step 3: Register Model (OPTIONAL)
    # ==============================================================
    step_register = None
    if step_train:
        step_register = RegisterModel(
            name="RegisterModel",
            estimator=estimator,
            model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
            content_types=["text/csv"],
            response_types=["text/csv"],
            inference_instances=["ml.t2.medium"],
            transform_instances=["ml.m5.large"],
            model_package_group_name="demo-model-group",
            approval_status="PendingManualApproval",
        )

    # ==============================================================
    # Build steps list ONCE (no append)
    # ==============================================================
    steps = [step_process, step_train, step_register]
    steps = [s for s in steps if s]

    if not steps:
        raise RuntimeError(
            "Pipeline must contain at least one step. "
            "Check MDL_PRE_PROCESSING_IMAGE / MDL_TRAINING_IMAGE env vars."
        )

    # ==============================================================
    # Pipeline
    # ==============================================================
    pipeline = Pipeline(
        name="SageMakerPipelinePOc11111111111111111111111111111111111111111111",
        parameters=[input_data],
        steps=steps,
        sagemaker_session=session,
    )

    return pipeline


# ------------------------------------------------------------------
# Local / CI entrypoint
# ------------------------------------------------------------------
def upsert_pipeline():
    pipe = get_pipeline()
    print("🔄 Updating SageMaker pipeline definition...")
    pipe.upsert(role_arn=role)
    details = pipe.describe()
    print("✅ Pipeline updated successfully")
    print("🔗 Pipeline ARN:", details["PipelineArn"])


if __name__ == "__main__":
    upsert_pipeline()
