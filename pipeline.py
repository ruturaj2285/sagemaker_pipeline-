import os
from dotenv import load_dotenv

import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.parameters import ParameterString
from sagemaker.processing import Processor, ProcessingInput, ProcessingOutput
from sagemaker.estimator import Estimator
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.pipeline_context import PipelineSession

# ------------------------------------------------------------------
# Load environment variables
# ------------------------------------------------------------------
load_dotenv()


def env_or_none(key: str) -> str:
    print(key)
    value = os.getenv(key,None)
    return value


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
region = "ap-northeast-1"
role = "arn:aws:iam::227295996532:role/sagemaker-service-role"
bucket = "ml-demo-bucket2286"

# ------------------------------------------------------------------
# IMAGE TAG STRATEGY (FROM ENV)  
# ------------------------------------------------------------------
# Latest (optional)
PREPROCESS_IMAGE_V1_LATEST = env_or_none("PREPROCESS_IMAGE_V1_LATEST")
PREPROCESS_IMAGE_V2_LATEST = env_or_none("PREPROCESS_IMAGE_V2_LATEST")
TRAIN_IMAGE_LATEST = env_or_none("TRAIN_IMAGE_LATEST")

# Fallback (stable)
PREPROCESS_IMAGE_V1_FALLBACK = env_or_none("PREPROCESS_IMAGE_V1_FALLBACK")
PREPROCESS_IMAGE_V2_FALLBACK = env_or_none("PREPROCESS_IMAGE_V2_FALLBACK")
TRAIN_IMAGE_FALLBACK = env_or_none("TRAIN_IMAGE_FALLBACK")

# ------------------------------------------------------------------
# FINAL IMAGE SELECTION
# ------------------------------------------------------------------
PREPROCESS_IMAGE_V1 = PREPROCESS_IMAGE_V1_LATEST or PREPROCESS_IMAGE_V1_FALLBACK
PREPROCESS_IMAGE_V2 = PREPROCESS_IMAGE_V2_LATEST or PREPROCESS_IMAGE_V2_FALLBACK
TRAIN_IMAGE_URI = TRAIN_IMAGE_LATEST or TRAIN_IMAGE_FALLBACK

# Fail fast if anything critical is missing
for name, image in {
    "PREPROCESS_IMAGE_V1": PREPROCESS_IMAGE_V1,
    "PREPROCESS_IMAGE_V2": PREPROCESS_IMAGE_V2,
    "TRAIN_IMAGE_URI": TRAIN_IMAGE_URI,
}.items():
    if not image:
        raise RuntimeError(f"{name} is not set via env or fallback")

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

    steps = []

    # ==============================================================
    # Step 1A: Preprocessing V1
    # ==============================================================
    processor_v1 = Processor(
        image_uri=PREPROCESS_IMAGE_V1,
        role=role,
        instance_type="ml.m5.large",
        instance_count=1,
        sagemaker_session=session,
    )

    step_pre_v1 = ProcessingStep(
        name="PreprocessDataV1",
        processor=processor_v1,
        inputs=[
            ProcessingInput(
                source=input_data,
                destination="/opt/ml/processing/input",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="train_data",
                source="/opt/ml/processing/output",
            )
        ],
    )
    steps.append(step_pre_v1)

    # ==============================================================
    # Step 1B: Preprocessing V2
    # ==============================================================
    processor_v2 = Processor(
        image_uri=PREPROCESS_IMAGE_V2,
        role=role,
        instance_type="ml.m5.large",
        instance_count=1,
        sagemaker_session=session,
    )

    step_pre_v2 = ProcessingStep(
        name="PreprocessDataV2",
        processor=processor_v2,
        inputs=[
            ProcessingInput(
                source=step_pre_v1.properties
                .ProcessingOutputConfig.Outputs["train_data"]
                .S3Output.S3Uri,
                destination="/opt/ml/processing/input",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="train_data",
                source="/opt/ml/processing/output",
            )
        ],
    )
    steps.append(step_pre_v2)

    training_input = (
        step_pre_v2.properties
        .ProcessingOutputConfig.Outputs["train_data"]
        .S3Output.S3Uri
    )

    # ==============================================================
    # Step 2: Training
    # ==============================================================
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
        inputs={"train": training_input},
    )
    steps.append(step_train)

    # ==============================================================
    # Step 3: Register Model
    # ==============================================================
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
    steps.append(step_register)

    # ==============================================================
    # Pipeline
    # ==============================================================
    pipeline = Pipeline(
        name="SageMakerPipelinePOC-ImageFallback222222222222222222222",
        parameters=[input_data],
        steps=steps,
        sagemaker_session=session,
    )

    return pipeline


# ------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------
def upsert_pipeline():
    pipeline = get_pipeline()
    print("🔄 Updating SageMaker pipeline definition...")
    pipeline.upsert(role_arn=role)
    print("✅ Pipeline updated successfully")


if __name__ == "__main__":
    upsert_pipeline()
