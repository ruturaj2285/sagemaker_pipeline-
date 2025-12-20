import os
from dotenv import load_dotenv

import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.parameters import ParameterString
from sagemaker.processing import Processor, ProcessingInput, ProcessingOutput
from sagemaker.estimator import Estimator
from sagemaker.workflow.pipeline_context import PipelineSession

# ------------------------------------------------------------------
# Load .env
# ------------------------------------------------------------------
load_dotenv()

def env_or_none(key: str):
    value = os.getenv(key)
    return value if value not in ("", "None", None) else None


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
REGION = "ap-northeast-1"
ROLE = "arn:aws:iam::227295996532:role/sagemaker-service-role"
BUCKET = "ml-demo-bucket2286"
PIPELINE_NAME = "SageMakerPipelinePOC-ImageFallback-ENV"


# ------------------------------------------------------------------
# Image selection from .env
# ------------------------------------------------------------------
PREPROCESS_IMAGE_V1 = env_or_none("PREPROCESS_IMAGE_V1_LATEST")
PREPROCESS_IMAGE_V2 = env_or_none("PREPROCESS_IMAGE_V2_LATEST")
TRAIN_IMAGE_URI     = env_or_none("TRAIN_IMAGE_LATEST")

print("🖼 Image selection from .env")
print("PREPROCESS_IMAGE_V1 =", PREPROCESS_IMAGE_V1)
print("PREPROCESS_IMAGE_V2 =", PREPROCESS_IMAGE_V2)
print("TRAIN_IMAGE_URI     =", TRAIN_IMAGE_URI)


# ------------------------------------------------------------------
# Pipeline factory
# ------------------------------------------------------------------
def get_pipeline():

    session = PipelineSession()
    steps = []

    # --------------------------------------------------------------
    # Pipeline parameters
    # --------------------------------------------------------------
    input_data = ParameterString(
        name="InputData",
        default_value=f"s3://{BUCKET}/data/iris.csv",
    )

    # 🔑 Pointer to the latest valid output
    last_output_uri = input_data

    # ==============================================================
    # Step 1: Preprocess V1 (optional)
    # ==============================================================
    if PREPROCESS_IMAGE_V1:
        processor_v1 = Processor(
            image_uri=PREPROCESS_IMAGE_V1,
            role=ROLE,
            instance_type="ml.m5.large",
            instance_count=1,
            sagemaker_session=session,
        )

        step_pre_v1 = ProcessingStep(
            name="PreprocessDataV1",
            processor=processor_v1,
            inputs=[
                ProcessingInput(
                    source=last_output_uri,
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

        last_output_uri = (
            step_pre_v1.properties
            .ProcessingOutputConfig.Outputs["train_data"]
            .S3Output.S3Uri
        )

    # ==============================================================
    # Step 2: Preprocess V2 (optional)
    # ==============================================================
    if PREPROCESS_IMAGE_V2:
        processor_v2 = Processor(
            image_uri=PREPROCESS_IMAGE_V2,
            role=ROLE,
            instance_type="ml.m5.large",
            instance_count=1,
            sagemaker_session=session,
        )

        step_pre_v2 = ProcessingStep(
            name="PreprocessDataV2",
            processor=processor_v2,
            inputs=[
                ProcessingInput(
                    source=last_output_uri,
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

        last_output_uri = (
            step_pre_v2.properties
            .ProcessingOutputConfig.Outputs["train_data"]
            .S3Output.S3Uri
        )

    # ==============================================================
    # Step 3: Training (optional)
    # ==============================================================
    if TRAIN_IMAGE_URI:
        estimator = Estimator(
            image_uri=TRAIN_IMAGE_URI,
            role=ROLE,
            instance_type="ml.m5.large",
            instance_count=1,
            output_path=f"s3://{BUCKET}/output/",
            sagemaker_session=session,
        )

        train_step = TrainingStep(
            name="TrainModel",
            estimator=estimator,
            inputs={"train": last_output_uri},
        )

        steps.append(train_step)

    # ==============================================================
    # Pipeline
    # ==============================================================
    pipeline = Pipeline(
        name=PIPELINE_NAME,
        parameters=[input_data],
        steps=steps,
        sagemaker_session=session,
    )

    return pipeline


# ------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------
if __name__ == "__main__":
    pipeline = get_pipeline()
    print("🔄 Upserting SageMaker pipeline definition...")
    pipeline.upsert(role_arn=ROLE)
    print("✅ Pipeline updated successfully")
