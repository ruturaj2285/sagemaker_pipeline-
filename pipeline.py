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
# Load .env
# ------------------------------------------------------------------
load_dotenv()

def env_or_none(key: str) -> str:
    print(key) 
    value = os.getenv(key,None) 
    print(value) 
    return value


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
REGION = "ap-northeast-1"
ROLE = "arn:aws:iam::227295996532:role/sagemaker-service-role"
BUCKET = "ml-demo-bucket2286"
PIPELINE_NAME = "SageMakerPipelinePOC-ImageFallback-ENV"


# ------------------------------------------------------------------
# IMAGE RESOLUTION (LATEST → FALLBACK → None)
# ------------------------------------------------------------------
PREPROCESS_IMAGE_V1 = (
    os.getenv("PREPROCESS_IMAGE_V1_LATEST", None)
)

PREPROCESS_IMAGE_V2 = (
    os.getenv("PREPROCESS_IMAGE_V2_LATEST", None)
)

TRAIN_IMAGE_URI = (
    os.getenv("TRAIN_IMAGE_LATEST", None)
)

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

    input_data = ParameterString(
        name="InputData",
        default_value=f"s3://{BUCKET}/data/iris.csv",
    )

    # ==============================================================
    # Step 1A: Preprocess V1 (if image exists)
    # ==============================================================
    step_pre_v1 = None

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

        print(f"step_pre_v1 {step_pre_v1}")
        print(f"step_pre_v1 {type(step_pre_v1)}")

        steps.append(step_pre_v1)

    # ==============================================================
    # Step 1B: Preprocess V2 (depends on V1)
    # ==============================================================
    # step_pre_v2 = None
    

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

    # ==============================================================
    # Step 2: Training (depends on V2)
    # ==============================================================
    # train_step = None

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
            inputs={
                "train": step_pre_v2.properties
                .ProcessingOutputConfig.Outputs["train_data"]
                .S3Output.S3Uri
            },
        )

        steps.append(train_step)

    # ==============================================================
    # Step 3: Register Model
    # ==============================================================
    # if train_step:
    #     register_step = RegisterModel(
    #         name="RegisterModel",
    #         estimator=estimator,
    #         model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
    #         content_types=["text/csv"],
    #         response_types=["text/csv"],
    #         inference_instances=["ml.t2.medium"],
    #         transform_instances=["ml.m5.large"],
    #         model_package_group_name="demo-model-group",
    #         approval_status="PendingManualApproval",
    #     )

    #     steps.append(register_step)

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
