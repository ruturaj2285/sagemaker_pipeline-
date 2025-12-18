import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.parameters import ParameterString
from sagemaker.processing import Processor
from sagemaker.estimator import Estimator
from sagemaker.workflow.step_collections import RegisterModel

region = "ap-northeast-1"
role = "arn:aws:iam::227295996532:role/sagemaker-service-role"
bucket = "ml-demo-bucket2286"

PREPROCESS_IMAGE_URI = (
    "227295996532.dkr.ecr.ap-northeast-1.amazonaws.com/"
    "jaime-dev-mdl-data-collection:79773e7"
)

TRAIN_IMAGE_URI = sagemaker.image_uris.retrieve(
    "xgboost", region=region, version="1.5-1"
)

sagemaker_session = sagemaker.session.Session(default_bucket=bucket)

input_data = ParameterString(
    name="InputData",
    default_value=f"s3://{bucket}/data/iris.csv"
)

# -----------------------------
# Preprocessing (Docker Image)
# -----------------------------
processor = Processor(
    image_uri=PREPROCESS_IMAGE_URI,
    role=role,
    instance_type="ml.m5.large",
    instance_count=1,
    sagemaker_session=sagemaker_session,
)

step_process = ProcessingStep(
    name="PreprocessData",
    processor=processor,
    inputs=[
        sagemaker.processing.ProcessingInput(
            source=input_data,
            destination="/opt/ml/processing/input"
        )
    ],
    outputs=[
        sagemaker.processing.ProcessingOutput(
            output_name="train_data",
            source="/opt/ml/processing/output"
        )
    ],
)

# -----------------------------
# Training
# -----------------------------
estimator = Estimator(
    image_uri=TRAIN_IMAGE_URI,
    role=role,
    instance_type="ml.m5.large",
    instance_count=1,
    output_path=f"s3://{bucket}/output/",
    sagemaker_session=sagemaker_session,
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

# -----------------------------
# Register Model
# -----------------------------
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

pipeline = Pipeline(
    name="SageMakerPipelinePOC2637262",
    parameters=[input_data],
    steps=[step_process, step_train, step_register],
    sagemaker_session=sagemaker_session,
)

# if __name__ == "__main__":
#     print("Updating SageMaker pipeline definition only.ddd..")

#     pipeline.upsert(role_arn=role)

#     details = pipeline.describe()
#     print("Pipeline updated successfully")
#     print("Pipeline ARN:", details["PipelineArn"])

def upsert_pipeline():
    print("🔄 Updating SageMaker pipeline definition...")
    pipeline.upsert(role_arn=role)
    details = pipeline.describe()
    print("✅ Pipeline updated successfully")
    print("🔗 Pipeline ARN:", details["PipelineArn"])


if __name__ == "__main__":
    upsert_pipeline()
