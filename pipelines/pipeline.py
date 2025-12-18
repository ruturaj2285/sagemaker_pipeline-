import argparse
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.parameters import ParameterString
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.estimator import Estimator
from sagemaker.workflow.step_collections import RegisterModel

# --------------------------------------------------
# Read environment
# --------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    required=True,
    choices=["development", "staging", "production"]
)
args = parser.parse_args()
ENV = args.env

print(f"✅ Running pipeline for environment: {ENV}")

# --------------------------------------------------
# Environment-specific config
# --------------------------------------------------
REGION = "ap-northeast-1"

ROLE_MAP = {
    "development": "arn:aws:iam::227295996532:role/sagemaker-service-role-dev",
    "staging": "arn:aws:iam::227295996532:role/sagemaker-service-role-stg",
    "production": "arn:aws:iam::227295996532:role/sagemaker-service-role-prod",
}

BUCKET_MAP = {
    "development": "ml-demo-bucket-dev",
    "staging": "ml-demo-bucket-stg",
    "production": "ml-demo-bucket-prod",
}

PIPELINE_NAME = f"SageMakerPipelinePOC-{ENV}"

role = ROLE_MAP[ENV]
bucket = BUCKET_MAP[ENV]

# --------------------------------------------------
# SageMaker session
# --------------------------------------------------
sagemaker_session = sagemaker.Session()

# --------------------------------------------------
# Pipeline parameters
# --------------------------------------------------
input_data = ParameterString(
    name="InputData",
    default_value=f"s3://{bucket}/data/iris.csv"
)

# --------------------------------------------------
# Step 1: Preprocessing
# --------------------------------------------------
processor = SKLearnProcessor(
    framework_version="1.2-1",
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
            destination="/opt/ml/processing/input/"
        )
    ],
    outputs=[
        sagemaker.processing.ProcessingOutput(
            output_name="train_data",
            source="/opt/ml/processing/output/"
        )
    ],
    code="src/preprocessing.py",
)

# --------------------------------------------------
# Step 2: Training
# --------------------------------------------------
image_uri = sagemaker.image_uris.retrieve(
    framework="xgboost",
    region=REGION,
    version="1.5-1"
)

estimator = Estimator(
    image_uri=image_uri,
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

# --------------------------------------------------
# Step 3: Register Model
# --------------------------------------------------
step_register = RegisterModel(
    name="RegisterModel",
    estimator=estimator,
    model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["text/csv"],
    response_types=["text/csv"],
    inference_instances=["ml.t2.medium"],
    transform_instances=["ml.m5.large"],
    model_package_group_name=f"demo-model-group-{ENV}",
    approval_status="PendingManualApproval" if ENV == "production" else "Approved",
)

# --------------------------------------------------
# Build pipeline
# --------------------------------------------------
pipeline = Pipeline(
    name=PIPELINE_NAME,
    parameters=[input_data],
    steps=[step_process, step_train, step_register],
    sagemaker_session=sagemaker_session,
)

# --------------------------------------------------
# Create / Update / Execute
# --------------------------------------------------
if __name__ == "__main__":
    print("✅ Creating / Updating pipeline...")
    pipeline.upsert(role_arn=role)

    print("✅ Starting pipeline execution...")
    execution = pipeline.start()
    print(f"🚀 Pipeline execution ARN: {execution.arn}")
