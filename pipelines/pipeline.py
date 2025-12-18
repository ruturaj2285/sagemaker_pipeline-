import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.parameters import ParameterString
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.estimator import Estimator
from sagemaker.workflow.step_collections import RegisterModel

region = "ap-northeast-1"
role = "arn:aws:iam::227295996532:role/sagemaker-service-role"
bucket = "ml-demo-bucket2285"

# Initialize SageMaker session
sagemaker_session = sagemaker.session.Session(default_bucket=bucket)

# Define input parameter for pipeline
input_data = ParameterString(
    name="InputData", default_value=f"s3://{bucket}/data/iris.csv"
)

# -----------------------------
# Step 1: Data Preprocessing 123
# -----------------------------
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
            source=input_data, destination="/opt/ml/processing/input/"
        )
    ],
    outputs=[
        sagemaker.processing.ProcessingOutput(
            output_name="train_data", source="/opt/ml/processing/output/"
        )
    ],
    code="src/preprocessing.py",
)

# -----------------------------
# Step 2: Training
# -----------------------------
image_uri = sagemaker.image_uris.retrieve("xgboost", region=region, version="1.5-1")

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
        "train": step_process.properties.ProcessingOutputConfig.Outputs[
            "train_data"
        ].S3Output.S3Uri
    },
)

# -----------------------------
# Step 3: Register Model
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
    approval_status="PendingManualApproval"
)

# -----------------------------
# Build the Pipeline
# -----------------------------
pipeline = Pipeline(
    name="SageMakerPipelinePOC",
    parameters=[input_data],
    steps=[step_process, step_train, step_register],
    sagemaker_session=sagemaker_session,
)

# -----------------------------
# Create / Update / Run
# -----------------------------
if __name__ == "__main__":
    print("✅ Building SageMaker pipeline definition...")

    # Create or update the pipeline definition in SageMaker
    pipeline.upsert(role_arn=role)
    # print("✅ Pipeline created or updated successfully!")
    details = pipeline.describe()
    print("✅ Pipeline ARN:", details["PipelineArn"])

    # Start execution
    execution = pipeline.start()
    print("✅ Pipeline execution started:", execution.arn)
