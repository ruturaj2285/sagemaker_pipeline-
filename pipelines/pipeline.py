import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.parameters import ParameterString
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.estimator import Estimator
from sagemaker.workflow.step_collections import RegisterModel  

region = "ap-northeast-1"
role = "arn:aws:iam::227295996532:role/service-role/AmazonSageMaker-ExecutionRole-20251111T094161"

sagemaker_session = sagemaker.session.Session()

# Define parameter for input data
input_data = ParameterString(
    name="InputData", default_value="s3://ml-demo-bucket/data/iris.csv"
)

# Step 1: Preprocessing
processor = SKLearnProcessor(
    framework_version="1.2-1",
    role=role,
    instance_type="ml.m5.large",
    instance_count=1,
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

# Step 2: Training
image_uri = sagemaker.image_uris.retrieve("xgboost", region=region, version="1.5-1")
estimator = Estimator(
    image_uri=image_uri,
    role=role,
    instance_type="ml.m5.large",
    instance_count=1,
    output_path=f"s3://ml-demo-bucket/output/",
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

# Step 3: Register model
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

# Build pipeline
pipeline = Pipeline(
    name="SageMakerPipelinePOC",
    parameters=[input_data],
    steps=[step_process, step_train, step_register],
)

# For GitHub Actions (optional print)
if __name__ == "__main__":
    definition = pipeline.definition()
    print("✅ SageMaker pipeline definition created successfully!")

