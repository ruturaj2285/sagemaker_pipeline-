FROM public.ecr.aws/sagemaker/sklearn:1.2-1-cpu-py3

WORKDIR /opt/ml/code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY preprocessing.py .

ENTRYPOINT ["python", "preprocessing.py"]
