# Deployment Guide

## Prerequisites
- AWS Account
- AWS CLI configured
- Terraform installed
- Docker installed

## Steps

### 1. Infrastructure
Initialize and apply Terraform configuration:
```bash
cd terraform
terraform init
terraform apply
```
Note the `api_endpoint` output.

### 2. Training
Run the training pipeline locally or via GitHub Actions:
```bash
python scripts/train_all_models.py
python scripts/select_best_model.py
```

### 3. Inference Deployment
Build and push the Lambda image:
```bash
# Login to ECR
aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin <account_id>.dkr.ecr.eu-west-1.amazonaws.com

# Build & Push
docker build -t anomaly-detection-lambda inference/lambda_function/
docker tag anomaly-detection-lambda:latest <repo_url>:latest
docker push <repo_url>:latest

# Update Lambda
aws lambda update-function-code --function-name anomaly-detection-inference --image-uri <repo_url>:latest
```

### 4. Streamlit App
Run locally:
```bash
streamlit run streamlit_app/app.py
```
Or build Docker image:
```bash
docker build -t anomaly-detection-app streamlit_app/
docker run -p 8501:8501 anomaly-detection-app
```
