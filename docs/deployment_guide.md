# Deployment Guide

This guide details the steps to deploy the Visual Anomaly Detection project to AWS.

## Prerequisites

Ensure you have the following tools installed and configured:

*   **AWS CLI**: Configured with `aws configure` (Region: `eu-west-1`).
*   **Terraform**: v1.0+.
*   **Docker**: Running locally.
*   **Make**: For running convenience commands.

## 1. Initial Deployment (Bootstrap)

When deploying for the very first time, there is a circular dependency:
1.  Terraform creates ECR repositories.
2.  Terraform tries to create Lambda/App Runner services which *require* Docker images.
3.  Docker images cannot be pushed until ECR repositories exist.

To solve this, we use a 2-step bootstrap process:

### Step 1: Create ECR Repositories Only
Run Terraform targeting only the ECR module to create the empty repositories.

```bash
cd terraform
terraform init
terraform apply -target=module.ecr
```

### Step 2: Build and Push Images
Once ECR repos exist, build and push the Docker images. You can use the provided Makefile commands, but you first need to login to ECR.

```bash
# Login to ECR
aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin <YOUR_AWS_ACCOUNT_ID>.dkr.ecr.eu-west-1.amazonaws.com

# Build and Push Streamlit App
docker build -f Dockerfile.app -t <YOUR_AWS_ACCOUNT_ID>.dkr.ecr.eu-west-1.amazonaws.com/mvtec-ad-streamlit:latest .
docker push <YOUR_AWS_ACCOUNT_ID>.dkr.ecr.eu-west-1.amazonaws.com/mvtec-ad-streamlit:latest

# Build and Push Inference Lambda
docker build -t <YOUR_AWS_ACCOUNT_ID>.dkr.ecr.eu-west-1.amazonaws.com/mvtec-ad-inference:latest inference/lambda_function/
docker push <YOUR_AWS_ACCOUNT_ID>.dkr.ecr.eu-west-1.amazonaws.com/mvtec-ad-inference:latest
```

### Step 3: Deploy Remaining Infrastructure
Now that images exist, deploy the rest of the infrastructure (Lambda, App Runner, API Gateway, etc.).

```bash
cd terraform
terraform apply
```

## 2. Automated Deployment (CI/CD)

Once the initial infrastructure is up, GitHub Actions handles updates automatically.

*   **Infrastructure Updates**: Pushing changes to `terraform/**` triggers `deploy-infrastructure.yml`.
*   **Application Updates**: Pushing changes to `streamlit_app/**` or `inference/**` triggers `deploy-app.yml`, which:
    1.  Builds new Docker images.
    2.  Pushes to ECR.
    3.  Updates the Lambda function code.
    *Note: App Runner automatically re-deploys when a new image is pushed if configured with auto-deploy.*

## 3. Manual Updates (Optional)

If you need to manually update without git push:

**Update Infrastructure:**
```bash
make deploy-infra
```

**Update Images (Requires manual docker commands as in Step 2):**
Currently, the `Makefile` has `docker-build` commands but they tag locally. You would need to tag with the ECR registry URL and push.

## Troubleshooting

*   **App Runner Creation Fails**: Ensure the `mvtec-ad-streamlit` image exists in ECR before running `terraform apply`.
*   **Lambda Error**: If the Lambda function fails to start, check CloudWatch Logs. Ensure the Docker image entrypoint is correct.
