# Migration to EC2 + Docker + MLflow

## Goal
Replace the current serverless architecture (Lambda, App Runner, SageMaker) with a single EC2 instance running Docker containers for:
1.  **MLflow Tracking Server**: For experiment tracking and model registry.
2.  **Streamlit Application**: For production inference (loading models from MLflow).
3.  **Training Jobs**: Executed on the same EC2 (or via docker run) logging to MLflow.

## User Review Required
> [!IMPORTANT]
> **Data Persistence**: MLflow data (experiments, models) will be stored on the EC2 instance volume. For production, we should mount an EBS volume or use S3 for artifacts and RDS for the backend store. For this iteration, we will use **local storage on EC2** and **SQLite** for simplicity, with **S3** as the artifact root if possible, or just local disk.
> *Decision*: We will use S3 for MLflow artifacts to ensure models are safe, but SQLite for the backend store on the EC2.

## Proposed Changes

### Terraform (`terraform/`)
#### [DELETE]
- `modules/lambda`
- `modules/app_runner`
- `modules/sagemaker`
- `modules/ecr` (We can keep ECR if we want to push images there, but for a single EC2 we can also just build locally or pull from ECR. Let's keep ECR for professional workflow).

#### [NEW] `modules/ec2`
- **Resource**: `aws_instance` (t3.medium or larger for training).
- **Security Group**: Allow ports 22 (SSH), 80 (HTTP), 5000 (MLflow), 8501 (Streamlit).
- **IAM Role**: Allow EC2 to read/write S3 (for MLflow artifacts) and pull from ECR.
- **User Data**: Script to install Docker, Docker Compose, and start the services.

### Docker (`/`)
#### [NEW] `docker-compose.yml`
- **Service: mlflow**
    - Image: `ghcr.io/mlflow/mlflow`
    - Command: `mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root s3://<bucket_name>/mlflow-artifacts --host 0.0.0.0`
    - Ports: `5000:5000`
- **Service: streamlit-app**
    - Build: `.` (Dockerfile.app)
    - Environment: `MLFLOW_TRACKING_URI=http://mlflow:5000`
    - Ports: `8501:8501`
    - Depends on: `mlflow`

### Scripts (`scripts/`)
#### [MODIFY] `train_all_models.py`
- Add `mlflow.set_tracking_uri("http://localhost:5000")` (or configurable).
- Add `mlflow.start_run()`, `mlflow.log_params()`, `mlflow.log_metrics()`, `mlflow.log_artifact()`.

## Verification Plan
1.  **Deploy Infra**: `terraform apply` -> Verify EC2 is running.
2.  **Check Services**: SSH into EC2, check `docker ps`.
3.  **Run Training**: SSH into EC2, run `python scripts/train_all_models.py`. Verify runs appear in MLflow UI (port 5000).
4.  **Test App**: Access Streamlit (port 8501), verify it can load the best model from MLflow.
