# Architecture

## Overview
The system is built on AWS using a serverless architecture for inference and containerized environments for training and hosting.

## Components

### Data Pipeline
- **Source**: MVTec AD dataset.
- **Storage**: S3 Bucket (`mvtec-ad-project-data`).
- **Processing**: Python scripts convert raw images to TFRecords.

### Training
- **Environment**: GitHub Actions (CI) or SageMaker Notebooks.
- **Models**:
  - **CNN Custom**: Baseline model.
  - **ResNet50/EfficientNet**: Transfer learning.
  - **Autoencoder**: Unsupervised anomaly detection.
- **Artifacts**: Saved models (.h5) stored in S3 (`mvtec-ad-project-models`).

### Inference
- **Compute**: AWS Lambda (Container Image).
- **Model**: TensorFlow Lite (quantized).
- **API**: AWS API Gateway (REST).

### Frontend
- **Framework**: Streamlit.
- **Hosting**: Docker container (can be deployed to ECS/Fargate or EC2).

## Diagram
(Mermaid diagram placeholder)
```mermaid
graph TD
    User[User] -->|Upload Image| Streamlit[Streamlit App]
    Streamlit -->|POST /predict| APIGW[API Gateway]
    APIGW -->|Invoke| Lambda[AWS Lambda]
    Lambda -->|Load| S3Model[S3 (Model)]
    Lambda -->|Return Prediction| Streamlit
```
