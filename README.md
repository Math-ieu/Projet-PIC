# Visual Anomaly Detection - MVTec AD

## Overview
End-to-end solution for industrial visual anomaly detection using Deep Learning.
Includes model training, evaluation, selection, and deployment to AWS.

## Architecture
- **Models**: Custom CNN, Transfer Learning (ResNet, EfficientNet), Autoencoder
- **Infrastructure**: AWS Lambda, API Gateway, S3, ECR
- **App**: Streamlit dashboard for real-time inference

## Quick Start
```bash
# Setup
make install

# Download dataset
make download-data

# Train all models
make train

# Deploy infrastructure
make deploy-infra

# Run Streamlit app locally
make run-app
```

## Project Structure
See `docs/architecture.md` for details.