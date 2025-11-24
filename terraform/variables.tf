variable "aws_region" {
  default = "eu-west-1"
}

variable "bucket_name" {
  default = "mvtec-ad-project-data"
}

variable "model_bucket_name" {
  default = "mvtec-ad-project-models"
}

variable "training_repo_name" {
  default = "mvtec-ad-training"
}

variable "app_repo_name" {
  default = "mvtec-ad-streamlit"
}

variable "inference_repo_name" {
  default = "mvtec-ad-inference"
}

variable "lambda_function_name" {
  default = "anomaly-detection-inference"
}
