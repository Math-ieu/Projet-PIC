variable "aws_region" {
  default = "eu-west-1"
}

variable "bucket_name" {
  default = "mvtec-ad-project-data"
}

variable "model_bucket_name" {
  default = "mvtec-ad-project-models"
}

variable "ecr_repo_name" {
  default = "anomaly-detection-lambda"
}

variable "lambda_function_name" {
  default = "anomaly-detection-inference"
}
