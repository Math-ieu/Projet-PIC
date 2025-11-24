provider "aws" {
  region = var.aws_region
}

module "s3" {
  source = "./modules/s3"
  bucket_name = var.bucket_name
  model_bucket_name = var.model_bucket_name
}

module "iam" {
  source = "./modules/iam"
}

module "ecr" {
  source = "./modules/ecr"
  training_repo_name = var.training_repo_name
  app_repo_name = var.app_repo_name
  inference_repo_name = var.inference_repo_name
}

module "lambda" {
  source = "./modules/lambda"
  function_name = var.lambda_function_name
  role_arn = module.iam.lambda_role_arn
  image_uri = "${module.ecr.inference_repo_url}:latest"
}

# SageMaker Training Setup
module "sagemaker" {
  source = "./modules/sagemaker"
}

# Streamlit App Hosting
module "app_runner" {
  source = "./modules/app_runner"
  service_name = "mvtec-ad-streamlit-app"
  image_identifier = "${module.ecr.app_repo_url}:latest"
  port = "8501"
}
