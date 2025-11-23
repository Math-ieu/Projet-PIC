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
  repo_name = var.ecr_repo_name
}

module "lambda" {
  source = "./modules/lambda"
  function_name = var.lambda_function_name
  role_arn = module.iam.lambda_role_arn
  image_uri = "${module.ecr.repository_url}:latest"
}

# Optional SageMaker
# module "sagemaker" {
#   source = "./modules/sagemaker"
#   role_arn = module.iam.lambda_role_arn # Should be a separate role in prod
# }
