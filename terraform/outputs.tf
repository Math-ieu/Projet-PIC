output "api_endpoint" {
  value = module.lambda.api_endpoint
}

output "s3_data_bucket" {
  value = module.s3.data_bucket_id
}

output "app_repo_url" {
  value = module.ecr.app_repo_url
}

output "inference_repo_url" {
  value = module.ecr.inference_repo_url
}
