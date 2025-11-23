output "api_endpoint" {
  value = module.lambda.api_endpoint
}

output "s3_data_bucket" {
  value = module.s3.data_bucket_id
}

output "ecr_repo_url" {
  value = module.ecr.repository_url
}
