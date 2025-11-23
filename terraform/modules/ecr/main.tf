resource "aws_ecr_repository" "lambda_repo" {
  name = var.repo_name
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

variable "repo_name" {
  type = string
}

output "repository_url" {
  value = aws_ecr_repository.lambda_repo.repository_url
}
