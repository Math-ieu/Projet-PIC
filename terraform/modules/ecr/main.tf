resource "aws_ecr_repository" "training_repo" {
  name = var.training_repo_name
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository" "app_repo" {
  name = var.app_repo_name
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository" "inference_repo" {
  name = var.inference_repo_name
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration {
    scan_on_push = true
  }
}

variable "training_repo_name" {
  type = string
}

variable "app_repo_name" {
  type = string
}

variable "inference_repo_name" {
  type = string
}

output "training_repo_url" {
  value = aws_ecr_repository.training_repo.repository_url
}

output "app_repo_url" {
  value = aws_ecr_repository.app_repo.repository_url
}

output "inference_repo_url" {
  value = aws_ecr_repository.inference_repo.repository_url
}
