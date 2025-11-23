resource "aws_sagemaker_notebook_instance" "ni" {
  name          = "mvtec-ad-notebook"
  role_arn      = var.role_arn
  instance_type = "ml.t2.medium"
}

variable "role_arn" { type = string }
