resource "aws_s3_bucket" "data_bucket" {
  bucket = var.bucket_name
}

resource "aws_s3_bucket" "model_bucket" {
  bucket = var.model_bucket_name
}

variable "bucket_name" {
  type = string
}

variable "model_bucket_name" {
  type = string
}

output "data_bucket_id" {
  value = aws_s3_bucket.data_bucket.id
}

output "model_bucket_id" {
  value = aws_s3_bucket.model_bucket.id
}
