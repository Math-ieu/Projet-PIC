resource "aws_sagemaker_notebook_instance" "ni" {
  name          = "mvtec-ad-notebook"
  role_arn      = aws_iam_role.sagemaker_execution_role.arn
  instance_type = "ml.t2.medium"
}

resource "aws_iam_role" "sagemaker_execution_role" {
  name = "mvtec_sagemaker_execution_role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "sagemaker_full_access" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
  role       = aws_iam_role.sagemaker_execution_role.name
}

resource "aws_iam_role_policy_attachment" "s3_full_access" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess" # Needed for training data
  role       = aws_iam_role.sagemaker_execution_role.name
}

output "sagemaker_role_arn" {
  value = aws_iam_role.sagemaker_execution_role.arn
}
