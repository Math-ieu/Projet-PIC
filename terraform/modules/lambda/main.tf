resource "aws_lambda_function" "inference_lambda" {
  function_name = var.function_name
  role          = var.role_arn
  image_uri     = var.image_uri
  package_type  = "Image"
  timeout       = 30
  memory_size   = 1024

  environment {
    variables = {
      MODEL_PATH = "model.tflite"
    }
  }
}

resource "aws_api_gateway_rest_api" "api" {
  name = "${var.function_name}-api"
}

resource "aws_api_gateway_resource" "resource" {
  path_part   = "predict"
  parent_id   = aws_api_gateway_rest_api.api.root_resource_id
  rest_api_id = aws_api_gateway_rest_api.api.id
}

resource "aws_api_gateway_method" "method" {
  rest_api_id   = aws_api_gateway_rest_api.api.id
  resource_id   = aws_api_gateway_resource.resource.id
  http_method   = "POST"
  authorization = "NONE"
}

resource "aws_api_gateway_integration" "integration" {
  rest_api_id             = aws_api_gateway_rest_api.api.id
  resource_id             = aws_api_gateway_resource.resource.id
  http_method             = aws_api_gateway_method.method.http_method
  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = aws_lambda_function.inference_lambda.invoke_arn
}

resource "aws_lambda_permission" "apigw" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.inference_lambda.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_api_gateway_rest_api.api.execution_arn}/*/*"
}

resource "aws_api_gateway_deployment" "deployment" {
  depends_on = [aws_api_gateway_integration.integration]
  rest_api_id = aws_api_gateway_rest_api.api.id
  stage_name  = "prod"
}

variable "function_name" { type = string }
variable "role_arn" { type = string }
variable "image_uri" { type = string }

output "api_endpoint" {
  value = "${aws_api_gateway_deployment.deployment.invoke_url}/predict"
}
