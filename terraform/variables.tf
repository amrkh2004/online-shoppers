variable "aws_region" {
  description = "AWS region for infrastructure deployment"
  type        = string
  default     = "us-east-1"
}

variable "bucket_name" {
  description = "S3 bucket name used as DVC remote storage"
  type        = string
  default     = "online-shoppers-dvc-remote-storage"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}
