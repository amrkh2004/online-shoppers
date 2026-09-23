output "dvc_bucket_name" {
  description = "The name of the created S3 bucket for DVC remote"
  value       = aws_s3_bucket.dvc_remote.id
}

output "dvc_bucket_arn" {
  description = "The ARN of the created S3 bucket"
  value       = aws_s3_bucket.dvc_remote.arn
}

output "dvc_remote_url" {
  description = "The DVC remote URL format for s3"
  value       = "s3://${aws_s3_bucket.dvc_remote.id}"
}
