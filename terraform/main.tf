# 1. AWS S3 Bucket for DVC Remote Storage
resource "aws_s3_bucket" "dvc_remote" {
  bucket        = var.bucket_name
  force_destroy = true

  tags = {
    Name        = "DVC Remote Storage"
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

# 2. Enable S3 Bucket Versioning
resource "aws_s3_bucket_versioning" "dvc_remote_versioning" {
  bucket = aws_s3_bucket.dvc_remote.id
  versioning_configuration {
    status = "Enabled"
  }
}

# 3. Server-Side Encryption (AES256)
resource "aws_s3_bucket_server_side_encryption_configuration" "dvc_remote_sse" {
  bucket = aws_s3_bucket.dvc_remote.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# 4. Block Public Access Controls
resource "aws_s3_bucket_public_access_block" "dvc_remote_public_block" {
  bucket = aws_s3_bucket.dvc_remote.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
