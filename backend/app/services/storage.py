"""S3-Compatible Storage Service using Boto3"""

import io
import logging
from datetime import datetime
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from app.core.config import settings

logger = logging.getLogger(__name__)


class StorageService:
    """Service for handling file uploads to S3-compatible storage"""
    
    def __init__(self):
        """Initialize the S3 client with credentials from settings"""
        self.s3_client = boto3.client(
            's3',
            endpoint_url=settings.STORAGE_ENDPOINT_URL,
            aws_access_key_id=settings.STORAGE_ACCESS_KEY_ID,
            aws_secret_access_key=settings.STORAGE_SECRET_ACCESS_KEY,
        )
        self.bucket_name = settings.STORAGE_BUCKET_NAME
    
    def upload_file(
        self,
        file_bytes: bytes,
        file_name: str,
        content_type: str = "image/png"
    ) -> str:
        """
        Upload a file to S3-compatible storage.
        
        Args:
            file_bytes: The file content as bytes
            file_name: The name/key for the file in the bucket
            content_type: MIME type of the file
            
        Returns:
            Public URL of the uploaded file
            
        Raises:
            Exception if upload fails
        """
        try:
            # Create a file-like object from bytes
            file_obj = io.BytesIO(file_bytes)
            
            # Generate a unique filename with timestamp
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            unique_file_name = f"{timestamp}_{file_name}"
            
            # Upload to S3
            self.s3_client.upload_fileobj(
                file_obj,
                self.bucket_name,
                unique_file_name,
                ExtraArgs={
                    'ContentType': content_type,
                    'ACL': 'public-read'
                }
            )
            
            # Construct the public URL
            # Format depends on the S3-compatible service
            # For most services: https://{bucket}.{endpoint}/{key}
            # For Cloudflare R2 with custom domain: https://{domain}/{key}
            
            # Generic approach - construct from endpoint
            if settings.STORAGE_ENDPOINT_URL:
                # Remove protocol from endpoint
                endpoint = settings.STORAGE_ENDPOINT_URL.replace('https://', '').replace('http://', '')
                file_url = f"https://{self.bucket_name}.{endpoint}/{unique_file_name}"
            else:
                # Fallback to standard S3 URL format
                file_url = f"https://{self.bucket_name}.s3.amazonaws.com/{unique_file_name}"
            
            logger.info(f"Successfully uploaded file: {unique_file_name}")
            return file_url
            
        except ClientError as e:
            logger.error(f"Failed to upload file to S3: {e}")
            raise Exception(f"Storage upload failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during file upload: {e}")
            raise
    
    def delete_file(self, file_name: str) -> bool:
        """
        Delete a file from storage.
        
        Args:
            file_name: The name/key of the file to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=file_name
            )
            logger.info(f"Successfully deleted file: {file_name}")
            return True
        except ClientError as e:
            logger.error(f"Failed to delete file from S3: {e}")
            return False

