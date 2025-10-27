"""UploadThing Storage Service"""

import io
import logging
from datetime import datetime
from typing import Optional
import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


class StorageService:
    """Service for handling file uploads to UploadThing"""
    
    def __init__(self):
        """Initialize the UploadThing client with credentials from settings"""
        self.secret = settings.UPLOADTHING_SECRET
        self.app_id = settings.UPLOADTHING_APP_ID
        # UploadThing API base URL
        self.base_url = "https://uploadthing.com"
    
    def upload_file(
        self,
        file_bytes: bytes,
        file_name: str,
        content_type: str = "image/png"
    ) -> str:
        """
        Upload a file to UploadThing.
        
        Args:
            file_bytes: The file content as bytes
            file_name: The name/key for the file
            content_type: MIME type of the file
            
        Returns:
            Public URL of the uploaded file
            
        Raises:
            Exception if upload fails
        """
        try:
            # Generate a unique filename with timestamp
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            unique_file_name = f"{timestamp}_{file_name}"
            file_size = len(file_bytes)
            
            # Step 1: Request presigned URL from UploadThing
            headers = {
                "Content-Type": "application/json",
                "x-uploadthing-api-key": self.secret
            }
            
            # Prepare request payload for presigned URL
            payload = {
                "files": [
                    {
                        "name": unique_file_name,
                        "size": file_size,
                        "type": content_type
                    }
                ],
                "appId": self.app_id
            }
            
            with httpx.Client() as client:
                # Get presigned URL
                logger.info(f"Requesting presigned URL for file: {unique_file_name}")
                response = client.post(
                    f"{self.base_url}/api/uploadFiles",
                    headers=headers,
                    json=payload,
                    timeout=30.0
                )
                
                if response.status_code != 200:
                    logger.error(f"Presigned URL request failed: {response.status_code} - {response.text}")
                    raise Exception(f"UploadThing presigned URL error: {response.status_code} - {response.text}")
                
                presigned_data = response.json()
                logger.info(f"Received presigned URL response: {presigned_data}")
                
                # Extract presigned URL and fields
                if not presigned_data or "data" not in presigned_data or len(presigned_data["data"]) == 0:
                    raise Exception("No presigned URL data returned from UploadThing")
                
                upload_data = presigned_data["data"][0]
                presigned_url = upload_data.get("url")
                fields = upload_data.get("fields", {})
                file_url = upload_data.get("fileUrl")
                
                if not presigned_url:
                    raise Exception("No presigned URL in response")
                
                # Step 2: Upload file to presigned URL
                logger.info(f"Uploading file to presigned URL: {presigned_url}")
                
                # Prepare multipart form data for S3-compatible upload
                multipart_data = {
                    **fields,  # Include all required fields from presigned URL
                    "file": (unique_file_name, io.BytesIO(file_bytes), content_type)
                }
                
                upload_response = client.post(
                    presigned_url,
                    files=multipart_data,
                    timeout=60.0
                )
                
                if upload_response.status_code not in [200, 201, 204]:
                    logger.error(f"File upload failed: {upload_response.status_code} - {upload_response.text}")
                    raise Exception(f"File upload failed: {upload_response.status_code}")
                
                # Use the fileUrl from the response
                public_url = file_url
                
                logger.info(f"Successfully uploaded file: {unique_file_name} -> {public_url}")
                return public_url
            
        except Exception as e:
            logger.error(f"Failed to upload file to UploadThing: {e}")
            raise Exception(f"Storage upload failed: {str(e)}")
    
    def delete_file(self, file_url: str) -> bool:
        """
        Delete a file from UploadThing.
        
        Args:
            file_url: The URL of the file to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract file key from URL
            # UploadThing URLs typically look like: https://utfs.io/f/{fileKey}
            if "/f/" in file_url:
                file_key = file_url.split("/f/")[-1]
            else:
                logger.warning(f"Could not extract file key from URL: {file_url}")
                return False
            
            headers = {
                "x-uploadthing-api-key": self.secret,
                "Content-Type": "application/json"
            }
            
            with httpx.Client() as client:
                response = client.delete(
                    f"{self.base_url}/api/deleteFile",
                    headers=headers,
                    json={"fileKey": file_key},
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    logger.info(f"Successfully deleted file: {file_key}")
                    return True
                else:
                    logger.error(f"Failed to delete file: {response.status_code} - {response.text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error deleting file from UploadThing: {e}")
            return False

    def download_file(self, file_url: str) -> bytes:
        """
        Download a file from UploadThing URL.

        Args:
            file_url: The URL of the file to download

        Returns:
            File content as bytes

        Raises:
            Exception if download fails
        """
        try:
            with httpx.Client() as client:
                response = client.get(file_url, timeout=60.0)
                
                if response.status_code == 200:
                    logger.info(f"Successfully downloaded file from: {file_url}")
                    return response.content
                else:
                    logger.error(f"Failed to download file: {response.status_code} - {response.text}")
                    raise Exception(f"Download failed: {response.status_code}")

        except Exception as e:
            logger.error(f"Error downloading file from {file_url}: {e}")
            raise Exception(f"Storage download failed: {str(e)}")

