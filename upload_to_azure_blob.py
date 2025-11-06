import os
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContentSettings
from pathlib import Path


def upload_directory_to_blob(connection_string: str, container_name: str, source_folder: str):
    """
    Uploads a directory to Azure Blob Storage, maintaining the folder structure.
    
    Args:
        connection_string (str): Azure Storage connection string
        container_name (str): Name of the container to upload to
        source_folder (str): Path to the local folder to upload
    """
    try:
        # Create the BlobServiceClient
        blob_service_client = BlobServiceClient.from_connection_string(
            connection_string)
        container_client = blob_service_client.get_container_client(container_name)
        
        # Create container if it doesn't exist
        try:
            container_client.create_container()
            print(f"Created container: {container_name}")
        except Exception as e:
            if "ContainerAlreadyExists" in str(e):
                print(f"Using existing container: {container_name}")
            else:
                raise
        
        # Convert to Path object for easier path handling
        source_path = Path(source_folder)
        
        # Walk through the directory and upload files
        for root, _, files in os.walk(source_path):
            for file in files:
                file_path = Path(root) / file
                
                # Create the blob path (relative to source_folder)
                blob_path = str(file_path.relative_to(source_path)).replace('\\', '/')
                
                # Get the blob client
                blob_client = container_client.get_blob_client(blob_path)
                
                # Upload the file
                with open(file_path, "rb") as data:
                    # Set content type based on file extension
                    content_settings = ContentSettings(
                        content_type=get_content_type(file_path.suffix.lower())
                    )
                    blob_client.upload_blob(
                        data,
                        overwrite=True,
                        content_settings=content_settings
                    )
                print(f"Uploaded: {blob_path}")
                
        print("\nUpload completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def get_content_type(file_extension: str) -> str:
    """
    Returns the appropriate content type based on file extension.
    """
    content_types = {
        '.txt': 'text/plain',
        '.csv': 'text/csv',
        '.json': 'application/json',
        '.py': 'text/x-python',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.pdf': 'application/pdf',
        '.zip': 'application/zip'
    }
    return content_types.get(file_extension, 'application/octet-stream')


if __name__ == "__main__":
    # Replace these values with your Azure Storage account details
    CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=rlstorageaccounticpna;AccountKey=lcx6O15YOz/05giHJYxxi/W6gnfj4TLGp8SZv0gGb+Yd5O0v4l0fPGc5VROXVPjEMVMaazloYf9J+AStoHP8Rg==;EndpointSuffix=core.windows.net"
    CONTAINER_NAME = "icpna"
    SOURCE_FOLDER = "project"  # The folder to upload
    
    # Check if source folder exists
    if not os.path.exists(SOURCE_FOLDER):
        print(f"Error: Source folder '{SOURCE_FOLDER}' not found!")
        exit(1)
        
    upload_directory_to_blob(CONNECTION_STRING, CONTAINER_NAME, SOURCE_FOLDER)
