import boto3

s3 = boto3.client('s3')

def upload_file_to_s3(file_path: str, bucket_name: str, s3_key: str):
    """Uploads a file to an S3 bucket."""
    try:
        s3.upload_file(file_path, bucket_name, s3_key)
        print(f"Successfully uploaded {file_path} to s3://{bucket_name}/{s3_key}")
    except Exception as e:
        print(f"Error uploading file: {e}")

def get_file_from_s3(bucket_name: str, s3_key: str, download_path: str):
    """Downloads a file from an S3 bucket."""
    try:
        s3.download_file(bucket_name, s3_key, download_path)
        print(f"Successfully downloaded s3://{bucket_name}/{s3_key} to {download_path}")
    except Exception as e:
        print(f"Error downloading file: {e}")