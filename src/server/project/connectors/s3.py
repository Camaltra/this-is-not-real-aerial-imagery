import boto3
from botocore.exceptions import NoCredentialsError
import logging

logging.basicConfig(level=logging.INFO)


class S3Manager:
    def __init__(self, bucket_name):
        self.s3 = boto3.client("s3")
        self.logger = logging.getLogger(__name__)
        self.bucket_name = bucket_name

    def upload_to_aws(self, local_file: str, s3_file_name: str) -> None:
        try:
            self.s3.upload_file(local_file, self.bucket_name, s3_file_name)
            self.logger.info(f"Upload Successful: {s3_file_name}")
        except FileNotFoundError:
            print("The file was not found")
            self.logger.error(f"File not found: {local_file}")
        except NoCredentialsError:
            print("Credentials not available")
            self.logger.error(f"Credentials not available")

    def list_files_from_bucket(self, folder_name: str) -> list[dict[str, str]]:
        try:
            self.logger.info(f"Listing files in bucket {self.bucket_name}")
            response = self.s3.list_objects_v2(
                Bucket=self.bucket_name, Prefix=folder_name
            )
            files = []
            for obj in response.get("Contents", []):
                file_info = {
                    "Key": obj["Key"],
                    "Size": obj["Size"],
                    "LastModified": obj["LastModified"].isoformat(),
                    "ETag": obj["ETag"],
                }
                files.append(file_info)
            return files

        except Exception as e:
            self.logger.error(
                f"Error listing files in the bucket %s{self.bucket_name}: %s{str(e)}"
            )
            return []

    def build_presigned_url(self, key: str, expiration_time: int = 3600) -> str:
        self.s3 = boto3.client("s3")
        self.logger.info(f"Generating presigned url for {key}")

        presigned_url = self.s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket_name, "Key": key},
            ExpiresIn=expiration_time,
        )

        return presigned_url

    def move_file_within_bucket(self, source_key: str, destination_key: str) -> None:
        s3 = boto3.client("s3")
        try:
            s3.copy_object(
                Bucket=self.bucket_name,
                CopySource={"Bucket": self.bucket_name, "Key": source_key},
                Key=destination_key,
            )

            s3.delete_object(Bucket=self.bucket_name, Key=source_key)
            self.logger.info(f"File moved from {source_key} to {destination_key}")
        except Exception as e:
            self.logger.error(f"Error moving file: {str(e)}")

    def create_s3_folder(self, folder_name: str) -> None:
        try:
            response = self.s3.put_object(Bucket=self.bucket_name, Key=folder_name)

            if response["ResponseMetadata"]["HTTPStatusCode"] == 200:
                self.logger.info(f"Folder '{folder_name}' created successfully.")
            else:
                self.logger.error(f"Error creating folder: {response}")

        except Exception as e:
            self.logger.error(f"Error creating folder: {str(e)}")

    def delete_folder(self, folder_name: str) -> None:
        self.logger.info(f"Deleting folder {folder_name}")
        try:
            self.s3.delete_object(Bucket=self.bucket_name, Key=folder_name)
            self.logger.info(f"Folder {folder_name} deleted successfully.")
        except Exception as e:
            self.logger.error(f"Error deleting folder: {str(e)}")

    def get_object(self, filepath: str) -> bytes | None:
        try:
            file_obj = self.s3.get_object(Bucket=self.bucket_name, Key=filepath)
            return file_obj["Body"].read()
        except Exception as e:
            self.logger.error(f"Error getting object: {str(e)}")
            return None
