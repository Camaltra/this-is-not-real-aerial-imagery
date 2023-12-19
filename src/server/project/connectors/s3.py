import boto3
from botocore.exceptions import NoCredentialsError


class S3Manager:
    def __init__(self, bucket_name):
        self.s3 = boto3.client("s3")
        self.bucket_name = bucket_name

    def upload_to_aws(self, local_file, bucket, s3_file):
        self.s3.upload_file(local_file, bucket, s3_file)
        print("Upload Successful")
        return True
        # except FileNotFoundError:
        #     print("The file was not found")
        #     return False
        # except NoCredentialsError:
        #     print("Credentials not available")
        #     return False

    def list_files_from_bucket(self, folder_name):
        try:
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
            print(f"Error listing files in the bucket {self.bucket_name}: {str(e)}")
            return []

    def get_presigned_url(self, key: str):
        self.s3 = boto3.client("s3")
        expiration_time = 3600

        presigned_url = self.s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket_name, "Key": key},
            ExpiresIn=expiration_time,
        )

        return presigned_url

    def move_file_within_bucket(self, source_key, destination_key):
        s3 = boto3.client("s3")

        try:
            s3.copy_object(
                Bucket=self.bucket_name,
                CopySource={"Bucket": self.bucket_name, "Key": source_key},
                Key=destination_key,
            )

            s3.delete_object(Bucket=self.bucket_name, Key=source_key)

            return True

        except Exception as e:
            print(f"Error moving file: {str(e)}")
            return False

    def create_s3_folder(self, folder_name):
        try:
            response = self.s3.put_object(Bucket=self.bucket_name, Key=folder_name)

            if response["ResponseMetadata"]["HTTPStatusCode"] == 200:
                print(f"Folder '{folder_name}' created successfully.")
                return True
            else:
                print(f"Error creating folder: {response}")
                return False

        except Exception as e:
            print(f"An error occurred: {e}")
            return False

    def delete_folder(self, folder_name: str):
        self.s3.delete_object(Bucket=self.bucket_name, Key=folder_name)

    def get_object(self, filepath: str):
        file_obj = self.s3.get_object(Bucket=self.bucket_name, Key=filepath)
        return file_obj["Body"].read()
