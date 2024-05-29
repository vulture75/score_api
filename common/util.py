import os
import pickle

import boto3
from botocore.client import Config as BotoConfig

from common.config import Config


def search_dir(dir_path: str, target_ext=["wav", "mp3"], except_file_name=[]):
    file_list = []

    if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
        raise ValueError("Invalid directory path")

    for root, _, files in os.walk(dir_path):
        for file in files:
            file_ext = file.split(".")[-1]
            if file_ext in target_ext:
                if os.path.splitext(os.path.basename(file))[0] not in except_file_name:
                    file_list.append(os.path.join(root, file))

    return file_list


def search_end_dir(path: str):
    if not os.path.exists(path) or not os.path.isdir(path):
        raise ValueError("Invalid directory path")

    file_list = search_dir(path)
    dir_set = set(map(lambda x: os.path.dirname(x), file_list))

    return list(dir_set)


def create_directory(path):
    try:
        os.makedirs(path)
        print(f"Directory '{path}' created successfully.")
    except FileExistsError:
        print(f"Directory '{path}' already exists.")


class File:
    @staticmethod
    def save(file_obj: object, save_path: str):
        with open(save_path, "wb") as fp:
            fp.write(file_obj.read())

        return save_path

    @staticmethod
    def load(load_path: str):
        with open(load_path, "rb") as fp:
            file_obj = fp.read()

        return file_obj


class Json:
    @staticmethod
    def save(data, save_path):
        import json

        with open(save_path, "w") as outfile:
            json.dump(data, outfile)

    @staticmethod
    def load(load_path):
        import json

        with open(load_path) as json_file:
            data = json.load(json_file)

        return data


class Pickle:
    @staticmethod
    def save_obj(path: str, obj: object, protocol=pickle.HIGHEST_PROTOCOL):
        with open(path, "wb") as handle:
            pickle.dump(obj, handle, protocol=protocol)

    @staticmethod
    def load_obj(path: str):
        with open(path, "rb") as handle:
            res_dic = pickle.load(handle)
        return res_dic


class S3:
    config: BotoConfig = BotoConfig(connect_timeout=5, retries={"max_attempts": 3})

    @staticmethod
    def init_s3_client():
        s3 = boto3.client(
            "s3",
            aws_access_key_id=Config.get_env_value("AWS_S3_ACCESS_KEY"),
            aws_secret_access_key=Config.get_env_value("AWS_S3_SECRET_KEY"),
            region_name="ap-northeast-2",
            config=S3.config,
        )
        return s3

    @staticmethod
    def download_file(s3_client, src_file_url, save_file_path, bucket_name: str = "2dub"):
        print(src_file_url, save_file_path, bucket_name)
        s3_client.download_file(bucket_name, src_file_url, save_file_path)

    @staticmethod
    def get_file_list(s3_client, dir_path, bucket_name):
        my_bucket = s3_client.Bucket(bucket_name)

        file_list = []
        for object_summary in my_bucket.objects.filter(Prefix=dir_path):
            print(object_summary.key)
            file_list.append(object_summary.key)

        return file_list

    @staticmethod
    def upload_file_to_s3(s3_client, src_file_path, bucket_name: str, s3_save_path: str):
        try:
            s3_client.upload_file(
                src_file_path,
                bucket_name,
                s3_save_path,
            )

            return 200
        except Exception as ex:
            print(ex)
            return 500


##### remove for public ###
##class ReportUtil:


if __name__ == "__main__":
    # ReportUtil.verify_report_file(Json.load("/home/user/task_student_187eb61b20f849f8b259d067cc6a7e00.json"), "class")
    pass