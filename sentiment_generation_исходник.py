# dags/sentiment_generation.py
import os
from typing import Any, Dict
import boto3
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.transfers.local_to_s3 import (
    LocalFilesystemToS3Operator,
)
from airflow.utils.dates import days_ago
import zipfile
import urllib.request


def download_sentiment_dataset(dataset_path: str) -> str:
    try:
        url = "https://github.com/masha-berd/MLOps_course/raw/main/HW3/data/sentiment_data.zip"
        zip_path = os.path.join(dataset_path, "sentiment_data.zip")

        urllib.request.urlretrieve(url, zip_path)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(dataset_path)

        return os.path.join(dataset_path, "sentiment_data")
    except Exception as e:
        raise Exception(f"Error downloading dataset: {e}")


def create_s3_bucket(bucket_name: str) -> str:
    try:
        s3_client = boto3.client("s3")
        existing_buckets = s3_client.list_buckets()["Buckets"]
        bucket_names = [bucket["Name"] for bucket in existing_buckets]
        if bucket_name not in bucket_names:
            s3_client.create_bucket(Bucket=bucket_name)
        return bucket_name
    except Exception as e:
        raise Exception(f"Error creating S3 bucket: {e}")


def prepare_sentiment_data():
    import shutil

    dataset_path = "/tmp/sentiment_data"
    processed_path = "/tmp/processed_sentiment"

    os.makedirs(processed_path, exist_ok=True)

    for file in ["train.txt", "dev.txt", "test.txt"]:
        shutil.copy(
            os.path.join(dataset_path, file), os.path.join(processed_path, file)
        )


default_args: Dict[str, Any] = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": days_ago(1),
    "retries": 3,
}

with DAG(
    "sentiment_generation",
    default_args=default_args,
    description="",
    schedule_interval=None,
    catchup=False,
) as dag:
    download_data_op = PythonOperator(
        task_id="download_data_op",
        python_callable=lambda: download_sentiment_dataset("/tmp"),
    )

    create_bucket_op = PythonOperator(
        task_id="create_bucket_op",
        python_callable=lambda: create_s3_bucket("sentiment-bucket"),
    )

    prepare_data_op = PythonOperator(
        task_id="prepare_data_op",
        python_callable=prepare_sentiment_data,
    )

    upload_train_data_op = LocalFilesystemToS3Operator(
        task_id="upload_train_data_op",
        filename="/tmp/processed_sentiment/train.txt",
        dest_key="data/train.txt",
        dest_bucket="sentiment-bucket",
        replace=True,
        aws_conn_id="aws_default",
    )

    upload_test_data_op = LocalFilesystemToS3Operator(
        task_id="upload_test_data_op",
        filename="/tmp/processed_sentiment/test.txt",
        dest_key="data/test.txt",
        dest_bucket="sentiment-bucket",
        replace=True,
        aws_conn_id="aws_default",
    )

    upload_val_data_op = LocalFilesystemToS3Operator(
        task_id="upload_test_data_op",
        filename="/tmp/processed_sentiment/dev.txt",
        dest_key="data/dev.txt",
        dest_bucket="sentiment-bucket",
        replace=True,
        aws_conn_id="aws_default",
    )

    (
        download_data_op
        >> create_bucket_op
        >> prepare_data_op
        >> [upload_train_data_op, upload_test_data_op, upload_val_data_op]
    )
