# dags/sentiment_generation.py
import os
import urllib.request
import zipfile
from typing import Any, Dict

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.operators.s3 import S3CreateBucketOperator
from airflow.providers.amazon.aws.transfers.local_to_s3 import (
    LocalFilesystemToS3Operator,
)
from airflow.utils.dates import days_ago


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

    create_bucket_op = S3CreateBucketOperator(
        task_id="create_bucket_op",
        bucket_name="sentiment-bucket",
        aws_conn_id="aws_default",
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
