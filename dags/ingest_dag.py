"""
Stage 1-3: Collect data -> store dataset -> version dataset.

Triggered manually or on a schedule. dataset_name is passed via DAG run
config, e.g.: airflow dags trigger ingest_dag --conf '{"dataset":"coco"}'
"""
from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

default_args = {"owner": "mlops", "retries": 1}

with DAG(
    dag_id="ingest_dag",
    default_args=default_args,
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["mlops", "ingest"],
) as dag:

    def _collect(**context):
        import subprocess

        dataset = context["dag_run"].conf.get("dataset", "flickr8k")
        limit = context["dag_run"].conf.get("limit")
        cmd = ["python", "/opt/airflow/project/data/dataset_loader.py", "--dataset", dataset]
        if limit:
            cmd += ["--limit", str(limit)]
        subprocess.run(cmd, check=True)

    collect_and_store = PythonOperator(
        task_id="collect_and_store_data",
        python_callable=_collect,
    )

    # Stage 3: version the freshly-landed data with DVC, pointed at MinIO as the remote.
    version_dataset = BashOperator(
        task_id="version_dataset_with_dvc",
        bash_command=(
            "cd /opt/airflow/project && "
            "dvc add data/raw/{{ dag_run.conf.get('dataset', 'flickr8k') }} && "
            "git add data/raw/*.dvc .gitignore && "
            "git commit -m 'data: version {{ dag_run.conf.get(\"dataset\", \"flickr8k\") }} ingest run {{ ds }}' && "
            "dvc push"
        ),
    )

    collect_and_store >> version_dataset
