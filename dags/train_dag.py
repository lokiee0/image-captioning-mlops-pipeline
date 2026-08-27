"""
Stage 4-7: Train model -> track experiment -> evaluate -> register model.
"""
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowSkipException

default_args = {"owner": "mlops", "retries": 1}

with DAG(
    dag_id="train_dag",
    default_args=default_args,
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["mlops", "train"],
) as dag:

    def _train(**context):
        import sys

        sys.path.append("/opt/airflow/project")
        from src.train import train
        from src.config import TrainConfig

        dataset = context["dag_run"].conf.get("dataset", "flickr8k")
        cfg = TrainConfig(dataset=dataset)
        run_id = train(cfg)
        context["ti"].xcom_push(key="run_id", value=run_id)
        context["ti"].xcom_push(key="dataset", value=dataset)

    def _evaluate_and_register(**context):
        import sys

        sys.path.append("/opt/airflow/project")
        from src.evaluate import evaluate
        from src.config import TrainConfig

        ti = context["ti"]
        run_id = ti.xcom_pull(task_ids="train_model", key="run_id")
        dataset = ti.xcom_pull(task_ids="train_model", key="dataset")

        promoted, score = evaluate(run_id, dataset, TrainConfig(dataset=dataset))
        if not promoted:
            raise AirflowSkipException(f"BLEU-4={score:.4f} below promotion threshold, model not registered")

    train_model = PythonOperator(task_id="train_model", python_callable=_train)
    evaluate_and_register = PythonOperator(
        task_id="evaluate_and_register_model", python_callable=_evaluate_and_register
    )

    train_model >> evaluate_and_register
