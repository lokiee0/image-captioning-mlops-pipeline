"""
Stage 8 (monitor -> retrain loop): scheduled drift/quality check that
triggers train_dag when the production model's inputs or outputs have
drifted past the configured threshold.
"""
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

default_args = {"owner": "mlops", "retries": 1}

with DAG(
    dag_id="retrain_trigger_dag",
    default_args=default_args,
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["mlops", "monitor"],
) as dag:

    def _check_drift(**context):
        import sys

        sys.path.append("/opt/airflow/project")
        from monitoring.evidently_check import run_drift_check
        from src.config import DRIFT_SHARE_THRESHOLD

        drift_share = run_drift_check()
        print(f"[monitor] drifted column share: {drift_share:.3f} (threshold={DRIFT_SHARE_THRESHOLD})")
        context["ti"].xcom_push(key="drift_share", value=drift_share)
        return drift_share > DRIFT_SHARE_THRESHOLD

    check_drift = PythonOperator(task_id="check_drift", python_callable=_check_drift)

    trigger_retrain = TriggerDagRunOperator(
        task_id="trigger_retrain",
        trigger_dag_id="train_dag",
        conf={"dataset": "{{ dag_run.conf.get('dataset', 'flickr8k') }}"},
    )

    check_drift >> trigger_retrain
