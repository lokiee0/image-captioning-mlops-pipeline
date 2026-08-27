# Image captioning MLOps pipeline

End-to-end pipeline for a lightweight (<500MB) image-to-text model, supporting
three interchangeable datasets: `coco`, `conceptual_captions`, `vizwiz`
(plus `flickr8k` for quick local smoke tests).

## Architecture

```
Collect data (Airflow) -> Store (MinIO + PostgreSQL) -> Version (DVC)
   -> Train (PyTorch)  -> Track (MLflow)   -> Evaluate (BLEU-4)
   -> Register (MLflow Model Registry)     -> Dockerize
   -> Deploy (FastAPI on Kubernetes)       -> Monitor (Prometheus + Grafana + Evidently AI)
   -> drift/quality signal triggers a new Airflow-orchestrated training run
```

Model: ResNet-18 CNN encoder (ImageNet-pretrained, frozen) + a small
transformer decoder. Total size ~60-120MB depending on vocab size — well
under the 500MB budget (vs. ~900MB-1GB for BLIP/ViT-GPT2 style models).

## 1. Bring up local infra

```bash
docker-compose up -d minio postgres mlflow prometheus grafana
```

- MinIO console: http://localhost:9001 (minioadmin / minioadmin)
- MLflow UI: http://localhost:5000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

## 2. Collect + store + version a dataset

```bash
pip install -r requirements.txt --break-system-packages
python data/dataset_loader.py --dataset coco --limit 5000     # swap: coco | conceptual_captions | vizwiz | flickr8k
dvc add data/raw/coco && dvc push
```

Swapping datasets is a one-flag change — nothing downstream needs editing.

## 3. Train + track

```bash
python src/train.py --dataset coco --epochs 10
```

Logs params/metrics/artifacts (`model.pt`, `vocab.json`) to MLflow.

## 4. Evaluate + register

```bash
python src/evaluate.py --run-id <RUN_ID_FROM_STEP_3> --dataset coco
```

Computes BLEU-4 on the test split; if it clears
`TrainConfig.bleu4_promotion_threshold` (default 0.15), registers the model
to the MLflow Model Registry in `Staging`. Promote to `Production` manually
or via CI once you're happy with it:

```python
client.transition_model_version_stage("image-captioner", version=<N>, stage="Production")
```

## 5. Dockerize + deploy

```bash
docker build -f serving/Dockerfile -t captioning-api:latest .
docker run -p 8000:8000 -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 captioning-api:latest

# or on Kubernetes
kubectl apply -f k8s/deployment.yaml
```

Test it:

```bash
curl -X POST -F "file=@some_photo.jpg" http://localhost:8000/caption
```

## 6. Orchestrate with Airflow

```bash
export AIRFLOW_HOME=~/airflow
airflow db init
cp dags/*.py $AIRFLOW_HOME/dags/
airflow standalone
```

- `ingest_dag` — collect, store, DVC-version a dataset (`{"dataset": "coco"}` as run config)
- `train_dag` — train, track, evaluate, register
- `retrain_trigger_dag` — runs daily, checks Evidently AI drift, auto-triggers `train_dag` if drift exceeds `DRIFT_SHARE_THRESHOLD`

## 7. Monitor

- Prometheus scrapes `/metrics` from the serving API (request count, latency)
- Grafana dashboards on top of Prometheus
- `monitoring/evidently_check.py` compares production request features
  against the training baseline; wire its data sources to your real request
  log table before relying on it in production (currently uses placeholder
  frames so the script runs standalone)

## Switching datasets

Everything is driven by `src/config.py::DATASETS`. To add a 4th dataset,
add an entry there with its HF hub id, image/caption column names, and
split names — no other file needs to change.
