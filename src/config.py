"""
Central configuration for the image-captioning MLOps pipeline.
Change DATASET_NAME (or pass --dataset on the CLI) to switch datasets
without touching any other stage of the pipeline.
"""
import os
from dataclasses import dataclass, field

# ---- Supported datasets -----------------------------------------------
# Each entry maps a short key -> HF hub id + the column names that hold
# the image and the captions in that dataset. dataset_loader.py normalizes
# all of these into a single common schema: {"image": PIL.Image, "captions": [str, ...]}
DATASETS = {
    "flickr8k": {
        "hf_id": "intro/flickr8k",
        "image_col": "image",
        "caption_cols": ["caption_0", "caption_1", "caption_2", "caption_3", "caption_4"],
        "splits": {"train": "train", "val": "dev", "test": "test"},
    },
    "coco": {
        "hf_id": "HuggingFaceM4/COCO",
        "image_col": "image",
        "caption_cols": ["sentences_raw"],  # list column, handled specially
        "splits": {"train": "train", "val": "validation", "test": "test"},
    },
    "conceptual_captions": {
        "hf_id": "google-research-datasets/conceptual_captions",
        "image_col": "image_url",  # this dataset ships URLs, not embedded images
        "caption_cols": ["caption"],
        "splits": {"train": "train", "val": "validation", "test": "validation"},
    },
    "vizwiz": {
        "hf_id": "lmms-lab/VizWiz-Caption",
        "image_col": "image",
        "caption_cols": ["captions"],  # list column
        "splits": {"train": "train", "val": "val", "test": "test"},
    },
}

DATASET_NAME = os.environ.get("MLOPS_DATASET", "flickr8k")

# ---- Storage (MinIO/S3) ------------------------------------------------
# MinIO configuration is loaded from environment variables (.env file)
# These defaults are for development/testing only
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "")  # No default - must be configured
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "")
MINIO_BUCKET = os.environ.get("MINIO_BUCKET", "image-captioning")
MINIO_SECURE = os.environ.get("MINIO_SECURE", "true").lower() == "true"

# ---- Database (for experiments/model tracking) ---------------------------
# Note: NOT used for dataset storage; datasets are stored in MinIO only.
POSTGRES_DSN = os.environ.get(
    "POSTGRES_DSN", "postgresql://mlops:mlops@localhost:5432/mlops"
)

# ---- Experiment tracking --------------------------------------------
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "image-captioning")
MLFLOW_REGISTERED_MODEL_NAME = os.environ.get(
    "MLFLOW_REGISTERED_MODEL_NAME", "image-captioner"
)

# ---- Model / training hyperparameters --------------------------------
@dataclass
class TrainConfig:
    dataset: str = DATASET_NAME
    embed_dim: int = 256
    decoder_layers: int = 2
    decoder_heads: int = 4
    max_caption_len: int = 30
    vocab_min_freq: int = 3
    batch_size: int = 32
    epochs: int = 10
    lr: float = 3e-4
    encoder_backbone: str = "resnet18"  # keeps total model well under 500MB
    freeze_encoder: bool = True
    device: str = os.environ.get("DEVICE", "cpu")

    # promotion gate used by evaluate.py before pushing to MLflow registry
    bleu4_promotion_threshold: float = 0.15


# ---- Monitoring / retraining -------------------------------------------
DRIFT_SHARE_THRESHOLD = float(os.environ.get("DRIFT_SHARE_THRESHOLD", "0.3"))
EVAL_BLEU4_ALERT_THRESHOLD = float(os.environ.get("EVAL_BLEU4_ALERT_THRESHOLD", "0.12"))
