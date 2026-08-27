"""
Stage 9: Deploy as API.

Loads the current "Production" stage model straight from the MLflow Model
Registry at startup, exposes /caption for inference, and /health for
Kubernetes liveness/readiness probes. Also logs request-level features
that monitoring/evidently_check.py consumes for drift detection.

Run locally:  uvicorn serving.app:app --host 0.0.0.0 --port 8000
"""
import io
import sys
import time
from pathlib import Path

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from prometheus_client import Counter, Histogram, make_asgi_app

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_REGISTERED_MODEL_NAME,
    TrainConfig,
)
from src.dataset import IMAGE_TRANSFORM
from src.model import CaptioningModel
from src.vocab import Vocabulary

app = FastAPI(title="Image Captioning API")

# Prometheus metrics, scraped by prometheus.yml -> visualized in Grafana
REQUEST_COUNT = Counter("caption_requests_total", "Total captioning requests")
REQUEST_LATENCY = Histogram("caption_request_latency_seconds", "Request latency")
app.mount("/metrics", make_asgi_app())

_model = None
_vocab = None
_cfg = TrainConfig()


def _load_production_model():
    """Pull the current Production-stage model + vocab from the MLflow registry."""
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    versions = client.get_latest_versions(MLFLOW_REGISTERED_MODEL_NAME, stages=["Production"])
    if not versions:
        raise RuntimeError(f"No Production version found for '{MLFLOW_REGISTERED_MODEL_NAME}'")
    mv = versions[0]

    local_dir = client.download_artifacts(mv.run_id, ".")
    vocab = Vocabulary.load(f"{local_dir}/vocab.json", _cfg.vocab_min_freq)

    model = CaptioningModel(
        vocab_size=len(vocab),
        embed_dim=_cfg.embed_dim,
        decoder_layers=_cfg.decoder_layers,
        decoder_heads=_cfg.decoder_heads,
        max_len=_cfg.max_caption_len,
        backbone=_cfg.encoder_backbone,
        freeze_encoder=_cfg.freeze_encoder,
    )
    model.load_state_dict(torch.load(f"{local_dir}/model.pt", map_location="cpu"))
    model.eval()
    return model, vocab


@app.on_event("startup")
def startup():
    global _model, _vocab
    _model, _vocab = _load_production_model()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/caption")
async def caption(file: UploadFile = File(...)):
    if _model is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    REQUEST_COUNT.inc()
    start = time.time()

    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tensor = IMAGE_TRANSFORM(img).unsqueeze(0)
        caption_text = _model.generate(tensor, _vocab)[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"could not process image: {e}")
    finally:
        REQUEST_LATENCY.observe(time.time() - start)

    return {"caption": caption_text}
