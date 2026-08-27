"""
Stage 6-7: Evaluate + register model.

Loads a completed MLflow run's artifacts, computes BLEU-4 against the test
split, and — if the score clears the promotion threshold — registers the
model to the MLflow Model Registry in "Staging".

python src/evaluate.py --run-id <RUN_ID> --dataset coco
"""
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import TrainConfig, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, MLFLOW_REGISTERED_MODEL_NAME
from src.vocab import Vocabulary
from src.dataset import CaptionDataset, collate_fn
from src.model import CaptioningModel


def bleu4(reference_lists: list[list[str]], hypotheses: list[str]) -> float:
    """BLEU-4 via nltk, one reference set per hypothesis."""
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

    refs = [[r.split() for r in refs] for refs in reference_lists]
    hyps = [h.split() for h in hypotheses]
    return corpus_bleu(refs, hyps, smoothing_function=SmoothingFunction().method1)


def evaluate(run_id: str, dataset: str, cfg: TrainConfig):
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    local_dir = client.download_artifacts(run_id, ".")
    vocab = Vocabulary.load(f"{local_dir}/vocab.json", cfg.vocab_min_freq)

    model = CaptioningModel(
        vocab_size=len(vocab),
        embed_dim=cfg.embed_dim,
        decoder_layers=cfg.decoder_layers,
        decoder_heads=cfg.decoder_heads,
        max_len=cfg.max_caption_len,
        backbone=cfg.encoder_backbone,
        freeze_encoder=cfg.freeze_encoder,
    )
    model.load_state_dict(torch.load(f"{local_dir}/model.pt", map_location="cpu"))
    model.eval()

    test_ds = CaptionDataset(dataset, "test", vocab, cfg.max_caption_len)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    references, hypotheses = [], []
    with torch.no_grad():
        for images, captions in test_loader:
            preds = model.generate(images, vocab)
            hypotheses.extend(preds)
            references.extend([[vocab.decode(c.tolist())] for c in captions])

    score = bleu4(references, hypotheses)
    print(f"[evaluate] BLEU-4 on {dataset}/test: {score:.4f}")

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("test_bleu4", score)

    if score >= cfg.bleu4_promotion_threshold:
        print(f"[evaluate] score clears threshold ({cfg.bleu4_promotion_threshold}) - registering model")
        model_uri = f"runs:/{run_id}/model.pt"
        result = mlflow.register_model(model_uri, MLFLOW_REGISTERED_MODEL_NAME)
        client.transition_model_version_stage(
            name=MLFLOW_REGISTERED_MODEL_NAME, version=result.version, stage="Staging"
        )
        print(f"[evaluate] registered as {MLFLOW_REGISTERED_MODEL_NAME} v{result.version} (Staging)")
        return True, score
    else:
        print("[evaluate] score below threshold - not registering")
        return False, score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset", default="flickr8k", choices=["flickr8k", "coco", "conceptual_captions", "vizwiz"])
    args = parser.parse_args()

    evaluate(args.run_id, args.dataset, TrainConfig(dataset=args.dataset))
