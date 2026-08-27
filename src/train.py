"""
Stage 4-5: Train model + track experiment.

python src/train.py --dataset coco --epochs 10
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import TrainConfig, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
from src.vocab import Vocabulary
from src.dataset import CaptionDataset, collate_fn
from src.model import CaptioningModel, count_params_mb


def build_vocab(dataset_name: str, split: str, min_freq: int) -> Vocabulary:
    import psycopg2
    from src.config import POSTGRES_DSN

    conn = psycopg2.connect(POSTGRES_DSN)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT captions FROM dataset_metadata WHERE dataset_name=%s AND split=%s",
            (dataset_name, split),
        )
        rows = cur.fetchall()
    conn.close()

    all_captions = [c for (caps,) in rows for c in caps]
    return Vocabulary(min_freq=min_freq).build(all_captions)


def train(cfg: TrainConfig):
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    device = torch.device(cfg.device)

    print(f"[train] building vocabulary from dataset={cfg.dataset} ...")
    vocab = build_vocab(cfg.dataset, "train", cfg.vocab_min_freq)
    print(f"[train] vocab size: {len(vocab)}")

    train_ds = CaptionDataset(cfg.dataset, "train", vocab, cfg.max_caption_len)
    val_ds = CaptionDataset(cfg.dataset, "val", vocab, cfg.max_caption_len)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    model = CaptioningModel(
        vocab_size=len(vocab),
        embed_dim=cfg.embed_dim,
        decoder_layers=cfg.decoder_layers,
        decoder_heads=cfg.decoder_heads,
        max_len=cfg.max_caption_len,
        backbone=cfg.encoder_backbone,
        freeze_encoder=cfg.freeze_encoder,
    ).to(device)

    size_mb = count_params_mb(model)
    print(f"[train] model size estimate: {size_mb:.1f} MB")
    assert size_mb < 500, "model exceeds the 500MB budget - reduce embed_dim/decoder_layers"

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx["<pad>"])

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "dataset": cfg.dataset,
                "embed_dim": cfg.embed_dim,
                "decoder_layers": cfg.decoder_layers,
                "decoder_heads": cfg.decoder_heads,
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "epochs": cfg.epochs,
                "encoder_backbone": cfg.encoder_backbone,
                "vocab_size": len(vocab),
                "model_size_mb": round(size_mb, 1),
            }
        )

        for epoch in range(cfg.epochs):
            model.train()
            total_loss = 0.0
            for images, captions in train_loader:
                images, captions = images.to(device), captions.to(device)
                decoder_in, decoder_target = captions[:, :-1], captions[:, 1:]

                optimizer.zero_grad()
                logits = model(images, decoder_in)
                loss = criterion(logits.reshape(-1, logits.size(-1)), decoder_target.reshape(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            train_loss = total_loss / max(len(train_loader), 1)
            val_loss = _eval_loss(model, val_loader, criterion, device)
            print(f"[train] epoch {epoch+1}/{cfg.epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

        # save artifacts: model weights + vocab, logged to MLflow for the evaluate stage
        ckpt_path = "model.pt"
        vocab_path = "vocab.json"
        torch.save(model.state_dict(), ckpt_path)
        vocab.save(vocab_path)
        mlflow.log_artifact(ckpt_path)
        mlflow.log_artifact(vocab_path)

        print(f"[train] run_id={run.info.run_id} - logged to MLflow")
        return run.info.run_id


def _eval_loss(model, loader, criterion, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for images, captions in loader:
            images, captions = images.to(device), captions.to(device)
            decoder_in, decoder_target = captions[:, :-1], captions[:, 1:]
            logits = model(images, decoder_in)
            loss = criterion(logits.reshape(-1, logits.size(-1)), decoder_target.reshape(-1))
            total += loss.item()
    return total / max(len(loader), 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="flickr8k", choices=["flickr8k", "coco", "conceptual_captions", "vizwiz"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    cfg = TrainConfig(dataset=args.dataset)
    if args.epochs:
        cfg.epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size

    train(cfg)
