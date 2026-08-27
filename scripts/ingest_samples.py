"""
Quickstart: pull a small, fixed-size sample (default 100 images) from each
of the three supported datasets, so you can exercise the whole pipeline
end-to-end without waiting on a full download.

    python scripts/ingest_samples.py --n 100
    python scripts/ingest_samples.py --n 100 --datasets coco vizwiz   # subset

Each dataset gets its own DVC-tracked path (data/raw/<dataset>) and its own
row in dataset_metadata, so training/evaluating against one of them just
means passing --dataset <name> downstream - nothing else changes.
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import DATASETS

ALL_DATASETS = [d for d in DATASETS if d != "flickr8k"]  # flickr8k kept for local-only smoke tests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100, help="total images per dataset (80/10/10 train/val/test)")
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS, choices=list(DATASETS))
    parser.add_argument("--no-push", action="store_true", help="dry run, skip MinIO/Postgres writes")
    args = parser.parse_args()

    for name in args.datasets:
        print(f"\n=== ingesting {args.n} images from '{name}' ===")
        cmd = [
            sys.executable, str(Path(__file__).resolve().parents[1] / "data" / "dataset_loader.py"),
            "--dataset", name,
            "--total-limit", str(args.n),
        ]
        if args.no_push:
            cmd.append("--no-push")
        subprocess.run(cmd, check=True)

    print(f"\n[done] ingested {args.n} images each from: {', '.join(args.datasets)}")
    print("Next: dvc add data/raw/<dataset> && dvc push, for each dataset you want versioned.")


if __name__ == "__main__":
    main()
