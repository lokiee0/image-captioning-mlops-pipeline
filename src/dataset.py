"""PyTorch Dataset reading versioned data back out of MinIO + PostgreSQL."""
import io
import random
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_BUCKET, POSTGRES_DSN

IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class CaptionDataset(Dataset):
    """
    Reads (image_key, captions) rows from Postgres for a given dataset_name +
    split, fetches the actual JPEG bytes from MinIO on __getitem__, and
    encodes one randomly-sampled caption per image per epoch.
    """

    def __init__(self, dataset_name: str, split: str, vocab, max_len: int = 30, transform=None):
        self.dataset_name = dataset_name
        self.split = split
        self.vocab = vocab
        self.max_len = max_len
        self.transform = transform or IMAGE_TRANSFORM
        self.rows = self._load_rows()

    def _load_rows(self):
        import psycopg2

        conn = psycopg2.connect(POSTGRES_DSN)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT image_key, captions FROM dataset_metadata WHERE dataset_name=%s AND split=%s",
                (self.dataset_name, self.split),
            )
            rows = cur.fetchall()
        conn.close()
        return rows

    def _minio_client(self):
        from minio import Minio

        if not hasattr(self, "_client"):
            self._client = Minio(
                MINIO_ENDPOINT,
                access_key=MINIO_ACCESS_KEY,
                secret_key=MINIO_SECRET_KEY,
                secure=False,
            )
        return self._client

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        image_key, captions = self.rows[idx]
        client = self._minio_client()
        resp = client.get_object(MINIO_BUCKET, image_key)
        img = Image.open(io.BytesIO(resp.read())).convert("RGB")
        resp.close()

        caption = random.choice(captions)
        img_tensor = self.transform(img)
        caption_ids = torch.tensor(self.vocab.encode(caption, self.max_len), dtype=torch.long)
        return img_tensor, caption_ids


def collate_fn(batch):
    images, captions = zip(*batch)
    return torch.stack(images), torch.stack(captions)
