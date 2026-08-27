"""
Lightweight image captioning model.

Encoder: ResNet-18 (ImageNet-pretrained, ~44MB), final FC layer stripped,
         pooled features projected down to embed_dim.
Decoder: small transformer decoder over word embeddings.

Total parameter footprint stays well under 500MB (typically 60-120MB
depending on vocab size), unlike BLIP/ViT-GPT2 style checkpoints (~1GB).
"""
import math
import torch
import torch.nn as nn
from torchvision import models


class EncoderCNN(nn.Module):
    def __init__(self, embed_dim: int, backbone: str = "resnet18", freeze: bool = True):
        super().__init__()
        if backbone == "resnet18":
            base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            feat_dim = base.fc.in_features
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.backbone = nn.Sequential(*list(base.children())[:-2])  # drop avgpool + fc -> spatial feature map
        self.project = nn.Linear(feat_dim, embed_dim)

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, images):
        feats = self.backbone(images)  # (B, C, H, W)
        b, c, h, w = feats.shape
        feats = feats.view(b, c, h * w).permute(0, 2, 1)  # (B, H*W, C)
        return self.project(feats)  # (B, H*W, embed_dim) -- used as memory for decoder


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class DecoderTransformer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 256, num_layers: int = 2, num_heads: int = 4, max_len: int = 30):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_enc = PositionalEncoding(embed_dim, max_len)
        layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.max_len = max_len

    def forward(self, memory, captions):
        """captions: (B, T) token ids, teacher-forced input (all but last token)."""
        tgt = self.embed(captions)
        tgt = self.pos_enc(tgt)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        out = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        return self.fc_out(out)


class CaptioningModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 256, decoder_layers: int = 2,
                 decoder_heads: int = 4, max_len: int = 30, backbone: str = "resnet18", freeze_encoder: bool = True):
        super().__init__()
        self.encoder = EncoderCNN(embed_dim, backbone, freeze_encoder)
        self.decoder = DecoderTransformer(vocab_size, embed_dim, decoder_layers, decoder_heads, max_len)
        self.max_len = max_len

    def forward(self, images, captions_in):
        memory = self.encoder(images)
        return self.decoder(memory, captions_in)

    @torch.no_grad()
    def generate(self, images, vocab, max_len: int = None):
        """Greedy decoding, one caption per image."""
        self.eval()
        max_len = max_len or self.max_len
        memory = self.encoder(images)
        b = images.size(0)
        device = images.device

        start_id = vocab.word2idx["<start>"]
        end_id = vocab.word2idx["<end>"]
        tokens = torch.full((b, 1), start_id, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            logits = self.decoder(memory, tokens)
            next_token = logits[:, -1, :].argmax(-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)
            if (next_token.squeeze(-1) == end_id).all():
                break

        return [vocab.decode(tokens[i].tolist()) for i in range(b)]


def count_params_mb(model: nn.Module) -> float:
    """Rough on-disk size estimate (float32 params)."""
    n_params = sum(p.numel() for p in model.parameters())
    return n_params * 4 / (1024 ** 2)
