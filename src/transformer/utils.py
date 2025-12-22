import torch
import os
from dataclasses import dataclass


def causal_mask(seq_length: int, device: torch.device) -> torch.Tensor:
    """
    Creates a causal (lower triangular) attention mask
    Shape: (1, 1, seq_length, seq_length) = (1, 1, T, T)
    """
    return torch.tril(
        torch.ones(seq_length, seq_length, device=device)
    ).unsqueeze(0).unsqueeze(0)

def load_checkpoint(
    path: str,
    model_class,
    device: torch.device,
):
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    config = checkpoint["config"]

    model = model_class(
        max_seq_length=config.max_seq_length,
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        d_ff=config.d_ff,
        h=config.n_heads,
        use_fixed_positional_embeddings=True,
        dropout=config.dropout,
        N=config.n_layers,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"[✓] Model loaded from {path}")

    return model, checkpoint


def save_checkpoint(
    model,
    optimizer,
    config,
    step: int,
    path: str = "checkpoint.pt"
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "step": step,
    }
    torch.save(checkpoint, path)
    print(f"[✓] Model checkpoint saved to {path}")


def print_model_summary(config, model):
    print("=" * 144)
    print("                                                 Decoder-Only Transformer Model Summary")
    print("=" * 144)

    print("\n[Model Type]")
    print("  • Decoder-only Transformer (GPT-style)")

    print("\n[Vocabulary]")
    print(f"  • Vocabulary size        : {config.vocab_size}")

    print("\n[Sequence Lengths]")
    print(f"  • Training seq_length    : {config.seq_length}")
    print(f"  • Max context length     : {config.max_seq_length}")

    print("\n[Embedding]")
    print(f"  • Embedding dimension    : {config.d_model}")

    print("\n[Transformer Architecture]")
    print(f"  • Number of layers       : {config.n_layers}")
    print(f"  • Number of heads        : {config.n_heads}")
    print(f"  • Head dimension         : {config.d_model // config.n_heads}")
    print(f"  • FeedForward dim (d_ff) : {config.d_ff}")

    print("\n[Regularization]")
    print(f"  • Dropout                : {config.dropout}")

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n[Parameters]")
    print(f"  • Total parameters       : {total_params:,}")
    print(f"  • Trainable parameters   : {trainable_params:,}")

    print("\n[Attention Details]")
    print("  • Attention type         : Masked self-attention (causal)")
    print("  • Positional encoding    : Fixed or Learned (configurable)")

    print("=" * 144)


@dataclass
class TransformerConfig:
    vocab_size: int = 65 # we can get this value by loading the data and get len(chars) | self.chars = sorted(list(set(text)))
    d_model: int = 512
    d_ff: int = 4 * d_model # Original Paper used d_ff as 4 * d_model
    n_layers: int = 6
    n_heads: int = 8
    dropout: float = 0.1
    seq_length = 90
    max_seq_length: int = 256

class DataLoader:
    def __init__(self, dataset_address: str = "tiny_shakespeare.txt", *, seq_length: int):
        text = open(dataset_address, 'r', encoding='utf-8').read()

        self.chars = sorted(list(set(text)))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}

        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        self.seq_length = seq_length  # sequence length

        n = int(0.9 * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def get_batch(self, split='train', batch_size = 64, *, device: torch.device):
        assert split in ("train", "val")
        source = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(source) - self.seq_length - 1, (batch_size,))
        X = torch.stack([source[i:i + self.seq_length] for i in ix])
        Y = torch.stack([source[i+1:i + self.seq_length + 1] for i in ix])
        return X.to(device), Y.to(device)


def top_k_logits(logits: torch.Tensor, k: int):
    k = min(k, logits.size(-1))
    if k <= 0:
        return logits

    values, _ = torch.topk(logits, k)
    min_values = values[:, -1].unsqueeze(-1)
    return torch.where(logits < min_values, torch.full_like(logits, -1e9), logits)

def top_p_logits(logits: torch.Tensor, p: float):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cum_probs = torch.cumsum(probs, dim=-1)

    cutoff = cum_probs > p
    cutoff[..., 1:] = cutoff[..., :-1].clone()
    cutoff[..., 0] = False

    sorted_logits[cutoff] = -1e9
    logits.scatter_(1, sorted_indices, sorted_logits)
    return logits

@torch.no_grad()
def generate(
    model,
    idx,
    max_new_tokens,
    device,
    temperature=1.0,
    top_k=None,
    top_p=None,
):
    model.eval()
    max_T = model.embeddings.pos_embedding.max_seq_length

    for _ in range(max_new_tokens):

        # sliding window
        idx_cond = idx[:, -max_T:]

        T = idx_cond.size(1)
        mask = causal_mask(T, device)

        logits = model(idx_cond, mask)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            logits = top_k_logits(logits, top_k)

        if top_p is not None:
            logits = top_p_logits(logits, top_p)

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, next_token], dim=1)

    return idx


