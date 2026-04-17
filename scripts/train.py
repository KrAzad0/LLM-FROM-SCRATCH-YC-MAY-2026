from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import torch

from llm_from_scratch.config import GPTConfig, TrainConfig
from llm_from_scratch.data import TextDataset
from llm_from_scratch.model import DecoderOnlyTransformer
from llm_from_scratch.tokenizer import CharTokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cosine_lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps

    progress = (step - cfg.warmup_steps) / max(1, (cfg.max_steps - cfg.warmup_steps))
    min_lr = cfg.learning_rate * 0.1
    return min_lr + 0.5 * (cfg.learning_rate - min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(
    model: DecoderOnlyTransformer,
    dataset: TextDataset,
    train_cfg: TrainConfig,
    device: str,
) -> dict[str, float]:
    model.eval()
    losses: dict[str, float] = {}

    for split in ["train", "val"]:
        split_losses = []
        for _ in range(train_cfg.eval_batches):
            xb, yb = dataset.get_batch(split=split, batch_size=train_cfg.batch_size, device=device)
            _, loss = model(xb, yb)
            assert loss is not None
            split_losses.append(loss.item())
        losses[split] = sum(split_losses) / len(split_losses)

    model.train()
    return losses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny decoder-only LLM from scratch")
    parser.add_argument("--text", type=str, default="data/input.txt", help="Path to text corpus")
    parser.add_argument("--out", type=str, default="artifacts", help="Output directory")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_cfg = TrainConfig(
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        device=args.device,
    )
    set_seed(train_cfg.seed)

    text_path = Path(args.text)
    text = text_path.read_text(encoding="utf-8")

    tokenizer = CharTokenizer.from_text(text)
    ids = tokenizer.encode(text)

    model_cfg = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=args.heads,
        dropout=args.dropout,
    )

    dataset = TextDataset.from_ids(ids, context_length=model_cfg.context_length)
    model = DecoderOnlyTransformer(model_cfg).to(train_cfg.device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay
    )

    os.makedirs(args.out, exist_ok=True)

    for step in range(train_cfg.max_steps):
        lr = cosine_lr(step, train_cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        xb, yb = dataset.get_batch("train", train_cfg.batch_size, train_cfg.device)
        _, loss = model(xb, yb)
        assert loss is not None

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()

        if step % train_cfg.eval_interval == 0 or step == train_cfg.max_steps - 1:
            losses = estimate_loss(model, dataset, train_cfg, train_cfg.device)
            print(
                f"step={step:04d} train_loss={losses['train']:.4f} "
                f"val_loss={losses['val']:.4f} lr={lr:.6f}"
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_cfg.__dict__,
            "train_config": train_cfg.__dict__,
            "stoi": tokenizer.stoi,
            "itos": tokenizer.itos,
        },
        Path(args.out) / "checkpoint.pt",
    )

    metadata = {
        "vocab_size": tokenizer.vocab_size,
        "train_tokens": len(dataset.train_data),
        "val_tokens": len(dataset.val_data),
        "context_length": model_cfg.context_length,
    }
    (Path(args.out) / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved model to {Path(args.out) / 'checkpoint.pt'}")


if __name__ == "__main__":
    main()
