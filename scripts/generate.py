from __future__ import annotations

import argparse

import torch

from llm_from_scratch.config import GPTConfig
from llm_from_scratch.model import DecoderOnlyTransformer
from llm_from_scratch.tokenizer import CharTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a trained checkpoint")
    parser.add_argument("--ckpt", type=str, default="artifacts/checkpoint.pt")
    parser.add_argument("--prompt", type=str, default="To be")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = torch.load(args.ckpt, map_location=args.device)

    model_cfg = GPTConfig(**payload["model_config"])
    model = DecoderOnlyTransformer(model_cfg)
    model.load_state_dict(payload["model_state_dict"])
    model.to(args.device)
    model.eval()

    tokenizer = CharTokenizer(stoi=payload["stoi"], itos={int(k): v for k, v in payload["itos"].items()})

    prompt_ids = tokenizer.encode(args.prompt)
    x = torch.tensor([prompt_ids], dtype=torch.long, device=args.device)
    out = model.generate(
        x,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    text = tokenizer.decode(out[0].tolist())
    print(text)


if __name__ == "__main__":
    main()
