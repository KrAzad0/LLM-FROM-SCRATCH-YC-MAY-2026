# LLM-FROM-SCRATCH-YC-MAY-2026

A complete, minimal decoder-only language model implementation in pure PyTorch.

## What this includes

- Character-level tokenizer
- Causal self-attention transformer (GPT-style)
- Training loop with:
  - AdamW
  - gradient clipping
  - warmup + cosine decay schedule
  - train/validation loss estimation
- Text generation script with temperature + top-k sampling
- Checkpoint save/load

## Project structure

```text
llm_from_scratch/
  config.py
  data.py
  model.py
  tokenizer.py
scripts/
  train.py
  generate.py
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch
```

## Prepare data

Put raw text into `data/input.txt`.

Example:

```bash
mkdir -p data
curl -L https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o data/input.txt
```

## Train

```bash
python scripts/train.py \
  --text data/input.txt \
  --out artifacts \
  --max-steps 2000 \
  --batch-size 32 \
  --context-length 256 \
  --d-model 256 \
  --layers 6 \
  --heads 8
```

The script will emit periodic train/validation loss and save:

- `artifacts/checkpoint.pt`
- `artifacts/metadata.json`

## Generate text

```bash
python scripts/generate.py \
  --ckpt artifacts/checkpoint.pt \
  --prompt "ROMEO:" \
  --max-new-tokens 300 \
  --temperature 0.8 \
  --top-k 40
```

## Notes

- This is intentionally transparent and compact, meant for learning.
- For production-grade LLMs you would add:
  - subword tokenization (BPE/SentencePiece)
  - mixed precision
  - fused/flash attention kernels
  - distributed training (DDP/FSDP/ZeRO)
  - checkpoint sharding and robust eval suites
