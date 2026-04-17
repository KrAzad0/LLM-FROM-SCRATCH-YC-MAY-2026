from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int
    context_length: int = 256
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    dropout: float = 0.1
    bias: bool = True


@dataclass
class TrainConfig:
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int = 2000
    warmup_steps: int = 200
    grad_clip: float = 1.0
    eval_interval: int = 200
    eval_batches: int = 20
    seed: int = 1337
    device: str = "cuda"
