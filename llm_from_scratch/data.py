from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TextDataset:
    train_data: torch.Tensor
    val_data: torch.Tensor
    context_length: int

    @classmethod
    def from_ids(cls, ids: list[int], context_length: int, train_ratio: float = 0.9) -> "TextDataset":
        data = torch.tensor(ids, dtype=torch.long)
        split = int(train_ratio * len(data))
        return cls(train_data=data[:split], val_data=data[split:], context_length=context_length)

    def get_batch(self, split: str, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.train_data if split == "train" else self.val_data
        max_start = len(data) - self.context_length - 1
        if max_start <= 0:
            raise ValueError("Dataset too small for selected context length")

        starts = torch.randint(0, max_start, (batch_size,))
        x = torch.stack([data[s : s + self.context_length] for s in starts])
        y = torch.stack([data[s + 1 : s + self.context_length + 1] for s in starts])
        return x.to(device), y.to(device)
