from typing import Optional, List

import logging

import torch

logger = logging.getLogger(__name__)


class Batch:

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_len: Optional[torch.Tensor] = None,
        y_len: Optional[torch.Tensor] = None,
        paths: Optional[List[str]] = None
    ):
        self.x = x
        self.y = y
        self.x_len = x_len
        self.y_len = y_len
        self.paths: List[str] = paths

    def cuda(self, non_blocking=False):
        self.x = self.x.to(device='cuda', non_blocking=non_blocking)
        self.y = self.y.to(device='cuda', non_blocking=non_blocking)
        self.x_len = self.x_len.to(device='cuda', non_blocking=non_blocking)
        self.y_len = self.y_len.to(device='cuda', non_blocking=non_blocking)

        return self

    def cpu(self):
        self.x = self.x.cpu()
        self.y = self.y.cpu()
        self.x_len = self.x_len.cpu()
        self.y_len = self.y_len.cpu()

        return self

    @property
    def size(self) -> int:
        return self.x.shape[0]

    def __repr__(self):
        lines = []
        for attr, value in self.__dict__.items():
            if value is not None:
                lines.append(f"Attr: {attr}:")

                if isinstance(value, torch.Tensor):
                    lines.append("Shape: {}".format(value.shape))
                lines.append(repr(value))

                lines.append("\n")

        return "\n".join(lines)
