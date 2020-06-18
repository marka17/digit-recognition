from typing import Optional

import torch

from .feature import MelSpectrogram
from ..hyperparams import Hyperparams
from .transforms.feature import FeatureProcessor
from .batch import Batch


class BatchProcessor:

    def __init__(
        self,
        hparams: Hyperparams,
        feature_processor: Optional[FeatureProcessor] = None
    ):
        self.hparams = hparams
        self.feature_processor = feature_processor

        self.feature: MelSpectrogram = MelSpectrogram(hparams)

    def __call__(self, batch: Batch) -> Batch:
        batch.x = self.feature(batch.x)
        batch.x_len = self.sample2frame(batch.x_len)

        if self.feature_processor is not None:
            batch = self.feature_processor(batch)

        return batch

    def to(self, device: torch.device):
        self.feature = self.feature.to(device)

        return self

    def sample2frame(self, x: torch.Tensor) -> torch.Tensor:
        """
        Переводит длины из `sample` размерности в `frame` размерность
        """
        return (x - self.hparams.win_length) // self.hparams.hop_length + 1
