from typing import List
import random

import torch
import torchaudio

from ..batch import Batch


class FeatureProcessor:

    def __call__(self, batch: Batch) -> Batch:
        return batch

    def cuda(self):
        return self


class ChainFeatureProcessor(FeatureProcessor):

    def __init__(self, chain: List[FeatureProcessor]):
        self.chain: List[FeatureProcessor] = chain

    def __call__(self, batch: Batch) -> Batch:
        for processor in self.chain:
            batch: Batch = processor(batch)

        return batch

    def cuda(self):
        for i, processor in enumerate(self.chain):
            self.chain[i] = processor.cuda()

        return self


class AmplitudeToDBProcessor(FeatureProcessor):

    def __init__(self):
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def __call__(self, batch: Batch) -> Batch:
        batch.x = self.amplitude_to_db(batch.x)
        return batch


class StackingMelProcessor(FeatureProcessor):
    """
    https://arxiv.org/pdf/1507.06947.pdf
    """

    def __init__(self, stacking_size: int = 3, stacking_step: int = 2):
        self.stacking_size: int = stacking_size
        self.stacking_step: int = stacking_step

    def __call__(self, batch: Batch) -> Batch:
        time_dim: int = 2
        feature_dim: int = 1

        batch.x = batch.x.unfold(time_dim, self.stacking_size, self.stacking_step) \
            .permute(0, 3, 1, 2) \
            .flatten(1, 2)

        return batch


class MelAugmenter(FeatureProcessor):
    """
    Augmenter for mel spectrogram
    Based on paper:
    https://arxiv.org/pdf/1904.08779.pdf
    """

    def __init__(
        self,
        prob: float = 0.1,
        max_time_len: float = 0.2,
        max_freq_count: int = 0.2,
        max_count: int = 2
    ):
        """
        :param prob: augmentation
        :param max_time_len: maximum block size for masking time
        :param max_freq_count: maximum block size for masking frequency
        :param max_count: max allowed count of masking
        """

        self.prob = prob
        self.T = max_time_len
        self.F = max_freq_count
        self.max_count = max_count

    def __call__(self, batch: Batch) -> Batch:
        """
        Make augmentation by masking freq and time
        """

        for i in range(batch.x.shape[0]):
            if random.random() < self.prob:
                batch.x[i] = self.time_mask(batch.x[i])
            if random.random() < self.prob:
                batch.x[i] = self.freq_mask(batch.x[i])

        return batch

    def time_mask(self, mel: torch.Tensor) -> torch.Tensor:
        cnt = random.randint(0, self.max_count)
        T = self.T * mel.shape[0]
        mel_mean = mel.mean().item()

        for _ in range(cnt):
            t = int(random.random() * T)
            t0 = int(random.random() * (mel.shape[0] - t))

            if t0 > 0:
                mel[t0:t0 + t] = mel_mean

        return mel

    def freq_mask(self, mel: torch.Tensor) -> torch.Tensor:
        cnt = random.randint(0, self.max_count)
        F = self.F * mel.shape[1]
        mel_mean = mel.mean().item()

        for _ in range(cnt):
            f = int(random.random() * F)
            f0 = int(random.random() * (mel.shape[1] - f))
            if f0 > 0:
                mel[:, f0:f0 + f] = mel_mean

        return mel
