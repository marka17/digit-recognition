from typing import Dict, Any, List

import random
import numpy as np

import torch
from torch.distributions import Distribution, Uniform

from .noise import NoiseDataset


class AudioProcessor:

    def __call__(self, instance: Dict[str, Any]) -> Dict[str, Any]:
        return instance


class ChainAudioProcessor(AudioProcessor):

    def __init__(self, chain: List[AudioProcessor]):
        self.chain: List[AudioProcessor] = chain

    def __call__(self, instance: Dict[str, Any]) -> Dict[str, Any]:
        for processor in self.chain:
            instance: Dict[str, Any] = processor(instance)

        return instance


class NoiseInjector(AudioProcessor):

    def __init__(
        self,
        noise_dataset: NoiseDataset,
        sir_sampler: Distribution = Uniform(0, 40),
        rolling: bool = True,
    ):
        self.noise_dataset = noise_dataset
        self.sir_sampler = sir_sampler
        self.rolling = rolling

    def __call__(self, instance: Dict[str, Any]) -> Dict[str, Any]:
        samples: torch.Tensor = instance['samples']

        noise: torch.Tensor = self.noise_dataset.random_noise()

        if self.rolling:
            noise = self._roll1d(noise, random.randint(-noise.shape[0], noise.shape[0]))

        repeat_num: int = samples.shape[0] // noise.shape[0]
        if repeat_num == 0:
            skip_num: int = noise.shape[0] - samples.shape[0] % noise.shape[0]
            left_skip: int = random.randint(0, skip_num - 1)
            noise = noise[left_skip:left_skip + samples.shape[0]]
        else:
            add_num: int = samples.shape[0] % noise.shape[0]
            left_add: int = int(random.randint(0, np.clip(add_num - 1, 0, None)))
            right_add = add_num - left_add
            noise = torch.cat([noise[noise.shape[0] - left_add:]] + [noise] * repeat_num + [noise[:right_add]])

        noise_level = self.sir_sampler.sample()

        noise_energy = torch.norm(noise)
        samples_energy = torch.norm(samples)
        if noise_energy == 0:
            noise_energy = samples_energy

        alpha = (samples_energy / noise_energy) * torch.pow(10, -noise_level / 20)

        corrupted_samples: torch.Tensor = samples + alpha * noise
        instance['samples'] = corrupted_samples

        return instance

    def _roll1d(self, x: torch.Tensor, shift: int) -> torch.Tensor:
        shift %= x.shape[0]

        if shift == 0:
            return x

        rolled_x = torch.zeros_like(x)
        rolled_x[shift:] = x[:-shift]
        rolled_x[:shift] = x[-shift:]

        return rolled_x


class AudioLevelAdjuster(AudioProcessor):

    def __init__(self, db_level_sampler: Distribution = Uniform(-3, 0)):
        self.db_level_sampler = db_level_sampler

    def __call__(self, instance: Dict[str, Any]) -> Dict[str, Any]:
        samples: torch.Tensor = instance['samples']
        db_level = self.db_level_sampler.sample().item()
        sig_peak = torch.max(torch.abs(samples))

        if sig_peak != 0:
            alpha = 10 ** (db_level / 20)
            instance['samples'] = alpha * samples * (1 / sig_peak)

        return instance
