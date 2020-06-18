import math

import torch
from torch import nn
from torchaudio import transforms
import torch.nn.functional as F

from ..hyperparams import Hyperparams


class Spectrogram(transforms.Spectrogram):

    def __init__(
        self,
        n_fft: int,
        win_length: int,
        hop_length: int,
        pad=0,
        window_fn=torch.hamming_window,
        power=2.,
        normalized=False,
        wkwargs=None
    ) -> None:
        super().__init__(n_fft, win_length, hop_length,
                         pad, window_fn, power, normalized, wkwargs)

        self.pad_mode = 'constant'
        self.onesided = True

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform_dim = waveform.dim()
        extended_shape = [1] * (3 - waveform_dim) + list(waveform.size())
        pad = (self.n_fft - self.win_length) / 2
        waveform = F.pad(waveform.view(extended_shape), (math.floor(pad), math.ceil(pad)), self.pad_mode)
        waveform = waveform.view(waveform.shape[-waveform_dim:])

        waveform_stft: torch.Tensor = torch.stft(
            waveform,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            pad_mode=self.pad_mode,
            normalized=self.normalized,
            onesided=self.onesided
        )

        waveform_stft.transpose_(-3, -2)
        waveform_stft = torch.norm(waveform_stft, 2, -1) ** self.power
        waveform_stft.transpose_(-1, -2)

        return waveform_stft


class MelSpectrogram(nn.Module):

    def __init__(self, hparams: Hyperparams):
        super().__init__()

        self.hparams = hparams

        self.spectrogram = Spectrogram(
            self.hparams.n_fft,
            self.hparams.win_length,
            self.hparams.hop_length
        )

        self.mel_scale = transforms.MelScale(
            self.hparams.n_mels,
            self.hparams.sr,
            n_stft=self.hparams.n_fft // 2 + 1
        )

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        spectrogram: torch.Tensor = self.spectrogram(samples)
        mel_spectrogram: torch.Tensor = self.mel_scale(spectrogram)

        return mel_spectrogram
