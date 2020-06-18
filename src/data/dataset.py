from typing import Dict, Any, Optional

import os
import librosa
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset

from .transforms.audio import AudioProcessor


class SpeechDataset(Dataset):
    _SAMPLE_RATE = 16000
    _ROOT_PREFIX = 'data/numbers'

    def __init__(
        self,
        path_to_csv: str,
        audio_processor: Optional[AudioProcessor] = None
    ):
        self.path_to_csv = path_to_csv
        self.audio_processor = audio_processor

        self.csv: pd.DataFrame = pd.read_csv(path_to_csv)

    @staticmethod
    def number2digits(number: np.ndarray) -> torch.Tensor:
        digits = list(map(int, list(str(number))))
        return torch.LongTensor(digits)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        output = self.csv.iloc[index]
        if len(output) == 3:
            path, gender, number = output
            y = self.number2digits(number)
        else:
            path, = output
            y = torch.LongTensor([-1])

        path = os.path.join(self._ROOT_PREFIX, path)

        wav, _ = librosa.load(path, sr=self._SAMPLE_RATE)
        wav = torch.from_numpy(wav).squeeze()

        if self.audio_processor is not None:
            pass

        instance = {
            'path': path,
            'x': wav,
            'y': y
        }

        return instance

    def __len__(self):
        return self.csv.shape[0]
