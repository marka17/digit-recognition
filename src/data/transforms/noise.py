import random
import os
import h5py

import torch
from torch.utils.data import Dataset


class NoiseDataset(Dataset):
    """
    # TODO
    """

    def __init__(self, index_path: str):
        self.index_path = index_path
        self.index_dir = os.path.dirname(self.index_path)

        with open(self.index_path, 'r') as f:
            files_paths = f.read().splitlines()
            self.paths = [p if os.path.isabs(p) else os.path.join(self.index_dir, p) for p in files_paths]

    def __getitem__(self, index: int) -> torch.Tensor:
        filename: str = self.paths[index]

        with h5py.File(filename, 'r') as f:
            samples = torch.from_numpy(f['samples'][()].squeeze()).to(torch.float32)

        return samples

    def random_noise(self):
        random_index: int = random.randint(0, len(self.paths) - 1)
        return self.__getitem__(random_index)

    def __len__(self):
        return len(self.paths)
