from typing import List, Dict

import torch
from torch.utils.data._utils.collate import default_collate

from .batch import Batch


class PaddingCollator:

    def __call__(self, instances: List[Dict]) -> Batch:
        batch_size: int = len(instances)
        x_len = torch.LongTensor([instance['x'].size(0) for instance in instances])
        y_len = torch.LongTensor([instance['y'].size(0) for instance in instances])
        paths = [instance['path'] for instance in instances]

        x: torch.Tensor = torch.zeros(batch_size, max(x_len), dtype=torch.float32)
        y: torch.Tensor = torch.zeros(batch_size, max(y_len), dtype=torch.long)

        for i, instance in enumerate(instances):
            x[i, :x_len[i]] = instance['x']
            y[i, :y_len[i]] = instance['y']

        # if y.size(1) == 1:
        #     y = y.squeeze(1)

        batch = Batch(x, y, x_len=x_len, y_len=y_len, paths=paths)
        return batch