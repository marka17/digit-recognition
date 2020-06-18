import torch
from torch import nn


class CTCLoss(nn.Module):
    """
    CTC loss
    """

    def __init__(self, blank=0):
        super().__init__()
        self.criterion = nn.CTCLoss(blank=blank, reduction='none')

    def __call__(self, log_probs, targets, input_length, target_length):
        input_length = input_length.long()
        target_length = target_length.long()
        targets = targets.long()

        max_input_length = log_probs.shape[1]
        input_length = torch.clamp(input_length, max=max_input_length).long()

        loss = self.criterion(
            log_probs.transpose(1, 0),
            targets,
            input_length,
            target_length
        )

        return torch.mean(loss)