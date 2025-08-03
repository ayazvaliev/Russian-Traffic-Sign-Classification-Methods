import torch.nn as nn
import torch

from numpy import unique
from torch.nn.functional import cross_entropy


class FeaturesLoss(nn.Module):
    """
    Contrasive loss for training feature extractor
    """

    def __init__(self, margin: float) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = outputs.device
        labels = labels.argmax(dim=-1)
        same_class_loss = torch.tensor(0, dtype=torch.float32, device=device)
        diff_class_loss = torch.tensor(0, dtype=torch.float32, device=device)
        classes = unique(labels.cpu().numpy())
        same_class_pairs_counter = 0
        for cls in classes:
            same_class_outputs = outputs[labels == cls]
            for i in range(same_class_outputs.size(0)):
                for j in range(i + 1, same_class_outputs.size(0)):
                    same_class_loss += torch.sum((same_class_outputs[i] - same_class_outputs[j])**2)
                    same_class_pairs_counter += 1
        same_class_loss /= same_class_pairs_counter
        diff_class_pairs_counter = outputs.size(0) * (outputs.size(0) - 1) // 2
        for cls_i in range(len(classes)):
            for cls_j in range(cls_i + 1, len(classes)):
                diff_class_loss = torch.tensor(0, dtype=torch.float32, device=device)
                for output_1 in outputs[labels == classes[cls_i]]:
                    for output_2 in outputs[labels == classes[cls_j]]:
                        diff_class_loss += torch.maximum(self.margin - torch.linalg.norm(output_1 - output_2, ord=2, dim=-1),
                                                         torch.tensor(0, dtype=torch.float32, device=device)) ** 2 / diff_class_pairs_counter
        return same_class_loss + diff_class_loss
    

class LabelSmoothingLoss(nn.Module):
    '''
    Cross-entropy loss with modifieable label-smoothing
    '''
    def __init__(self, classes_cnt: int, smoothing=0.0):
        assert 0 <= smoothing < 1

        super().__init__()
        self.smoothing = smoothing
        self.classes_cnt = classes_cnt

    def forward(self, p, y):

        K = self.classes_cnt
        alpha = self.smoothing
        smooth_y = (1 - alpha) * y + (alpha / K)

        loss = cross_entropy(p, smooth_y)
        return loss