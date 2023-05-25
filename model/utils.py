import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List


class Eops():
  # Following https://betterprogramming.pub/how-to-make-your-pytorch-code-run-faster-93079f3c1f7b
  def zero_grad(self):
    for parameter in self.parameters():
      parameter.grad = None

  def _init_weights(self, module: nn.Module):
    classname = module.__class__.__name__
    if classname.find("Linear") != -1:
      nn.init.xavier_uniform_(module.weight)
      if module.bias is not None:
        nn.init.zeros_(module.bias)
    elif classname.find("Conv") != -1:
      nn.init.xavier_uniform_(module.weight)

  def spectral_norm(self, module: nn.Module):
    classname = module.__class__.__name__
    if classname.find("Linear") != -1 or classname.find("Conv") != -1:
      module = torch.nn.utils.parametrizations.spectral_norm(module=module)

  def num_params(self):
    return f"{self.__class__.__name__}: {sum(p.numel() for p in self.parameters())/1e6} M parameters"


class TripletLoss(nn.Module):
  def __init__(self, margin: float = 1.5):
    super(TripletLoss, self).__init__()
    self.register_buffer("margin", torch.tensor(margin))

  def forward(self, anchor, positive, negative):
    """
    Goal: Anchor / Positive -> minimize distance, Anchor / Negative -> maximize distance 
    """
    euclidean_distance_positive = F.pairwise_distance(anchor, positive, keepdim=True)
    euclidean_distance_negative = F.pairwise_distance(anchor, negative, keepdim=True)

    return torch.mean(torch.relu(torch.pow(euclidean_distance_positive, 2) - torch.pow(euclidean_distance_negative, 2) + self.margin))