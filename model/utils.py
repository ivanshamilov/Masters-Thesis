import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple


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
  
  def load_pretrained(self, filepath, device="cpu"):
    print(f"Loading weights for {self.__class__.__name__} from {filepath}")
    print(self.load_state_dict(torch.load(filepath, map_location=torch.device(device))))


class TripletLoss(nn.Module):
  def __init__(self, margin: float = 1.5):
    super(TripletLoss, self).__init__()
    self.register_buffer("margin", torch.tensor(margin))

  def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor = None) -> Tuple[torch.Tensor]:
    """
    Goal: Anchor / Positive -> minimize distance, Anchor / Negative -> maximize distance 
    Implementation available in PyTorch (does not include squaring the distances): nn.TripletMarginLoss(1.5)(anchor, positive, negative).
    """
    loss_dict = { "loss": None }
    euclidean_distance_positive = torch.mean(F.pairwise_distance(anchor, positive, keepdim=True), dim=1)
    loss_dict["ap_distance"] = euclidean_distance_positive
    if negative is not None:
      euclidean_distance_negative = torch.mean(F.pairwise_distance(anchor, negative, keepdim=True), dim=1)
      loss = torch.mean(torch.relu(torch.pow(euclidean_distance_positive, 2) - torch.pow(euclidean_distance_negative, 2) + self.margin))
      loss_dict["loss"] = loss
      loss_dict["an_distance"] = euclidean_distance_negative
      # difference should be > 0 (Anchor-Negative distance should be greater than Anchor-Positive distance)
      loss_dict["an_ap_diff"] = loss_dict["an_distance"] - loss_dict["ap_distance"]

    return loss_dict