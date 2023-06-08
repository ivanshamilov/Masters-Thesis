import sys

sys.path.append("../../masters_thesis")

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from typing import List, Tuple, Union

from model.GAN.model import Generator, Discriminator
from model.GAN.train import train_loop, device


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

  def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor = None):
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


def make_predictions(generated_sample: torch.Tensor, real_time_sample: torch.Tensor, ks_symbols: torch.Tensor, 
                     generator: nn.Module, typenet: nn.Module, device: str, threshold: float, return_raw: bool = True):
  generator.eval()
  typenet.eval()
  generator.to(device)
  typenet.to(device)
  generated_sample = generated_sample.to(device)
  real_time_sample = real_time_sample.to(device)

  ks_symbols = ks_symbols.float() / 100

  generated_sample = torch.cat((generated_sample, ks_symbols.unsqueeze(dim=-1)), dim=2)
  real_time_sample = torch.cat((real_time_sample, ks_symbols.unsqueeze(dim=-1)), dim=2)
  _, _, _, loss = typenet(anchor=generated_sample, positive=real_time_sample)

  if return_raw:
    return loss["ap_distance"]

  return (loss["ap_distance"] < threshold).float()


def get_decimal_places(a: float):
  return len(str(int(1 / a))) * -1


def find_learning_rate(learning_rates: np.array, dataloader: torch.data.utils.DataLoader, gen_pretrained_path: str = None, 
                       disc_pretrained_path: str = None, device: Union[str, torch.device] = device, num_epochs: int = 20, rl_loss_lambda: int = 50):
  kls, losses = [], []

  for lr in learning_rates:
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    if gen_pretrained_path:
      generator.load_pretrained(gen_pretrained_path)
    if disc_pretrained_path:
      discriminator.load_pretrained(disc_pretrained_path)
    
    _, _ , _, _, mse_losses, _, kl_divs, _, _, _ = train_loop(generator, discriminator, dataloader, dataloader, generator_lr=lr, 
                                                              discriminator_lr=lr + 10 ** get_decimal_places(lr), verbose=1, num_epochs=num_epochs, rl_loss_lambda=rl_loss_lambda, output=False)
    print(lr, mse_losses[0] / mse_losses[-1], kl_divs[0] / kl_divs[-1])
    losses.append(mse_losses[0] / mse_losses[-1])
    kls.append(kl_divs[0] / kl_divs[-1])

  results = [i * j for i, j in zip(losses, kls)]
  best_index = np.argmax(results)

  return learning_rates[best_index], learning_rates[best_index] + 10 ** get_decimal_places(learning_rates[best_index]), losses, kls