import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from typing import Union

from .GAN.model import Generator, Discriminator
from .GAN.train import train_loop, device
from .utils import get_decimal_places


def find_learning_rate(learning_rates: np.array, dataloader: torch.utils.data.DataLoader, gen_pretrained_path: str = None, 
                       disc_pretrained_path: str = None, device: Union[str, torch.device] = device, num_epochs: int = 20, rl_loss_lambda: int = 50):
  kls, losses = [], []

  for lr in learning_rates:
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    if gen_pretrained_path:
      generator.load_pretrained(gen_pretrained_path)
    if disc_pretrained_path:
      discriminator.load_pretrained(disc_pretrained_path)
    
    _, _ , _, _, mse_losses, _, kl_divs, _, _ = train_loop(generator, discriminator, dataloader, dataloader, generator_lr=lr, 
                                                           discriminator_lr=lr + 10 ** get_decimal_places(lr), verbose=1, num_epochs=num_epochs, rl_loss_lambda=rl_loss_lambda, output=False)
    print(lr, mse_losses[0] / mse_losses[-1], kl_divs[0] / kl_divs[-1])
    losses.append(mse_losses[0] / mse_losses[-1])
    kls.append(kl_divs[0] / kl_divs[-1])

  results = [i * j for i, j in zip(losses, kls)]
  best_index = np.argmax(results)

  return learning_rates[best_index], learning_rates[best_index] + 10 ** get_decimal_places(learning_rates[best_index]), losses, kls


@torch.no_grad()
def make_predictions(generated_sample: torch.Tensor, real_time_sample: torch.Tensor, ks_symbols: torch.Tensor, 
                     typenet: nn.Module, device: str, threshold: float, return_raw: bool = True):
  typenet.eval()
  typenet.to(device)
  generated_sample = generated_sample.to(device)
  real_time_sample = real_time_sample.to(device)

  ks_symbols = ks_symbols.float() / 100

  generated_sample = torch.cat((ks_symbols.unsqueeze(dim=-1), generated_sample), dim=2)
  real_time_sample = torch.cat((ks_symbols.unsqueeze(dim=-1), real_time_sample), dim=2)

  print(generated_sample.shape)

  if generated_sample.shape[0] > 128:
    generated_sample = torch.split(generated_sample, 128, dim=0)
    real_time_sample = torch.split(real_time_sample, 128, dim=0)
    ks_symbols = torch.split(ks_symbols, 128, dim=0)
  else:
    generated_sample = generated_sample.unsqueeze(dim=0)
    real_time_sample = real_time_sample.unsqueeze(dim=0)
    ks_symbols = ks_symbols.unsqueeze(dim=0)
  
  print(len(generated_sample))

  losses = torch.tensor([])

  for i in range(len(generated_sample)):
    _, _, _, loss = typenet(anchor=generated_sample[i], positive=real_time_sample[i])
    losses = torch.cat((losses, loss["ap_distance"]))

  if return_raw:
    return losses

  return (losses < threshold).float()