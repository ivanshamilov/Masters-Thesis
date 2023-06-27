import torch 
import torch.nn as nn

from tqdm import tqdm
from typing import Union


@torch.no_grad()
def evaluate(classificator: nn.Module, dataloader: torch.utils.data.DataLoader, device: Union[str, torch.device] = "cpu"):
  classificator.eval()
  loss_fn = nn.BCELoss()
  losses, accuracies = 0, 0

  for (X, y) in dataloader:
    X, y = X.to(device), y.to(device)
    out = classificator(X)
    loss = loss_fn(out, y)
    losses += loss.item()
    accuracies += (torch.round(out) == y).float().mean().item()

  classificator.train()

  return losses, accuracies


def train_loop(classificator: nn.Module, train_dataloader: torch.utils.data.DataLoader, validation_dataloader: torch.utils.data.DataLoader, device: Union[str, torch.device], 
               num_epochs: int, learning_rate: float):
  classificator.train()
  loss_fn = nn.BCELoss()

  losses, accuracies = [], []
  val_losses, val_accuracies = [], []

  optim = torch.optim.Adam(classificator.parameters(), lr=learning_rate, weight_decay=0.0001, betas=(0.9, 0.999))

  for epoch in tqdm(range(num_epochs)):
    epoch_loss = 0
    epoch_accuracy = 0
    for i, (X, y) in enumerate(train_dataloader):
      X, y = X.to(device), y.to(device)
      classificator.zero_grad()
      out = classificator(X)
      loss = loss_fn(out, y)
      epoch_loss += loss.item()
      epoch_accuracy += (torch.round(out) == y).float().mean().item()

      loss.backward()
      optim.step()

    epoch_loss /= len(train_dataloader)
    epoch_accuracy /= len(train_dataloader)

    val_loss, val_acc = evaluate(classificator, validation_dataloader, device)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    losses.append(epoch_loss)
    accuracies.append(epoch_accuracy)


  return losses, accuracies, val_losses, val_accuracies
