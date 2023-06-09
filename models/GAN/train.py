import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from typing import Union, Tuple

#================ Variables ================#

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 128
num_epochs = 500    # -> number of time the model will see whole dataset
epoch_log = 1 # -> prints per epoch 
evaluation_interval = 25 # -> evaluate model every 'revaluation_interval' epochs
evaluation_steps = 50  # -> number of iterations for evaluation process (how many batches will be used)

generator_lr = 3e-4  # -> generator learning rate
discriminator_lr = 4e-4 # -> discriminator learning rate
adam_beta1 = 0.5 # -> beta1 for AdamW optimizer
adam_beta2 = 0.999 # -> beta2 (momentum) value for AdamW optimizer
latent_dim = 500

#================ Methods ================#

@torch.no_grad()
def evaluate_model(generator: nn.Module, discriminator: nn.Module, dataloader: torch.utils.data.DataLoader):
  generator.eval()
  discriminator.eval()
  real_accuracy, fake_accuracy = 0, 0
  reconstruction_loss = 0
  kl_div = 0

  for i, (ks_symbols, ks_times) in enumerate(dataloader):
    ks_times, ks_symbols = ks_times.to(device), ks_symbols.to(device)
    latent_space = torch.randn(ks_symbols.shape[0], latent_dim, device=device)
    generated_out = generator(latent_space, ks_symbols)
    reconstruction_loss += F.mse_loss(input=generated_out, target=ks_times).item()
    kl_div += distribution_similarity(generated_out, ks_times)

    labels = torch.ones(ks_symbols.shape[0], 1, device=device)
    disc_real_output = discriminator(ks_times, ks_symbols)
    real_accuracy += (torch.round(disc_real_output) == labels).float().mean().item()
    
    labels.fill_(0.)

    disc_fake_output = discriminator(generated_out, ks_symbols)
    fake_accuracy += (torch.round(disc_fake_output) == labels).float().mean().item()

    if i == evaluation_steps - 1:
      break

  reconstruction_loss /= evaluation_steps
  kl_div /= evaluation_steps
  real_accuracy /= evaluation_steps
  fake_accuracy /= evaluation_steps

  discriminator.train()
  generator.train()

  return reconstruction_loss, kl_div, real_accuracy, fake_accuracy


def distribution_similarity(generated_dist: torch.Tensor, actual_dist: torch.Tensor):
  generated_scaled = (generated_dist - actual_dist.mean()) / actual_dist.std()  # if similar to actual -> result should be close to the normal one 
  kl_loss = torch.log(generated_scaled.std()) + (1 + torch.square(generated_scaled.mean())) / (2 * torch.square(generated_scaled.std())) - 0.5 # compare with the normal one 
  return kl_loss


def calculate_loss(input: torch.Tensor, target: torch.Tensor):
  return F.binary_cross_entropy(input, target)


def reconstruction_loss(input: torch.Tensor, target: torch.Tensor):
  return F.mse_loss(input=input, target=target).item()


def train_step(models, optims, keystrokes, keystroke_times, rl_loss_lambda):
  generator, discriminator = models
  optim_G, optim_D = optims

  # 1. Prepare data (set real / fake labels)
  keystrokes, real_keystroke_times = keystrokes.to(device), keystroke_times.to(device)
  real_label = torch.ones(keystrokes.shape[0], 1, device=device)
  fake_label = torch.zeros(keystrokes.shape[0], 1, device=device)

  # 2. Train the discriminator
  discriminator.zero_grad()
  real_loss_D = calculate_loss(discriminator(real_keystroke_times, keystrokes), real_label)
  latent_space = torch.randn(keystrokes.shape[0], latent_dim, device=device)
  generated_keystroke_times = generator(latent_space, keystrokes)
  fake_loss_D = calculate_loss(discriminator(generated_keystroke_times.detach(), keystrokes), fake_label)
  total_loss_D = (real_loss_D + fake_loss_D) / 2
  mse_loss = reconstruction_loss(generated_keystroke_times, real_keystroke_times)
  total_loss_D = total_loss_D + rl_loss_lambda * mse_loss
  total_loss_D.backward()
  optim_D.step()

  # 3. Train the generator
  generator.zero_grad()
  loss_G = calculate_loss(discriminator(generated_keystroke_times, keystrokes), real_label)
  loss_G = loss_G + rl_loss_lambda * mse_loss
  loss_G.backward()
  optim_G.step()

  kl_div = distribution_similarity(generated_keystroke_times.detach(), keystroke_times.detach()).item()

  return loss_G, total_loss_D, mse_loss, kl_div


def train_loop(generator, discriminator, train_dataloader, validation_dataloader, num_epochs=num_epochs, generator_lr=generator_lr, discriminator_lr=discriminator_lr, device=device, rl_loss_lambda=5, verbose=10, output=True):
  actuals, outputs = [], []
  loss_list_D, loss_list_G, mse_losses = [], [], []
  real_accuracies, fake_accuracies, eval_reconstruction_losses, kl_divergences = [], [], [], [] 
  train_kl_divs = []
  generator, discriminator = generator.to(device), discriminator.to(device)
  optim_G = torch.optim.AdamW(generator.parameters(), lr=generator_lr, betas=(adam_beta1, adam_beta2))
  optim_D = torch.optim.AdamW(discriminator.parameters(), lr=discriminator_lr, betas=(adam_beta1, adam_beta2))

  for epoch in tqdm(range(1, num_epochs + 1)):
    curr_loss_G, curr_loss_D, curr_mse, curr_kl = 0, 0, 0, 0
    for index, (keystroke_symbols, keystroke_times) in enumerate(train_dataloader):
      loss_G, loss_D, mse_loss, kl_div = train_step(models=(generator, discriminator), optims=(optim_G, optim_D), 
                                                    keystrokes=keystroke_symbols, keystroke_times=keystroke_times, rl_loss_lambda=rl_loss_lambda)
      curr_loss_G += loss_G.item()
      curr_loss_D += loss_D.item()
      curr_mse += mse_loss
      curr_kl += kl_div

      if output and index % (len(train_dataloader) // epoch_log) == 0:
        print(f"[Epoch: {epoch} / {num_epochs}][{index:4d}/{len(train_dataloader):4d}] Generator loss: {loss_G:2.5f}, discriminator loss: {loss_D:2.5f}")
    
    if epoch % evaluation_interval == 0:
      acts, outs, loss, kl_div, real_accuracy, fake_accuracy = evaluate_model(generator, discriminator, validation_dataloader)
      actuals.append(acts)
      outputs.append(outs)
      real_accuracies.append(real_accuracy)
      fake_accuracies.append(fake_accuracy)
      eval_reconstruction_losses.append(loss)
      kl_divergences.append(kl_div)
      if output:
        print(f"MSE loss: {loss:2.5f}, KL div: {kl_div:2.5f}, Real accuracy: {real_accuracy:2.5f}, Fake accuracy: {fake_accuracy:2.5f}")

    curr_loss_G /= len(train_dataloader)
    curr_loss_D /= len(train_dataloader)
    curr_mse /= len(train_dataloader)
    curr_kl /= len(train_dataloader)

    train_kl_divs.append(curr_kl)
    loss_list_D.append(curr_loss_D)
    loss_list_G.append(curr_loss_G)
    mse_losses.append(curr_mse)

    if output and epoch % verbose == 0:
      print(f"###### [Epoch: {epoch} / {num_epochs}] Epoch MSE loss: {curr_mse:2.10f} Epoch generator loss: {curr_loss_G:2.5f}, Epoch discriminator loss: {curr_loss_D:2.5f}")

  return actuals, outputs, loss_list_G, loss_list_D, mse_losses, eval_reconstruction_losses, train_kl_divs, real_accuracies, fake_accuracies