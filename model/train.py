import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

#================ Variables ================#
num_epochs = 200    # -> number of time the model will see whole dataset
epoch_log = 5 # -> prints per epoch 
evaluation_interval = 25 # -> evaluate model every 'AssertionErrorevaluation_interval' epochs
evaluation_steps = 10  # -> number of iterations for evaluation process (how many batches will be used)
log_to_file = 2 # log output to the file every # epochs

# Parameters following https://arxiv.org/pdf/1805.08318.pdf
generator_lr = 2e-5  # -> generator learning rate
discriminator_lr = 5e-5 # -> discriminator learning rate
adam_beta1 = 0 # -> beta1 for AdamW optimizer
adam_beta2 = 0.9 # -> beta2 (momentum) value for AdamW optimizer
latent_dim = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate_model(generator, discriminator, dataloader):
  generator.eval()
  discriminator.eval()
  real_accuracy, fake_accuracy = 0, 0
  loss = 0

  for i, (ks_symbols, ks_times) in enumerate(dataloader):
    ks_times, ks_symbols = ks_times.to(device), ks_symbols.to(device)
    latent_space = torch.randn(ks_symbols.shape[0], latent_dim, device=device)
    generated_out = generator(latent_space, ks_symbols)
    loss += F.mse_loss(input=generated_out, target=ks_times).item()

    labels = torch.ones(ks_symbols.shape[0], 1, device=device)
    disc_real_output = discriminator(ks_times, ks_symbols)
    real_accuracy += (torch.round(disc_real_output) == labels).float().mean().item()
    
    labels.fill_(0.)

    disc_fake_output = discriminator(generated_out, ks_symbols)
    fake_accuracy += (torch.round(disc_fake_output) == labels).float().mean().item()

    if i == evaluation_steps - 1:
      break

  loss /= evaluation_steps
  real_accuracy /= evaluation_steps
  fake_accuracy /= evaluation_steps

  discriminator.train()
  generator.train()

  return loss, real_accuracy, fake_accuracy


def calculate_loss(input, target):
  return F.binary_cross_entropy(input, target)


def write_to_file(outfile, data):
  with open(outfile, 'w') as file:
    file.writelines('\t'.join(str(j) for j in i) + '\n' for i in data)


def train_step(models, optims, keystrokes, keystroke_times):
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
  total_loss_D.backward()
  optim_D.step()

  # 3. Train the generator
  generator.zero_grad()
  loss_G = calculate_loss(discriminator(generated_keystroke_times, keystrokes), real_label)
  loss_G.backward()
  optim_G.step()

  return loss_G, total_loss_D


def train_loop(generator, discriminator, train_dataloader, validation_dataloader, device=device):
  loss_list_D, loss_list_G = [], []
  generator, discriminator = generator.to(device), discriminator.to(device)
  optim_G = torch.optim.AdamW(generator.parameters(), lr=generator_lr, betas=(adam_beta1, adam_beta2))
  optim_D = torch.optim.AdamW(discriminator.parameters(), lr=discriminator_lr, betas=(adam_beta1, adam_beta2))

  for epoch in tqdm(range(1, num_epochs + 1)):
    curr_loss_G, curr_loss_D = 0, 0
    for index, (keystroke_symbols, keystroke_times) in enumerate(train_dataloader):
      loss_G, loss_D = train_step(models=(generator, discriminator), optims=(optim_G, optim_D), 
                                  keystrokes=keystroke_symbols, keystroke_times=keystroke_times)
      curr_loss_G += loss_G.item()
      curr_loss_D += loss_D.item()

      if index % (len(train_dataloader) // epoch_log) == 0:
        print(f"[Epoch: {epoch} / {num_epochs}][{index:4d}/{len(train_dataloader):4d}] Generator loss: {loss_G:2.5f}, discriminator loss: {loss_D:2.5f}")
    
    if epoch % evaluation_interval:
      loss, real_accuracy, fake_accuracy = evaluate_model(generator, discriminator, validation_dataloader)
      print(f"MSE loss: {loss:2.5f}, Real accuracy: {real_accuracy:2.5f}, Fake accuracy: {fake_accuracy:2.5f}")  
    curr_loss_G /= len(train_dataloader)
    curr_loss_D /= len(train_dataloader)
    loss_list_D.append(curr_loss_D)
    loss_list_G.append(curr_loss_G)
    print(f"###### [Epoch: {epoch} / {num_epochs}] Epoch generator loss: {curr_loss_G:2.5f}, Epoch discriminator loss: {curr_loss_D:2.5f}")

  return loss_list_G, loss_list_D