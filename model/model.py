import torch 
import json
import sys
import os

from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F
from data import *
from time import time 
from tqdm import tqdm

sys.path.append(f"../masters_thesis")
from analysis.helpers import *


#================ Variables ================#

torch.manual_seed(20801)

num_epochs = 200    # -> number of time the model will see whole dataset
epoch_log = 0.25 # -> prints per epoch 
evaluation_iters = 500  # -> number of iterations for evaluation process (how many batches will be used)
log_to_file = 2 # log output to the file every # epochs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 128 # (TODO: 128)
leaky_relu_slope = 0.1
# Parameters following https://arxiv.org/pdf/1805.08318.pdf
generator_lr = 1e-4  # -> generator learning rate
discriminator_lr = 2e-4 # -> discriminator learning rate
adam_beta1 = 0 # -> beta1 for AdamW optimizer
adam_beta2 = 0.9 # -> beta2 (momentum) value for AdamW optimizer

block_size = 8  # -> window size for the dataset (length of the context)
latent_dim = 100 # -> size of the latent space
embedding_dim = 96 # 192 -> embedding dimension for the condition (TODO: 384)
n_attention_blocks_gen = 6  # -> number of consecutive Self-Attention blocks in generator
n_attention_blocks_dis = 3 # -> number of consecutive Self-Attention blocks in discriminator
sa_num_heads = 6  # -> head size in Self-Attention blocks
ca_num_heads = 6 # -> number of heads in Cross-Attention block - each head with a size of embedding_dim // num_heads = 192 / 6 = 32
attn_dropout = 0.2 # -> dropout for Attention blocks
ffn_dropout = 0.2 # -> dropout for feed-forward networks
fc_cond_inner = 128 # -> size of inner layer of condition embedding (TODO: 256)
inner_mapping = Mapper().inner_mapping   # -> used to convert the real ASCII key symbols to range [0; 100] 
vocab_size = max(inner_mapping.values()) + 1 # -> to be used in ks_embedding_table, equals to 100
generator_output_dim = 4

class EfficientZeroGrad():
  # Following https://betterprogramming.pub/how-to-make-your-pytorch-code-run-faster-93079f3c1f7b
  def zero_grad(self):
    for parameter in self.parameters():
      parameter.grad = None

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


#================ Feed Forward Network ================#

class FeedForward(nn.Module):
  def __init__(self, inner_dim: int, num_inner_layers: int = 0, in_dim: int = embedding_dim, out_dim: int = embedding_dim):
    super(FeedForward, self).__init__()
    layers = [
      nn.Linear(in_features=in_dim, out_features=inner_dim),
      nn.LeakyReLU(leaky_relu_slope, inplace=True),
    ]
    for _ in range(num_inner_layers):
      layers.append(nn.Linear(in_features=inner_dim, out_features=inner_dim))
      layers.append(nn.LeakyReLU(leaky_relu_slope, inplace=True))
      layers.append(nn.Dropout(ffn_dropout))

    layers.append(nn.Linear(in_features=inner_dim, out_features=out_dim))
    layers.append(nn.Dropout(ffn_dropout))
    
    self.ffn = nn.Sequential(*layers)

  def forward(self, x):
    return self.ffn(x)

#================ Condition Embedding ================#

class ConditionEmbedding(nn.Module):
  def __init__(self, embedding_dim: int, inner_dim: int):
    super(ConditionEmbedding, self).__init__()
    # Embed the condition's keystrokes and position
    self.ks_embedding_table_cond = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
    self.position_embedding_table_cond = nn.Embedding(num_embeddings=block_size, embedding_dim=embedding_dim)
    # Linear projection
    self.ffn = FeedForward(inner_dim=inner_dim)
  
  def forward(self, x):
    batch_dim, time_dim = x.shape
    ks_embd = self.ks_embedding_table_cond(x)  # (batch_dim, time_dim, embedding_dim)
    pos_embd = self.position_embedding_table_cond(torch.arange(time_dim, device=device)) #  (batch_dim, embedding_dim)
    x = ks_embd + pos_embd #  (batch_dim, time_dim, embedding_dim)
    x = self.ffn(x) # (batch_dim, time_dim, embedding_dim)
    return x

#================ Self-Attention Block ===============#

class AttentionBlock(nn.Module):
  def __init__(self, fc_inner_dim: int):
    super(AttentionBlock, self).__init__()
    self.sa_block = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=sa_num_heads, dropout=attn_dropout)
    self.ln1 = nn.LayerNorm(embedding_dim)
    self.ffn = FeedForward(inner_dim=fc_inner_dim)
    self.ln2 = nn.LayerNorm(embedding_dim)

  def forward(self, x):
    x = self.ln1(x)
    x_attn, _ = self.sa_block(query=x, key=x, value=x)
    x = x + x_attn
    x = x + self.ffn(self.ln2(x))
    return x

#================ Generator ================#

class Generator(nn.Module, EfficientZeroGrad):
  def __init__(self):
    super(Generator, self).__init__()
    self.condition_embd = ConditionEmbedding(embedding_dim, fc_cond_inner)
    self.latent_fc = nn.Linear(in_features=latent_dim, out_features=block_size * embedding_dim)
    self.l_relu = nn.LeakyReLU(leaky_relu_slope, inplace=True)
    self.cross_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=ca_num_heads, dropout=attn_dropout)
    self.sa_blocks = nn.Sequential(*[AttentionBlock(fc_inner_dim=4*embedding_dim) for _ in range(n_attention_blocks_gen)])
    self.ln = nn.LayerNorm(embedding_dim)
    self.ffn_head = FeedForward(inner_dim=128, num_inner_layers=2, out_dim=generator_output_dim)
    self.apply(self._init_weights)

  def forward(self, latent_space, condition):
    batch_dim, time_dim = condition.shape
    # Embed the condition
    cond = self.condition_embd(condition)  # (batch_dim, time_dim, embedding_dim)
    # Project the latent space to get an appropriate shape 
    latent = self.l_relu(self.latent_fc(latent_space))  # (batch_dim, embedding_dim)
    latent = latent.view(batch_dim, time_dim, embedding_dim)  # (batch_dim, time_dim, embedding_dim)
    # Cross-Attention: condition (key, value), latent space (query)
    x, _ = self.cross_attention(query=latent, value=cond, key=cond)  # (batch_dim, time_dim, embedding_dim)
    x = self.sa_blocks(x)  # (batch_dim, time_dim, embedding_dim)
    x = self.ffn_head(self.ln(x))  # (batch_dim, time_dim, 4)

    return x
  
#================ Discriminator ================#

class Discriminator(nn.Module, EfficientZeroGrad):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.condition_embd = ConditionEmbedding(embedding_dim, fc_cond_inner)
    self.latent_fc = nn.Linear(in_features=generator_output_dim, out_features=embedding_dim)
    self.l_relu = nn.LeakyReLU(leaky_relu_slope, inplace=True)
    self.cross_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=ca_num_heads, dropout=attn_dropout)
    self.sa_blocks = nn.Sequential(*[AttentionBlock(fc_inner_dim=2*embedding_dim) for _ in range(n_attention_blocks_dis)])
    self.ffn = FeedForward(inner_dim=4 * embedding_dim, num_inner_layers=1)
    self.ln1 = nn.LayerNorm(embedding_dim)
    self.disc_head = nn.Linear(in_features=block_size * embedding_dim, out_features=1)
    self.sigmoid = nn.Sigmoid()
    self.apply(self._init_weights)

  def forward(self, keystroke_times, condition_symbols):
    batch_dim, time_dim, _ = keystroke_times.shape
    cond = self.condition_embd(condition_symbols)  # (batch_dim, time_dim, embedding_dim)
    latent = self.l_relu(self.latent_fc(keystroke_times))  # (batch_dim, time_dim, embedding_dim)
    x, _ = self.cross_attention(query=latent, value=cond, key=cond)  # (batch_dim, time_dim, embedding_dim)
    x = self.sa_blocks(x) # (batch_dim, time_dim, embedding_dim)
    x = self.ln1(self.ffn(x)) # (batch_dim, time_dim, embedding_dim)
    x = x.view(batch_dim, -1) # (batch_dim, time_dim * embedding_dim)
    x = self.sigmoid(self.disc_head(x)) # (batch_dim, time_dim, 1)
    return x


def calculate_loss(input, target):
  return F.binary_cross_entropy(input, target)


def write_to_file(outfile, data):
  with open(outfile, 'w') as file:
    file.writelines('\t'.join(str(j) for j in i) + '\n' for i in data)


@torch.no_grad()
def generator_proximity(generator, dataloader):
  generator.eval()
  loss = 0
  for i, (ks_symbols, ks_times) in enumerate(dataloader):
    ks_times, ks_symbols = ks_times.to(device), ks_symbols.to(device)
    latent_space = torch.randn(ks_symbols.shape[0], latent_dim, device=device)
    generated_out = generator(latent_space, ks_symbols)
    loss += F.mse_loss(input=generated_out, target=ks_times).item()
    if i == evaluation_iters - 1:
      break
  
  loss /= evaluation_iters
  generator.train()
  return loss


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


def train_loop(generator, discriminator, dataloader, device=device):
  loss_list_D, loss_list_G = [], []
  generator, discriminator = generator.to(device), discriminator.to(device)
  optim_G = torch.optim.AdamW(generator.parameters(), lr=generator_lr, betas=(adam_beta1, adam_beta2))
  optim_D = torch.optim.AdamW(discriminator.parameters(), lr=discriminator_lr, betas=(adam_beta1, adam_beta2))

  for epoch in tqdm(range(1, num_epochs + 1)):
    curr_loss_G, curr_loss_D = 0, 0
    for index, (keystroke_symbols, keystroke_times) in enumerate(dataloader):
      loss_G, loss_D = train_step(models=(generator, discriminator), optims=(optim_G, optim_D), 
                                  keystrokes=keystroke_symbols, keystroke_times=keystroke_times)
      curr_loss_G += loss_G.item()
      curr_loss_D += loss_D.item()
      loss_list_D.append(loss_D.item())
      loss_list_G.append(loss_G.item())
      if index % (len(dataloader) * epoch_log) == 0:
        print(f"[Epoch: {epoch} / {num_epochs}][{index:4d}/{len(dataloader):4d}] Generator loss: {loss_G:2.5f}, discriminator loss: {loss_D:2.5f}")
        print(f"MSE loss: {generator_proximity(generator, dataloader)}")
    curr_loss_G /= len(dataloader)
    curr_loss_D /= len(dataloader)
    print(f"[Epoch: {epoch} / {num_epochs}] Epoch generator loss: {curr_loss_G:2.5f}, Epoch discriminator loss: {curr_loss_D:2.5f}")
    if epoch % log_to_file == 0:
      latent_space = torch.randn(keystroke_symbols.shape[0], latent_dim, device=device)
      out = generator(latent_space, keystroke_symbols)
      write_to_file(f"results/keystroke_{epoch}.txt", out[0].detach().tolist())

  return loss_list_G, loss_list_D


if __name__ == "__main__":
  # generator = Generator()
  # discriminator = Discriminator()
  # print(sum(p.numel() for p in generator.parameters())/1e6, 'M parameters')
  # print(sum(p.numel() for p in discriminator.parameters())/1e6, 'M parameters')
  dataloader = create_dataloader(path=BIG_DATA_DIR, window_size=block_size, batch_size=128, shuffle=True) 

  torch.save(dataloader, "data_20000_128.pt")

  # train_loop(generator, discriminator, dataloader)

  # for i, (ks, ks_time) in enumerate(dataloader):
  #   latent_space = torch.randn(ks.shape[0], latent_dim, device=device)
  #   generated_out = generator(latent_space, ks)
  #   print(ks_time[1])
  #   print("Generated:")
  #   print(generated_out[1])
  #   break
