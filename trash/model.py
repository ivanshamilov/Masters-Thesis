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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 128 # (TODO: 128)
leaky_relu_slope = 0.1

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
