import torch 
import json
import sys
import os

import torch.nn as nn
import torch.nn.functional as F
from data import *

sys.path.append(f"../masters_thesis")
from analysis.helpers import *

#================ variables ================#

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 64
leaky_relu_slope = 0.05
block_size = 16  # -> window size for the dataset (length of the context)
latent_dim = 100 # -> size of the latent space
embedding_dim = 96 # 192 -> embedding dimension for the condition
n_attention_blocks_gen = 3 # -> number of consecutive Self-Attention blocks in generator
n_attention_blocks_dis = 1 # -> number of consecutive Self-Attention blocks in discriminator
sa_num_heads = 6  # -> head size in Self-Attention blocks
ca_num_heads = 6 # -> number of heads in Cross-Attention block - each head with a size of embedding_dim // num_heads = 192 / 6 = 32
attn_dropout = 0.15 # -> dropout for Attention blocks
ffn_dropout = 0.1 # -> dropout for feed-forward networks
fc_cond_inner = 128 # -> size of inner layer of condition embedding
inner_mapping = Mapper().inner_mapping   # -> used to convert the real ASCII key symbols to range [0; 100] 
vocab_size = max(inner_mapping.values()) + 1 # -> to be used in ks_embedding_table, equals to 100
generator_output_dim = 4

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
    x += x_attn
    x += self.ffn(self.ln2(x))
    return x

#================ Generator ================#

class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()
    self.condition_embd = ConditionEmbedding(embedding_dim, fc_cond_inner)
    self.latent_fc = nn.Linear(in_features=latent_dim, out_features=block_size * embedding_dim)
    self.l_relu = nn.LeakyReLU(leaky_relu_slope, inplace=True)
    self.cross_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=ca_num_heads, dropout=attn_dropout)
    self.sa_blocks = nn.Sequential(*[AttentionBlock(fc_inner_dim=2*embedding_dim) for _ in range(n_attention_blocks_gen)])
    self.ln = nn.LayerNorm(embedding_dim)
    self.ffn_head = FeedForward(inner_dim=128, out_dim=generator_output_dim)

  def forward(self, latent_space, x):
    batch_dim, latent_dim = latent_space.shape
    # Embed the condition
    cond = self.condition_embd(x)  # (batch_dim, time_dim, embedding_dim)
    # Project the latent space to get an appropriate shape 
    latent = self.l_relu(self.latent_fc(latent_space))  # (batch_dim, embedding_dim)
    latent = latent.view(batch_dim, block_size, embedding_dim)  # (batch_dim, time_dim, embedding_dim)
    # Cross-Attention: condition (key, value), latent space (query)
    x, _ = self.cross_attention(query=latent, value=cond, key=cond)  # (batch_dim, time_dim, embedding_dim)
    x = self.sa_blocks(x)  # (batch_dim, time_dim, embedding_dim)
    x = self.ffn_head(self.ln(x))  # (batch_dim, time_dim, 4)

    return x
  
#================ Discriminator ================#

class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.condition_embd = ConditionEmbedding(embedding_dim, fc_cond_inner)
    self.latent_fc = nn.Linear(in_features=generator_output_dim, out_features=embedding_dim)
    self.l_relu = nn.LeakyReLU(leaky_relu_slope, inplace=True)
    self.cross_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=ca_num_heads, dropout=attn_dropout)
    self.sa_blocks = nn.Sequential(*[AttentionBlock(fc_inner_dim=4*embedding_dim) for _ in range(n_attention_blocks_dis)])
    self.ffn = FeedForward(inner_dim=4 * embedding_dim, num_inner_layers=1)
    self.ln1 = nn.LayerNorm(embedding_dim)
    self.disc_head = nn.Linear(in_features=embedding_dim, out_features=1)
    self.sigmoid = nn.Sigmoid()

  def forward(self, keystrokes, condition_symbols):
    batch_dim, latent_dim = latent_space.shape
    cond = self.condition_embd(condition_symbols)  # (batch_dim, time_dim, embedding_dim)
    latent = self.l_relu(self.latent_fc(keystrokes))  # (batch_dim, time_dim, embedding_dim)
    x, _ = self.cross_attention(query=latent, value=cond, key=cond)  # (batch_dim, time_dim, embedding_dim)
    x = self.sa_blocks(x) # (batch_dim, time_dim, embedding_dim)
    x = self.ln1(self.ffn(x)) # (batch_dim, time_dim, embedding_dim)
    x = self.sigmoid(self.disc_head(x)) # (batch_dim, time_dim, 1)
    return x

  
if __name__ == "__main__":
  dataloader = create_dataloader(path=MAIN_DIR, window_size=block_size, batch_size=batch_size, shuffle=True, limit=1)
  generator = Generator()
  discriminator = Discriminator()
  for i, (ks, ks_time) in enumerate(dataloader):
    latent_space = torch.randn(ks.shape[0], latent_dim, device=device)
    # out = generator(latent_space, ks)
    out = discriminator(ks_time, ks)
    print(out.shape)
    break
