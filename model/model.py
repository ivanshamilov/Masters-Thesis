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


block_size = 16  # -> window size for the dataset (length of the context)
latent_dim = 100 # -> size of the latent space
embedding_dim = 96 # 192 -> embedding dimension for the condition
ca_num_heads = 6 # -> number of heads in Cross-Attention block - each head with a size of embedding_dim // num_heads = 192 / 6 = 32
ca_dropout = 0.15 # -> dropout for Cross-Attention block
ffn_dropout = 0.1 # -> dropout for feed-forward networks
sa_num_heads = 6  # -> head size in Self-Attention blocks
fc_cond_inner = 128 # -> size of inner layer of condition embedding
inner_mapping = Mapper().inner_mapping   # -> used to convert the real ASCII key symbols to range [0; 100] 
vocab_size = max(inner_mapping.values()) + 1 # -> to be used in ks_embedding_table, equals to 100

#================ Condition Embedding ================#

class ConditionEmbedding(nn.Module):
  def __init__(self, embedding_dim, inner_dim):
    super(ConditionEmbedding, self).__init__()
    # Embed the condition's keystrokes and position
    self.ks_embedding_table_cond = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
    self.position_embedding_table_cond = nn.Embedding(num_embeddings=block_size, embedding_dim=embedding_dim)
    # Linear projection
    self.ffn = nn.Sequential(
      nn.Linear(in_features=embedding_dim, out_features=inner_dim),
      nn.ReLU(True),
      nn.Linear(in_features=inner_dim, out_features=embedding_dim),
      nn.Dropout(ffn_dropout)
    )
  
  def forward(self, x):
    batch_dim, time_dim = x.shape
    ks_embd = self.ks_embedding_table_cond(x)  # (batch_dim, time_dim, embedding_dim)
    pos_embd = self.position_embedding_table_cond(torch.arange(time_dim, device=device)) #  (batch_dim, embedding_dim)
    x = ks_embd + pos_embd #  (batch_dim, time_dim, embedding_dim)
    x = self.ffn(x) # (batch_dim, time_dim, embedding_dim)
    return x

#================ Generator ================#

class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()
    self.condition_embd = ConditionEmbedding(embedding_dim, fc_cond_inner)
    self.latent_fc = nn.Linear(in_features=latent_dim, out_features=block_size * embedding_dim)
    self.cross_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=ca_num_heads, dropout=ca_dropout)

  def forward(self, latent_space, x):
    batch_dim, latent_dim = latent_space.shape
    # Embed the condition
    cond = self.condition_embd(x)  # (batch_dim, time_dim, embedding_dim)
    print(f"Condition data shape: {cond.shape}")
    # Project the latent space to get an appropriate shape 
    latent = self.latent_fc(latent_space)  # (batch_dim, embedding_dim)
    latent = latent.view(batch_dim, block_size, embedding_dim)
    latent = F.leaky_relu(latent, 0.2)
    print(f"Latent space shape: {latent.shape}")
    # Cross-Attention: condition (key, value), latent space (query)
    x, _ = self.cross_attention(query=latent, value=cond, key=cond)

    return x
  

if __name__ == "__main__":
  dataloader = create_dataloader(path=MAIN_DIR, window_size=block_size, batch_size=batch_size, shuffle=True, limit=1)
  generator = Generator()

  for i, (ks, ks_time) in enumerate(dataloader):
    latent_space = torch.randn(ks.shape[0], latent_dim, device=device)
    print(f"Output shape: {generator(latent_space, ks).shape}")
    break
