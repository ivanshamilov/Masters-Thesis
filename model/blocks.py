import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List


class Eops():
  # Following https://betterprogramming.pub/how-to-make-your-pytorch-code-run-faster-93079f3c1f7b
  def zero_grad(self):
    for parameter in self.parameters():
      parameter.grad = None

  def _init_weights(self, module: nn.Module):
    classname = module.__class__.__name__
    if classname.find("Linear") != -1:
      nn.init.xavier_uniform(module.weight)
      if module.bias.data is not None:
        nn.init.zeros_(module.bias)
    elif classname.find("Conv") != -1:
      nn.init.xavier_uniform(module.weight)

  def spectral_norm(self, module: nn.Module):
    pass


class FullyConnected(nn.Module):
  """
  Fully Connected block with specified number of hidden layers
  """
  def __init__(self, in_dim: int, out_dim: int, lrelu_slope: float = 1.0, dropout: float = 0.0, hidden_layers_dim: List[int] = [128]):
    super(FullyConnected, self).__init__()
    layers = [
        nn.Linear(in_features=in_dim, out_features=hidden_layers_dim[0]),
        nn.LeakyReLU(lrelu_slope, inplace=True),
    ]

    for i in range(1, len(hidden_layers_dim)):
      layers.append(nn.Linear(in_features=hidden_layers_dim[i-1], out_features=hidden_layers_dim[i]))
      layers.append(nn.LeakyReLU(lrelu_slope, inplace=True))
      layers.append(nn.Dropout(dropout))
    
    layers.append(nn.Linear(in_features=hidden_layers_dim[-1], out_features=out_dim))
    layers.append(nn.Dropout(dropout))

    self.ffn = nn.Sequential(*layers)

  def forward(self, x):
    return self.ffn(x)
  

class ConditionEmbedding(nn.Module):
  def __init__(self, vocab_size: int, block_size: int, embedding_dim: int, inner_dim: int, device: torch.device, out_dim: int = 0):
    super(ConditionEmbedding, self).__init__()
    if out_dim == 0:
      out_dim = embedding_dim
    # Embed the condition's keystrokes and position
    self.ks_embedding_table_cond = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
    self.position_embedding_table_cond = nn.Embedding(num_embeddings=block_size, embedding_dim=embedding_dim)
    # Linear projection
    self.projection = FullyConnected(in_dim=embedding_dim, out_dim=out_dim, hidden_layers_dim=[inner_dim])
    self.register_buffer("pos_enc", torch.arange(block_size, device=device))
  
  def forward(self, x):
    batch_dim, time_dim = x.shape
    ks_embd = self.ks_embedding_table_cond(x)  # (batch_dim, time_dim, embedding_dim)
    pos_embd = self.position_embedding_table_cond(self.pos_enc) #  (batch_dim, embedding_dim)
    x = ks_embd + pos_embd #  (batch_dim, time_dim, embedding_dim)
    x = self.projection(x) # (batch_dim, time_dim, embedding_dim)
    return x


class CrossAttentionCondition(nn.Module):
  def __init__(self, vocab_size: int, block_size: int, latent_dim: int, embedding_dim: int, inner_dim: int, device: torch.device,
               ca_num_heads: int = 6, lrelu_slope: float = 0.2, dropout: float = 0.2, out_dim: int = 0):
    super(CrossAttentionCondition, self).__init__()
    if out_dim == 0:
      out_dim = embedding_dim
    self.cond_embd = ConditionEmbedding(vocab_size=vocab_size, block_size=block_size, embedding_dim=embedding_dim, inner_dim=inner_dim,
                                        device=device, out_dim=out_dim)
    self.latent_fc = nn.Linear(in_features=latent_dim, out_features=block_size * out_dim)
    self.l_relu = nn.LeakyReLU(lrelu_slope, inplace=True)
    self.cross_attention = nn.MultiheadAttention(embed_dim=out_dim, num_heads=ca_num_heads, dropout=dropout)
    
  def forward(self, latent_space, condition): 
    # Embed the condition
    cond = self.cond_embd(condition)  # (batch_dim, time_dim, embedding_dim)
    batch_dim, time_dim, embedding_dim = cond.shape
    # Project the latent space to get an appropriate shape 
    latent = self.l_relu(self.latent_fc(latent_space))  # (batch_dim, embedding_dim)
    latent = latent.view(batch_dim, time_dim, embedding_dim)  # (batch_dim, time_dim, embedding_dim)
    # Cross-Attention: condition (key, value), latent space (query)
    x, _ = self.cross_attention(query=latent, value=cond, key=cond)  # (batch_dim, time_dim, embedding_dim)
    return x
