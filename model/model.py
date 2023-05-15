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
from blocks import *

#================ Variables ================#
torch.manual_seed(20801)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 128 
leaky_relu_slope = 0.1

block_size = 16  # -> window size for the dataset (length of the context)
latent_dim = 100 # -> size of the latent space
embedding_dim = 192 # 192 -> embedding dimension for the condition (TODO: 384)
sa_num_heads = 6  # -> head size in Self-Attention blocks
ca_num_heads = 6 # -> number of heads in Cross-Attention block - each head with a size of embedding_dim // num_heads = 192 / 6 = 32
attn_dropout = 0.2 # -> dropout for Attention blocks
ffn_dropout = 0.15 # -> dropout for feed-forward networks
fc_cond_inner = 128 # -> size of inner layer of condition embedding (TODO: 256)
inner_mapping = Mapper().inner_mapping   # -> used to convert the real ASCII key symbols to range [0; 100] 
vocab_size = max(inner_mapping.values()) + 1 # -> to be used in ks_embedding_table, equals to 100
generator_output_dim = 4
lstm_hidden_size = 192  # the value divisible by sa_num heads. self-attention block: 192 / 6 = 32 channels per head


class Generator(nn.Module, Eops):
  def __init__(self):
    super(Generator, self).__init__()
    self.latent_cond_concat = CrossAttentionCondition(vocab_size=vocab_size, block_size=block_size, latent_dim=latent_dim,
                                                      embedding_dim=embedding_dim, ca_num_heads=ca_num_heads, inner_dim=128, device=device)
    self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=lstm_hidden_size, num_layers=4, batch_first=True, dropout=0.15)
    self.sa_block = AttentionBlock(embedding_dim=lstm_hidden_size, sa_num_heads=sa_num_heads, fc_inner_dim=128, dropout=attn_dropout,
                                   lrelu_slope=leaky_relu_slope)
    self.ffn1 = FullyConnected(in_dim=lstm_hidden_size, out_dim=generator_output_dim, hidden_layers_dim=[256, 512, 128], dropout=ffn_dropout,
                               lrelu_slope=leaky_relu_slope)
    self.apply(self.spectral_norm)
    
  def forward(self, latent_space, condition):
    x = self.latent_cond_concat(latent_space, condition)
    x, _ = self.lstm(x)
    x = self.ffn1(x)
    return F.sigmoid(x)  # keystrokes normalized to [0; 1]


class Discriminator(nn.Module, Eops):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.latent_cond_concat = CrossAttentionCondition(vocab_size=vocab_size, block_size=block_size, latent_dim=generator_output_dim,
                                                      embedding_dim=embedding_dim, ca_num_heads=ca_num_heads, inner_dim=128, device=device,
                                                      generator=False) 
    self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=lstm_hidden_size, num_layers=2, batch_first=True, dropout=0.15)
    self.sa_block = AttentionBlock(embedding_dim=lstm_hidden_size, sa_num_heads=sa_num_heads / 2, fc_inner_dim=128, dropout=attn_dropout,
                                   lrelu_slope=leaky_relu_slope)
    self.ffn1 = FullyConnected(in_dim=lstm_hidden_size, out_dim=1, hidden_layers_dim=[128, 256], dropout=ffn_dropout,
                               lrelu_slope=leaky_relu_slope)
    self.sigmoid = nn.Sigmoid()
    self.apply(self.spectral_norm)

  def forward(self, keystroke_times, condition_symbols):
    x = self.latent_cond_concat(keystroke_times, condition_symbols)
    x, _ = self.lstm(x)
    x = self.ffn1(x)
    x = self.sigmoid(x)
    return x


if __name__ == "__main__":
  limit = 500
  dataloader = create_dataloader(path=BIG_DATA_DIR, window_size=block_size, batch_size=128, shuffle=True, limit=limit) 
  torch.save(dataloader, f"norm_data_{limit}_{batch_size}.pt")
  # dataloader = torch.load("norm_data_500_128.pt")
  # dataloader = torch.load("big_data_10000_128.pt")
  # generator = Generator()
  # discriminator = Discriminator()
  # print(sum(p.numel() for p in generator.parameters())/1e6, 'M parameters')
  # print(sum(p.numel() for p in discriminator.parameters())/1e6, 'M parameters')

  # for i, (ks, ks_time) in enumerate(dataloader):
  #   print(ks.shape, ks_time.shape)
  #   latent_space = torch.randn(ks.shape[0], latent_dim, device=device)
  #   generated_out = generator(latent_space, ks)
  #   print(generated_out)
  #   print(generated_out.max(), generated_out.min(), generated_out.mean(), generated_out.std())
  #   # out = discriminator(generated_out, ks)
  #   # print(out.shape)
  #   break