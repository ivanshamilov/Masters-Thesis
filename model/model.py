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
batch_size = 128 # (TODO: 128)
leaky_relu_slope = 0.1

block_size = 16  # -> window size for the dataset (length of the context)
latent_dim = 100 # -> size of the latent space
embedding_dim = 96 # 192 -> embedding dimension for the condition (TODO: 384)
sa_num_heads = 6  # -> head size in Self-Attention blocks
ca_num_heads = 6 # -> number of heads in Cross-Attention block - each head with a size of embedding_dim // num_heads = 192 / 6 = 32
attn_dropout = 0.2 # -> dropout for Attention blocks
ffn_dropout = 0.2 # -> dropout for feed-forward networks
fc_cond_inner = 128 # -> size of inner layer of condition embedding (TODO: 256)
inner_mapping = Mapper().inner_mapping   # -> used to convert the real ASCII key symbols to range [0; 100] 
vocab_size = max(inner_mapping.values()) + 1 # -> to be used in ks_embedding_table, equals to 100
generator_output_dim = 4


if __name__ == "__main__":
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