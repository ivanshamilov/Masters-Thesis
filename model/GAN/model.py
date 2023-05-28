import torch 
import sys

import torch.nn as nn
import torch.nn.functional as F
from data import *

sys.path.append(f"../../masters_thesis")
from analysis.helpers import *
from model.utils import *

#================ Variables ================#
torch.manual_seed(20801)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 128 
leaky_relu_slope = 0.05

block_size = 16  # -> window size for the dataset (length of the context)
latent_dim = 500 # -> size of the latent space
embedding_dim = 144 # embedding dimension for the condition
ffn_dropout = 0.2 # -> dropout for feed-forward networks
inner_mapping = Mapper().inner_mapping   # -> used to convert the real ASCII key symbols to range [0; 100] 
vocab_size = max(inner_mapping.values()) + 1 # -> to be used in ks_embedding_table, equals to 100
generator_output_dim = 2

#================ Models ================#

class Generator(nn.Module, Eops):
  def __init__(self):
    super(Generator, self).__init__()
    self.label_conditioned_generator = nn.Sequential(
      nn.Embedding(vocab_size, embedding_dim),
      nn.Linear(embedding_dim, 512, bias=False)
    )

    self.latent = nn.Sequential(
      nn.Linear(latent_dim, 8 * 16 * 64, bias=False),
      nn.LeakyReLU(leaky_relu_slope, inplace=True)
    )

    self.lstm1 = nn.LSTM(input_size=1024, hidden_size=128, num_layers=2, bias=False, batch_first=True, dropout=0.2)
    self.fc = nn.Linear(in_features=128, out_features=512, bias=False)

    self.model = nn.Sequential(
      nn.LeakyReLU(leaky_relu_slope, inplace=True),
      nn.Linear(in_features=512, out_features=1024, bias=False),
      nn.LayerNorm(1024),
      nn.LeakyReLU(leaky_relu_slope, inplace=True),
      nn.Linear(in_features=1024, out_features=256, bias=False),
      nn.LayerNorm(256),
      nn.LeakyReLU(leaky_relu_slope, inplace=True),
      nn.Dropout(ffn_dropout),
      nn.Linear(in_features=256, out_features=128, bias=False),
      nn.LayerNorm(128),
      nn.LeakyReLU(leaky_relu_slope, inplace=True),
      nn.Linear(in_features=128, out_features=generator_output_dim, bias=False),
      nn.Sigmoid()
    )

    self.apply(self._init_weights)
    print(self.num_params())

  def forward(self, latent_space, condition):
    condition_out = self.label_conditioned_generator(condition)
    latent_out = self.latent(latent_space)
    condition_out = condition_out.view(-1, 16, 4 * 128)
    latent_out = latent_out.view(-1, 16, 4 * 128)
    concat = torch.cat((latent_out, condition_out), dim=2)

    x, _ = self.lstm1(concat)
    x = self.fc(x)
    x = F.dropout(x, 0.5)
    x = self.model(x)

    return x


class Discriminator(nn.Module, Eops):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.label_conditioned_discriminator = nn.Sequential(
      nn.Embedding(vocab_size, embedding_dim),
      nn.Linear(embedding_dim, 32, bias=False)
    )

    self.embed_keystroke = nn.Sequential(
      nn.Linear(in_features=generator_output_dim, out_features=512),
      nn.LeakyReLU(leaky_relu_slope, inplace=True)
    )

    self.lstm1 = nn.LSTM(input_size=544, hidden_size=128, num_layers=2, bias=False, batch_first=True, dropout=0.2)
    self.fc = nn.Linear(in_features=128, out_features=1024, bias=False)

    self.model = nn.Sequential(
      nn.LeakyReLU(leaky_relu_slope, inplace=True),
      nn.Linear(1024, 512),
      nn.LeakyReLU(leaky_relu_slope, inplace=True),
      nn.Dropout(ffn_dropout * 2),
      nn.Linear(512, 256),
      nn.LeakyReLU(leaky_relu_slope, inplace=True),
      nn.Linear(256, 1),
    )

    self.head = nn.Linear(16, 1)
    self.sigmoid = nn.Sigmoid()
    self.apply(self._init_weights)
    print(self.num_params())

  def forward(self, keystroke_times, condition_symbols):
    condition_out = self.label_conditioned_discriminator(condition_symbols)
    keystroke_times = self.embed_keystroke(keystroke_times)
    x = torch.cat((keystroke_times, condition_out), dim=2)
    x, _ = self.lstm1(x)
    x = self.fc(x)
    x = F.dropout(x, 0.5)
    x = self.model(x)
    x = x.view(batch_size, -1)
    x = self.head(x)

    return self.sigmoid(x)
  