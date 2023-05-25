import torch 
import sys

import torch.nn as nn
import torch.nn.functional as F
from data import *

sys.path.append(f"../masters_thesis")
from analysis.helpers import *
from utils import *

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
generator_output_dim = 4


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
  

class TypeNet(nn.Module, Eops):
  """
  Implementation of the TypeNet with a Triplet Loss (https://arxiv.org/pdf/2101.05570.pdf)
  """
  def __init__(self, window_size: int, interlayer_dropout: float, recurrent_dropout: float):
    super(TypeNet, self).__init__()
    # input size -> [batch_size, 48 (3 time series with the length of window_size), 3 features (keycode, HL, IKI)]
    self.bn1 = nn.BatchNorm1d(window_size)
    self.register_buffer("recurrent_dropout", torch.tensor(recurrent_dropout))
    self.register_buffer("window_size", torch.tensor(window_size))
    self.lstm1 = nn.LSTM(input_size=3, hidden_size=128, num_layers=1, batch_first=True)
    self.interlayer_dropout = nn.Dropout(p=interlayer_dropout)
    self.bn2 = nn.BatchNorm1d(128)
    self.lstm2 = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True)
    print(self.num_params())

  def lstm_forward(self, layer, x):
    _, time_steps, _ = x.size()
    hx = torch.randn(1, 128)
    cx = torch.randn(1, 128)
    output = []
    for i in range(time_steps):
      out, (hx, cx) = layer(x[:, i], (hx, cx))
      hx, cx = F.dropout(hx, p=self.recurrent_dropout), F.dropout(cx, p=self.recurrent_dropout)  # recurrent dropout
      output.append(out)
    
    output = torch.stack(output, dim=0)
    return output, (hx, cx)

  def single_forward(self, x):
    x = self.bn1(x)
    x, _ = self.lstm_forward(self.lstm1, x)
    x = self.interlayer_dropout(x)
    x = self.bn2(x)
    x, _ = self.lstm_forward(self.lstm2, x)
    return x

  def forward(self, x, calculate_loss: bool = False):
    """
    Triplet loss will be used -> the model will return 3 outputs
    A triplet is composed by three different samples from two different classes: 
    Anchor (A) and Positive (P) are different keystroke sequences from the same subject, 
    and Negative (N) is a keystroke sequence from a different subject
    """
    data1, data2, data3 = torch.split(tensor=x, split_size_or_sections=16, dim=1)

    anchor = self.single_forward(data1)
    positive = self.single_forward(data2)
    negative = self.single_forward(data3)

    if not calculate_loss:
      loss = None
    else:
      criterion = TripletLoss(margin=1.5)
      loss = criterion(anchor=anchor, positive=positive, negative=negative)

    return anchor, positive, negative, loss
