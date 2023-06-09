import torch 
import sys

import torch.nn as nn
import torch.nn.functional as F

from models.utils import *


class TypeNet(nn.Module, Eops):
  """
  Implementation of the TypeNet with a Triplet Loss (https://arxiv.org/pdf/2101.05570.pdf)
  """
  def __init__(self, window_size: int, interlayer_dropout: float, recurrent_dropout: float, input_size: int = 3):
    super(TypeNet, self).__init__()
    # input size -> [batch_size, 48 (3 time series with the length of window_size), 3 features (keycode, HL, IKI)]
    self.bn1 = nn.BatchNorm1d(window_size)
    self.register_buffer("recurrent_dropout", torch.tensor(recurrent_dropout))
    self.register_buffer("window_size", torch.tensor(window_size))
    self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=128, num_layers=1, batch_first=True)
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
    x = x.permute(0, 2, 1)
    x = self.bn2(x)
    x, _ = self.lstm_forward(self.lstm2, x.permute(0, 2, 1))
    return x

  def forward(self, anchor, positive, negative = None, calculate_loss: bool = True):
    """
    Triplet loss will be used -> the model will return 3 outputs
    A triplet is composed by three different samples from two different classes: 
    Anchor (A) and Positive (P) are different keystroke sequences from the same subject, 
    and Negative (N) is a keystroke sequence from a different subject
    """
    anchor = self.single_forward(anchor)
    positive = self.single_forward(positive)

    if negative is not None:
      negative = self.single_forward(negative)

    if calculate_loss:
      criterion = TripletLoss(margin=1.5)
      losses = criterion(anchor=anchor, positive=positive, negative=negative)

    return anchor, positive, negative, losses


class Classificator(TypeNet):
  def __init__(self, window_size: int, interlayer_dropout: float, recurrent_dropout: float, input_size: int = 3):
    super(Classificator, self).__init__(window_size=window_size, interlayer_dropout=interlayer_dropout, recurrent_dropout=recurrent_dropout)
    self.ln1 = nn.Linear(in_features=128, out_features=512)
    self.dropout = nn.Dropout(p=0.25)
    self.ln_head = nn.Linear(in_features=512, out_features=1)
  
  def forward(self, x):
    x = self.single_forward(x)
    x = self.ln1(x)
    x = self.dropout(x)
    x = self.ln_head(x)
    return F.sigmoid(x)

  