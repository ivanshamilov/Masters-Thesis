import torch 
import sys

import torch.nn as nn
import torch.nn.functional as F

from typing import Union

from models.TypeNet.model import TypeNet


class Classificator(TypeNet):
  def __init__(self, window_size: int, interlayer_dropout: float, recurrent_dropout: float, input_size: int = 3):
    super(Classificator, self).__init__(window_size=window_size, interlayer_dropout=interlayer_dropout, recurrent_dropout=recurrent_dropout)
    self.flatten = nn.Flatten(start_dim=1)
    self.ln1 = nn.Linear(in_features=16 * 128, out_features=512)
    self.dropout = nn.Dropout(p=0.25)
    self.ln_head = nn.Linear(in_features=512, out_features=1)
    self.sigmoid = nn.Sigmoid()
  
  def forward(self, x):
    x = self.single_forward(x)
    x = self.flatten(x)
    x = self.ln1(x)
    x = self.dropout(x)
    x = self.ln_head(x)
    return self.sigmoid(x)
