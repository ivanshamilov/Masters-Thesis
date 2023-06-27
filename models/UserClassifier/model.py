import torch 
import sys

import torch.nn as nn
import torch.nn.functional as F

from models.TypeNet.model import TypeNet


class UserClassifier(TypeNet):
  def __init__(self, window_size: int, interlayer_dropout: float, recurrent_dropout: float):
    super(UserClassifier, self).__init__(window_size=window_size, interlayer_dropout=interlayer_dropout, recurrent_dropout=recurrent_dropout)
    self.flatten = nn.Flatten(start_dim=1)
    self.ln1 = nn.Linear(in_features=16 * 128, out_features=512)
    self.dropout1 = nn.Dropout(0.25)
    self.ln2 = nn.Linear(in_features=512, out_features=256)
    self.dropout2 = nn.Dropout(0.5)
    self.ln_head = nn.Linear(in_features=256, out_features=1)
    self.sigmoid = nn.Sigmoid()
  
  def forward(self, x):
    x = self.single_forward(x)
    x = self.flatten(x)
    x = F.relu(self.ln1(x))
    x = self.dropout1(x)
    x = F.relu(self.ln2(x))
    x = self.dropout2(x)
    x = self.ln_head(x)
    return self.sigmoid(x)
