import torch
import torch.nn as nn
from torchaudio.models import Conformer

VOCAB = " abcdefghijklmnopqrstuvwxyz,.'"

class Recognizer(nn.Module):
  def __init__(self):
    super().__init__()

    out_c = 16
    self.conv = nn.Sequential(
      nn.Conv2d(1, out_c, kernel_size=(3,3), stride=(2,2)),
      nn.ReLU(),
      nn.Conv2d(out_c, out_c, kernel_size=(3,3), stride=(2,2)),
      nn.ReLU()
    )

    H = 32 # time
    self.linear = nn.Sequential(
      nn.Linear(out_c*(((80 - 1) // 2 - 1) // 2), H),
      nn.Dropout(0.2)
    )

    self.conformer = Conformer(
      input_dim=H, num_heads=4, ffn_dim=H*4,
      num_layers=8, depthwise_conv_kernel_size=17
    )

    self.decode = nn.Sequential(
      nn.Dropout(0.5),
      nn.Linear(H, len(VOCAB))
    )

  def forward(self, x, y):
    if self.is_cuda:
      x = x.cuda()
    x = self.encode(x)
    x = self.linear(x)
    x, z = self.conformer(x, y)
    x = self.decode(x).reshape(x.shape[0], x.shape[1], len(VOCAB))
    return torch.nn.functional.softmax(x, dim=2), z

  def encode(self, x):
    x = self.conv(x)
    # At this point x should have shape
    # (batch, time, channels, freq)
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(x.shape[0], x.shape[1], -1)

  @property
  def is_cuda(self):
    return next(self.parameters()).is_cuda


if __name__ == '__main__':
  pass
