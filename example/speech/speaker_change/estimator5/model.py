#coding: utf8
#author: Xinyi Wu

from example.speech.speaker_change.estimator5 import *


class ConvBNBlock(nn.Module):
  def __init__(self,
               in_channels,
               out_channels,
               kernel_size,
               stride=1,
               padding=None,
               nonlinear=None,
               dropout_ratio=0.5):
    super(ConvBNBlock, self).__init__()
    assert (kernel_size - 1) % 2 == 0
    conv1d = nn.Conv1d(in_channels,
                       out_channels,
                       kernel_size,
                       stride=stride,
                       padding=padding)
    norm = nn.BatchNorm1d(out_channels)
    dropout = nn.Dropout(dropout_ratio)
    if nonlinear == 'relu':
      self.net = nn.Sequential(conv1d, norm, nn.ReLU(), dropout)
    elif nonlinear == 'tanh':
      self.net = nn.Sequential(conv1d, norm, nn.Tanh(), dropout)
    elif nonlinear == 'LeakyReLU':
      self.net = nn.Sequential(conv1d, norm, nn.LeakyReLU(), dropout)
    else:
      self.net = nn.Sequential(conv1d, norm, dropout)

  def forward(self, x):
    output = self.net(x)
    return output


class SCDetConv(nn.Module):
  """
  Speaker Change Detector classifier
  """
  def __init__(self, param):
    """
    Args:
      input_dim (int): input feature dim
      input_win (int): input window length
      conv_sizes (list int): sizes of conv layers
      linear_sizes (list int): sizes of linear/dense layers
      strides (list int): sizes of stride for layers
      kernel_sizes (list int): kernel sizes for each conv layer
    """
    super(SCDetConv, self).__init__()
    self.param = param

    self.convolutions = nn.ModuleList()
    conv_sizes = [param.input_dim] + param.conv_sizes
    conv_out_len = param.input_win

    for i in range(len(conv_sizes) - 1):
      # TODO: padding size can be different if stride != 1
      # pad_size = (param.kernel_sizes[i] - 1) // 2
      pad_size = (param.strides[i] * (conv_out_len - 1) - conv_out_len +
                  param.kernel_sizes[i]) // 2
      self.convolutions.append(
          ConvBNBlock(conv_sizes[i],
                      conv_sizes[i + 1],
                      kernel_size=param.kernel_sizes[i],
                      stride=param.strides[i],
                      padding=pad_size,
                      nonlinear=param.activation))
      # Work out the len of time axis of conv layer output
      conv_out_len = int((conv_out_len + 2 * pad_size -
                          param.kernel_sizes[i]) / param.strides[i] + 1)

    self.linears = nn.ModuleList()
    linear_sizes = [conv_sizes[-1] * conv_out_len] + param.linear_sizes + [1]
    for i in range(len(linear_sizes) - 1):
      self.linears.append(
          nn.Linear(linear_sizes[i], linear_sizes[i + 1], bias=True))
    self._init_weights()

  def _init_weights(self):
    for name, w in self.named_parameters():
      if "weight" in name:
        if len(w.size()) > 1:
          nn.init.xavier_normal_(w)
        else:
          nn.init.normal_(w)
      elif "bias" in name:
        nn.init.zeros_(w)
      else:
        Logger.warn(f"Unintialized '{name}'")

  def forward(self, x):
    """
    Args:
      x: B x T x D
    Return:
      x: B x 1
    """
    x = x.transpose(1, 2)
    for conv in self.convolutions:
      x = conv(x)
    x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
    for line in self.linears:
      x = line(x)
    return x
