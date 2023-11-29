#coding: utf8
#author: Tian Xia

import typing
import torch
import math
from torch import nn
from torch.nn import functional as func
from topspin.tools.helper import Logger, is_none_or_empty


def nan_tensor():
  return torch.tensor(float("NaN"))


def uniform_sample(probs: torch.Tensor):
  samples = func.gumbel_softmax(torch.log(probs), tau=1, hard=True)
  return torch.argmax(samples, -1)


def isabnormal(x: torch.Tensor):
  return not is_none_or_empty(x) and \
         torch.any(torch.logical_or(torch.isinf(x), torch.isnan(x))).item()


def norm_0_1(x: torch.Tensor):
  shape = x.shape
  x = x.view(-1, shape[-1])
  minv = x.min(0).values
  maxv = x.max(0).values
  x = (x - minv) / (maxv - minv)
  x = x.view(*shape)

  return x


def norm_mean(x: torch.Tensor):
  shape = x.shape
  x = x.view(-1, shape[-1])
  meanv = x.mean(0)
  x = x - meanv
  x = x.view(*shape)

  return x


def check_nan(tensor, name):
  ret = torch.any(torch.isnan(tensor))
  if ret:
    Logger.error(f"{name} is_nan")


def reverse_sequence(inputs, lengths):
  batch_size, max_len = inputs.size(0), inputs.size(1)
  ind = [
      list(reversed(range(0, length))) + list(range(length, max_len))
      for length in lengths
  ]
  ind = torch.LongTensor(ind)
  for dim in range(2, inputs.dim()):
    ind = ind.unsqueeze(dim)
  ind = ind.expand_as(inputs)
  ind = ind.to(inputs.device)
  reversed_inputs = torch.gather(inputs, 1, ind)

  return reversed_inputs


def sequence_mask(real_len: torch.Tensor,
                  max_size: typing.Union[int, None] = None,
                  key_padding_mask: bool = False):
  '''
  real_len: [batch]
  return: [batch, max_size]
  '''
  if max_size is None:
    max_size = real_len.max()

  size = real_len.new_tensor(range(1, max_size + 1))
  mask = real_len.unsqueeze(1) >= size.unsqueeze(0)

  if key_padding_mask:
    return torch.logical_not(mask)
  return mask


def display_model_parameters(model: nn.Module) -> int:
  import functools, operator

  Logger.info("-" * 64)
  Logger.info(f"module parameters:")
  total_num = 0
  info = []
  for name, var in model.named_parameters():
    var_num = functools.reduce(operator.mul, var.shape, 1)
    total_num += var_num
    info.append([var_num, name, var])

  info.sort(reverse=True)
  accum_num = 0
  for var_num, name, var in info:
    accum_num += var_num
    Logger.info(f"var: {name}, shape={var.shape}, num={var_num:_}, "
                f"ratio={var_num * 100 / total_num:.2f}%, "
                f"accum num={accum_num:_}, "
                f"accum ratio={accum_num * 100 / total_num:.2f}%")
  Logger.info()
  Logger.info(f"#paramters: {total_num:_}")
  Logger.info("-" * 64)

  return total_num


def run_rnn_seq(layer: torch.nn.Module, x, real_len, hidden,
                packed_input: bool):
  '''
  :param x:  [b, len, dim]
  :param real_len: [b]
  :return:
  '''
  if not packed_input:
    output, _ = layer(x, hidden)
    return output

  x1 = torch.nn.utils.rnn.pack_padded_sequence(
      x,
      real_len,
      batch_first=True,
      enforce_sorted=False,
  )
  output, _ = layer(x1, hidden)
  output, _ = torch.nn.utils.rnn.pad_packed_sequence(output,
                                                     batch_first=True,
                                                     total_length=x.size(1))

  return output


def one_hot(batch, label_num, y, value=1):
  '''
  :param y: [batch]
  '''
  y_onehot = torch.zeros(batch, label_num, device=y.device)
  y_onehot.scatter_(1, y.unsqueeze(1), value)

  return y_onehot


def label_smoothing_loss(logits,
                         labels,
                         smoothing_factor,
                         reduction="batchmean"):
  '''
  :param logits:  [batch, label_num]
  :param labels:  [batch]
  :param smoothing_factor: float, 0 means no smoothing.
  :param reduction: batchmean, sum, mean, none
  :return:
  '''

  assert len(logits.shape) == 2
  assert len(labels.shape) == 1

  input = torch.log_softmax(logits, 1)
  label_num = logits.size(1)
  smoothing_value = smoothing_factor / (label_num - 1)
  labels_prob1 = smoothing_value * torch.ones(
      logits.size(0), logits.size(1), device=logits.device)
  labels_prob2 = one_hot(logits.size(0), logits.size(1), labels,
                         (1 - smoothing_factor) - smoothing_value)
  labels_prob = labels_prob1 + labels_prob2

  loss = torch.nn.functional.kl_div(input, labels_prob, reduction=reduction)

  return loss


def topK_mask_logits(logits, topk: int = -1, acum_prob_limit: float = -1):
  c_num = logits.size(-1)
  dim = len(logits.shape)
  if 0 < topk < c_num:
    topk_indexes = torch.topk(-logits, c_num - topk)[1]
    ret = logits + torch.scatter(logits, dim - 1, topk_indexes, -10000)
    return ret

  elif 0 < acum_prob_limit < 1:
    topk_scores, topk_indexes = torch.topk(logits, c_num)
    topk_probs = nn.Softmax(dim=-1)(topk_scores)
    bottom_mask = torch.cumsum(topk_probs, dim=dim - 1) > acum_prob_limit
    topk_scores = bottom_mask * -10000 + topk_scores
    orignal_indexes = torch.sort(topk_indexes)[1]
    ret = torch.gather(topk_scores, dim - 1, orignal_indexes)
    return ret

  else:
    return logits


def to_device(v, device):
  '''
  :param v: any type
  :param device:
  :return:
  '''
  if isinstance(v, (list, tuple)):
    return [to_device(e, device) for e in v]
  elif isinstance(v, dict):
    return {k: to_device(v, device) for k, v in v.items()}
  elif isinstance(v, torch.Tensor):
    return v.to(device)
  else:
    return v


class FocalLoss(nn.Module):
  def __init__(self, gamma=2):
    super(FocalLoss, self).__init__()
    self._gamma = gamma

  def forward(self, logits, labels):
    '''
    input:  [batch, label_num]
    target: [batch]
    '''
    assert len(logits.shape) == 2
    assert len(labels.shape) == 1

    labels = labels.unsqueeze(-1)
    logpt = func.log_softmax(logits, dim=1)
    logpt = logpt.gather(1, labels).view(-1)
    pt = torch.exp(logpt)

    loss = -1 * (1 - pt)**self._gamma * logpt
    return loss.mean()


class PrePostProcessingWrapper(nn.Module):
  def __init__(self, layer, input_dim, dropout_to_drop_prob=0):
    super(PrePostProcessingWrapper, self).__init__()

    self._layer = layer
    self._layer_norm = nn.LayerNorm(input_dim)
    self._dropout = nn.Dropout(dropout_to_drop_prob)

  def forward(self, x, *args, **kwargs):
    y = self._layer(self._layer_norm(x), *args, **kwargs)
    y = self._dropout(y)

    return y


class PenalizedTanh(torch.nn.Module):
  def forward(self, x):
    return 0.75 * func.relu(func.tanh(x)) + 0.25 * func.tanh(x)


class Swish(nn.Module):
  def forward(self, x):
    return x * torch.sigmoid(x)


class Gelu(nn.Module):
  def forward(self, x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class FFN(nn.Module):
  def __init__(self,
               input_dim,
               hidden_dim,
               output_dim,
               activation: nn.Module = Gelu(),
               dropout=0):
    super(FFN, self).__init__()
    self._layer = nn.Sequential(torch.nn.Linear(input_dim, hidden_dim),
                                activation, torch.nn.Dropout(dropout),
                                torch.nn.Linear(hidden_dim, output_dim))

  def forward(self, x: torch.Tensor):
    return self._layer(x)


class Dense(nn.Module):
  def __init__(self, linear_layer: nn.Linear, activation=nn.LeakyReLU()):
    super(Dense, self).__init__()
    if activation is None:
      self._layer = linear_layer
    else:
      self._layer = nn.Sequential(linear_layer, activation)

  def forward(self, x: torch.Tensor):
    return self._layer(x)


class AttenDense(nn.Module):
  def __init__(self,
               linear_layer: nn.Linear,
               activation=nn.LeakyReLU(),
               max_emb_num=None,
               dropout=0.0):
    super(AttenDense, self).__init__()

    self._in_features = linear_layer.in_features
    if max_emb_num is None:
      self._out_features = linear_layer.out_features
    else:
      self._out_features = min(linear_layer.out_features, max_emb_num)
    self._atten_emb = nn.Parameter(
        torch.Tensor(self._out_features, self._in_features))
    self._dropout = nn.Dropout(dropout)

    if activation is None:
      self._layer = linear_layer
    else:
      self._layer = nn.Sequential(linear_layer, activation)

    self._reset_parameters()

  def _reset_parameters(self):
    self._atten_emb.data.normal_(mean=0.0, std=0.1)

  def forward(self, x: torch.Tensor):
    '''
    :param x: [batch, * ..., dim]
    :return:
    '''
    shape = list(x.shape)
    x = x.reshape(-1, shape[-1])
    scores = torch.einsum("bx,cx->bc", self._dropout(x), self._atten_emb)
    probs = torch.softmax(scores, dim=1)
    atten_x = probs @ self._atten_emb
    x = x + atten_x
    x = x.view(shape)

    return self._layer(x)


class Attention(nn.Module):
  def __init__(self, query_dim, values_dim, atten_dim):
    super(Attention, self).__init__()
    self._query_dense = Dense(nn.Linear(query_dim, atten_dim))
    self._values_dense = Dense(nn.Linear(values_dim, atten_dim))
    self._logit_out = Dense(nn.Linear(atten_dim, 1))

  def forward(self, query, values, real_len):
    '''
    query: [batch, dim]
    values: [batch, len, dim]
    real_len: [batch], IntTensor
    '''
    m = self._values_dense(values)
    m = m + self._query_dense(query).unsqueeze(1).expand_as(m)
    logit = self._logit_out(torch.tanh(m)).squeeze(2)
    key_padding_mask = sequence_mask(real_len, values.shape[1], True)
    logit.data.masked_fill_(key_padding_mask, -1e-9)
    prob = func.softmax(logit, dim=1)
    output = prob.unsqueeze(1).matmul(values).squeeze(1)

    return output


class MultiHeadAttention(nn.Module):
  def __init__(self, dim, multihead=4, dropout=0.0):
    super(MultiHeadAttention, self).__init__()
    self._multihead_att = nn.MultiheadAttention(dim,
                                                multihead,
                                                dropout=dropout)

  def forward(self, query, values, real_len=None):
    '''
    query:    [batch, l, dim]
    values:   [batch, s, dim]
    real_len: [batch]
    Return:
      output: [batch, l, dim]
      weights: [batch, l, s]
    '''
    if real_len is None:
      key_padding_mask = None
    else:
      key_padding_mask = sequence_mask(real_len, values.shape[1], True)

    query = query.transpose(0, 1)
    values = values.transpose(0, 1)  # delete a ','
    output, weights = self._multihead_att(query,
                                          values,
                                          values,
                                          key_padding_mask=key_padding_mask)
    output = output.transpose(0, 1)

    return output, weights


class InnerAttention(nn.Module):
  def __init__(self, dim, atten_dim=512, multihead=4):
    super(InnerAttention, self).__init__()
    self._input_dense = Dense(nn.Linear(dim, atten_dim))
    self._multihead_att = MultiHeadAttention(atten_dim, multihead)
    self._query_weight = nn.Parameter(torch.Tensor(1, 1, atten_dim))
    self._output_dense = Dense(nn.Linear(atten_dim, dim))

    self._reset_parameters()

  def _reset_parameters(self):
    # nn.init.xavier_uniform_(self._query)
    nn.init.uniform_(self._query_weight, 0, 1)

  def forward(self, values, real_len=None):
    '''
    values:   [batch, seq-len, dim]
    real_len: [batch]
    '''
    values = self._input_dense(values)
    query = self._query_weight.expand(values.size(0), 1, -1)
    output = self._multihead_att(query, values, real_len)[0]
    output = self._output_dense(output)

    return output.squeeze(1)


class ResidualGRU(nn.Module):
  def __init__(self, hidden_size, dropout=0.1, num_layers=2):
    super(ResidualGRU, self).__init__()
    self._enc_layer = nn.GRU(input_size=hidden_size,
                             hidden_size=hidden_size // 2,
                             num_layers=num_layers,
                             batch_first=True,
                             dropout=dropout,
                             bidirectional=True)
    self._out_norm = nn.LayerNorm(hidden_size)

  def forward(self, input):
    output, _ = self._enc_layer(input)
    return self._out_norm(output + input)


class VallinaDecoder(nn.Module):
  def __init__(self, emb_dim, hidden_dim, enc_outputs_dim, out_dropout):
    super(VallinaDecoder, self).__init__()

    self._rnn_cell1 = nn.GRUCell(emb_dim, hidden_dim)
    self._enc_attn = Attention(hidden_dim, enc_outputs_dim, enc_outputs_dim)
    self._rnn_cell2 = nn.GRUCell(enc_outputs_dim, hidden_dim)
    self._x_dense = Dense(nn.Linear(emb_dim, hidden_dim))

    self._out_dropout = nn.Dropout(out_dropout)

  def forward(self, x, hidden, enc_outputs, enc_real_len):
    '''
    x:      [batch, emb-dim]
    hidden: [batch, hidden-dim]
    x_mask: [batch, max-seq]
    enc_output: [batch, max-seq, dim]
    '''
    hidden = self._rnn_cell1(x, hidden)
    attn_enc = self._enc_attn(hidden, enc_outputs, enc_real_len)
    hidden = self._rnn_cell2(attn_enc, hidden)
    output = torch.tanh(self._x_dense(x) + attn_enc + hidden)
    output = self._out_dropout(output)

    return output, hidden


class TextCNN(nn.Module):
  def __init__(self,
               kernels: typing.List[int],
               in_channel: int,
               out_channel: int,
               max_seq_len: int,
               dim: int,
               activation=nn.LeakyReLU(),
               dropout=0):
    super(TextCNN, self).__init__()

    cnns = [
        nn.Sequential(
            nn.Conv2d(in_channel, out_channel, (kernel, dim)),
            activation,
            nn.MaxPool2d((max_seq_len - kernel + 1, 1)),
        ) for kernel in kernels
    ]
    self._cnn_num = len(cnns)
    for idx, cnn in enumerate(cnns):
      setattr(self, f"_cnns_{idx}", cnn)

    self._in_channel = in_channel
    self._out_dropout = nn.Dropout(dropout)
    self._output_size = len(kernels) * out_channel

  def forward(self, x):
    '''
    x: [batch, channel, word_num, dim]
    '''
    outs = [getattr(self, f"_cnns_{idx}")(x) for idx in range(self._cnn_num)]
    out = torch.cat(outs, -1)
    out = out.flatten(1, -1)
    assert out.shape[-1] == self._output_size, \
      f"{out.shape} != {self._output_size}"
    out = self._out_dropout(out)

    return out


class TranformerEncoder(nn.Module):
  '''A simplified version'''

  MAX_LEN = 512

  def __init__(self,
               layer_num,
               input_dim,
               hidden_dim,
               multi_head_num,
               pos_emb_dim=0,
               add_first_token=True,
               out_dropout: float = 0):
    super(TranformerEncoder, self).__init__()

    self._layer_num = layer_num
    self._input_dense = Dense(nn.Linear(input_dim + pos_emb_dim, hidden_dim))
    self._pos_emb_dim = pos_emb_dim
    self._first_token = torch.nn.Parameter(torch.Tensor(input_dim))
    self._add_first_token = add_first_token
    assert pos_emb_dim > 0
    self._pos_emb_weight = torch.nn.Parameter(
        torch.Tensor(self.MAX_LEN + 1, pos_emb_dim))

    for _ in range(layer_num):
      setattr(self, f"_preprocess_layernorm_{_}", nn.LayerNorm(hidden_dim))
      setattr(self, f"_postprocess_dropout_{_}", nn.Dropout(out_dropout))
      setattr(self, f"_self_atten_{_}",
              MultiHeadAttention(hidden_dim, multi_head_num, out_dropout))
      setattr(self, f"_fft_{_}",
              FFN(hidden_dim, hidden_dim, hidden_dim, dropout=out_dropout))

    self._reset_parameters()

  def _reset_parameters(self):
    nn.init.uniform_(self._first_token, 0, 1)
    nn.init.uniform_(self._pos_emb_weight, 0, 1)

  def forward(self, x, real_len):
    '''
    :param x: [batch, len, dim]
    :param real_len: [batch]
    :return:
    '''
    assert x.size(1) <= self.MAX_LEN

    if self._add_first_token:
      real_len = real_len + 1
      batch, length, dim = x.shape
      first_toekn = self._first_token.expand(batch, 1, dim)
      x = torch.cat([first_toekn, x], 1)

    pos_emb = self._pos_emb_weight[:x.size(1)]
    pos_emb = pos_emb.unsqueeze(0).expand(
        [x.size(0), x.size(1), pos_emb.size(1)])
    x = torch.cat([x, pos_emb], 2)

    x = self._input_dense(x)
    for _ in range(self._layer_num):
      layernorm = getattr(self, f"_preprocess_layernorm_{_}")
      dropout = getattr(self, f"_postprocess_dropout_{_}")
      self_atten = getattr(self, f"_self_atten_{_}")
      fft = getattr(self, f"_fft_{_}")

      x1 = layernorm(x)
      x2 = x1 + self_atten(x1, x1, real_len)[0]
      x3 = fft(x2) + x2
      x4 = dropout(x3)

      x = x4

    return x


def entropy(logit):
  prob = torch.exp(torch.log_softmax(logit, 1))
  x = -prob * torch.log(prob)
  y = torch.ones(x.size(1), 1, device=x.device)
  z = x @ y
  ret = torch.mean(z)

  return ret
