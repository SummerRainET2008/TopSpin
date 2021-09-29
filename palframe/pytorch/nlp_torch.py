#coding: utf8
#author: Tian Xia 

from palframe import *
from palframe.nlp import Logger
from palframe import nlp
from torch.nn import functional as func
from palframe.pytorch import *
import torch

def nan_tensor():
  return torch.tensor(float("NaN"))

def uniform_sample(probs: torch.Tensor):
  samples = func.gumbel_softmax(torch.log(probs), tau=1, hard=True)
  return torch.argmax(samples, -1)

def isabnormal(x: torch.Tensor):
  return not nlp.is_none_or_empty(x) and \
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
  ind = [list(reversed(range(0, length))) + list(range(length, max_len))
         for length in lengths]
  ind = torch.LongTensor(ind)
  for dim in range(2, inputs.dim()):
    ind = ind.unsqueeze(dim)
  ind = ind.expand_as(inputs)
  ind = ind.to(inputs.device)
  reversed_inputs = torch.gather(inputs, 1, ind)

  return reversed_inputs

def sequence_mask(real_len: torch.Tensor,
                  max_size: typing.Union[int, None]=None,
                  key_padding_mask: bool=False):
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

def display_model_parameters(model: nn.Module)-> int:
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
    x, real_len, batch_first=True, enforce_sorted=False,
  )
  output, _ = layer(x1, hidden)
  output, _ = torch.nn.utils.rnn.pad_packed_sequence(
    output, batch_first=True, total_length=x.size(1)
  )

  return output

def one_hot(batch, label_num, y, value=1):
  '''
  :param y: [batch]
  '''
  y_onehot = torch.zeros(batch, label_num, device=y.device)
  y_onehot.scatter_(1, y.unsqueeze(1), value)

  return y_onehot

def label_smoothing_loss(logits, labels, smoothing_factor,
                         reduction="batchmean"):
  '''
  :param logits:  [batch, label_num]
  :param labels:  [batch]
  :param smoothing_factor: float, 0 means no smoothing.
  :param reduction: batchmean, sum, mean, none
  :return:
  '''
  input = torch.log_softmax(logits, 1)
  label_num = logits.size(1)
  smoothing_value = smoothing_factor / (label_num - 1)
  labels_prob1 = smoothing_value * torch.ones(logits.size(0), logits.size(1),
                                              device=logits.device)
  labels_prob2 = one_hot(logits.size(0), logits.size(1), labels,
                         (1 - smoothing_factor) - smoothing_value)
  labels_prob = labels_prob1 + labels_prob2

  loss = torch.nn.functional.kl_div(input, labels_prob, reduction=reduction)

  return loss

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
               activation: nn.Module=Gelu(),
               dropout=0):
    super(FFN, self).__init__()
    self._layer = nn.Sequential(
      torch.nn.Linear(input_dim, hidden_dim),
      activation,
      torch.nn.Dropout(dropout),
      torch.nn.Linear(hidden_dim, output_dim)
    )

  def forward(self, x: torch.Tensor):
    return self._layer(x)

class Dense(nn.Module):
  def __init__(self, linear_layer: nn.Linear, activation=nn.LeakyReLU()):
    super(Dense, self).__init__()
    if activation is None:
      self._layer = linear_layer
    else:
      self._layer = nn.Sequential(
        linear_layer, activation
      )

  def forward(self, x: torch.Tensor):
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
    self._multihead_att = nn.MultiheadAttention(
      dim, multihead, dropout=dropout
    )

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
    values = values.transpose(0, 1) # delete a ','
    output, weights = self._multihead_att(
      query, values, values, key_padding_mask=key_padding_mask
    )
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

# google's style
class RNNEncoder1(nn.Module):
  def __init__(self,
               layer_num,
               input_dim,
               hidden_dim,
               out_dropout: float=0):
    super(RNNEncoder1, self).__init__()

    self._input_dense = Dense(nn.Linear(input_dim, hidden_dim))

    for layer_id in range(layer_num):
      if layer_id == 0:
        self._layer_0_left = nn.GRU(
          input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1,
          batch_first=True, bidirectional=False,
        )
        self._layer_0_right = nn.GRU(
          input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1,
          batch_first=True, bidirectional=False,
        )

      else:
        layer = nn.GRU(
          input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1,
          batch_first=True, bidirectional=False
        )
        self.add_module(f"_layer_{layer_id}", layer)

      self.add_module(f"_pre_process_{layer_id}", nn.LayerNorm(hidden_dim))
      self.add_module(f"_post_process_{layer_id}", nn.Dropout(out_dropout))

    self._combine_dense = Dense(nn.Linear(2 * hidden_dim, hidden_dim))
    self._out_dropout = nn.Dropout(out_dropout)

    self._layer_num = layer_num
    self._hidden_dim = hidden_dim

  def _init_hidden(self, batch_size):
    weight = next(self.parameters())
    hiddens = []
    hiddens.append([
      weight.new_zeros(1, batch_size, self._hidden_dim),
      weight.new_zeros(1, batch_size, self._hidden_dim),
    ])
    for _ in range(1, self._layer_num):
      hiddens.append(weight.new_zeros(1, batch_size, self._hidden_dim))

    return hiddens

  def forward(self, x, real_len, packed_input=False):
    '''
    x:        [batch, max-seq, emb-dim]
    real_len: [batch]
    '''
    hiddens = self._init_hidden(x.size(0))
    x = self._input_dense(x)
    for layer_id in range(self._layer_num):
      x = getattr(self, f"_pre_process_{layer_id}")(x)

      if layer_id == 0:
        output_left = run_rnn_seq(
          self._layer_0_left,
          x,
          real_len, hiddens[0][0], packed_input
        )
        output_right = run_rnn_seq(
          self._layer_0_right,
          reverse_sequence(x, real_len),
          real_len, hiddens[0][1], packed_input
        )

        output_right = reverse_sequence(output_right, real_len)
        output = self._combine_dense(torch.cat([output_left, output_right], 2))

      else:
        output = run_rnn_seq(
          getattr(self, f"_layer_{layer_id}"),
          x, real_len, hiddens[layer_id], packed_input
        )

      x = getattr(self, f"_post_process_{layer_id}")(output + x)

    return x

class ResidualGRU(nn.Module):
  def __init__(self, hidden_size, dropout=0.1, num_layers=2):
    super(ResidualGRU, self).__init__()
    self._enc_layer = nn.GRU(
      input_size=hidden_size, hidden_size=hidden_size // 2,
      num_layers=num_layers, batch_first=True, dropout=dropout,
      bidirectional=True
    )
    self._out_norm = nn.LayerNorm(hidden_size)

  def forward(self, input):
    output, _ = self._enc_layer(input)
    return self._out_norm(output + input)

class VallinaDecoder(nn.Module):
  def __init__(self,
               emb_dim,
               hidden_dim,
               enc_outputs_dim,
               out_dropout):
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
    output = torch.tanh(
      self._x_dense(x) + attn_enc + hidden
    )
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
      )
      for kernel in kernels
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

#todo: not finished. Do NOT use it.
class SLSTM(nn.Module):
  def __init__(self, dim_input, dim_hidden, window_size,
               num_sent_nodes=1, bias=True, init_method='normal'):
    super(SLSTM, self).__init__()
    self._dim_input = dim_input
    self._dim_hidden = dim_hidden
    self._window_size = window_size
    self._num_sent_nodes = num_sent_nodes
    self._init_name = init_method
    self._add_bias = bias
    self._ptanh = PenalizedTanh()
    self._all_gate_weights = []

    # define parameters for word nodes
    word_gate_dict = dict([('input_gate', 'i'),
                           ('left_forget_gate', 'l'),
                           ('right_forget_gate', 'r'),
                           ('forget_gate', 'f'),
                           ('sentence_forget_gate', 's'),
                           ('output_gate', 'o'),
                           ('recurrent_input', 'u')])

    for (gate_name, gate_tag) in word_gate_dict.items():
      # weight: (out_features, in_features)
      w_w = nn.Parameter(torch.Tensor(dim_hidden,
                                      (window_size * 2 + 1) * dim_hidden))
      w_u = nn.Parameter(torch.Tensor(dim_hidden, dim_input))
      w_v = nn.Parameter(torch.Tensor(dim_hidden, dim_hidden))
      w_b = nn.Parameter(torch.Tensor(dim_hidden))

      gate_params = (w_w, w_u, w_v, w_b)
      param_names = ['w_w{}', 'w_u{}', 'w_v{}', 'w_b{}']
      param_names = [x.format(gate_tag) for x in param_names]
      for name, param in zip(param_names, gate_params):
        setattr(self, name, param)  # self.w_w{i} = w_w
      self._all_gate_weights.append(param_names)

    # define parameters for sentence node
    sentence_gate_dict = dict([('sentence_forget_gate', 'g'),
                               ('word_forget_gate', 'f'),
                               ('output_gate', 'o')])
    for (gate_name, gate_tag) in sentence_gate_dict.items():
      # weight: (out_features, in_features)
      s_w = nn.Parameter(torch.Tensor(dim_hidden, dim_hidden))
      s_u = nn.Parameter(torch.Tensor(dim_hidden, dim_hidden))
      s_b = nn.Parameter(torch.Tensor(dim_hidden))
      gate_params = (s_w, s_u, s_b)
      param_names = ['s_w{}', 's_u{}', 's_b{}']
      param_names = [x.format(gate_tag) for x in param_names]
      for name, param in zip(param_names, gate_params):
        setattr(self, name, param)  # self.s_w{i} = s_w
      self._all_gate_weights.append(param_names)

    self._init_params(self._init_name)

  def _init_params(self, init_method):
    if init_method == 'normal':
      std = 0.1
      for weight in self.parameters():
        weight.data.normal_(mean=0.0, std=std)

    else:  # uniform: make std of weights as 0
      stdv = 1.0 / math.sqrt(self._dim_hidden)
      for weight in self.parameters():
        weight.data.uniform_(-stdv, stdv)

  def _in_window_context(self, hx, window_size=1):
    '''
    Args: hx: (l,b,d)
    Returns: (l,b,hidden*(window*2+1)
    '''
    slices = torch.unbind(hx, dim=0)
    zeros = torch.unbind(torch.zeros_like(hx), dim=0)

    context_left = [torch.stack(zeros[:i] + slices[: -i], dim=0)
                    for i in range(window_size, 0, -1)]
    context_left.append(hx)
    context_right = [torch.stack(slices[i + 1:] + zeros[: i + 1], dim=0)
                     for i in range(0, window_size)]

    context = context_left + context_right
    # context is a list of length window size*2+1,
    # every element covering different part of original sent,
    # every element in context is of same length
    return torch.cat(context, dim=2)

  def forward(self, src_seq, src_len, state=None):
    '''
    src_seq: (b, len, d)
    src_len: (b)
    state = (h_0, c_0)
    h_0: (seq_len+sentence_nodes, batch, hidden_size)
    c_0: (seq_len+sentence_nodes, batch, hidden_size)

    Returns: (h_1, c_1)
    h_1: (seq_len+sentence_nodes, batch, hidden_size)
    c_1: (seq_len+sentence_nodes, batch, hidden_size)
    '''
    #todo: should set initial state here.
    seq_mask = nlp_torch.sequence_mask(src_len, src_seq.size(1))
    seq_mask = seq_mask.transpose(0, 1).unsqueeze(2).logical_not()
    # sent node is in the end
    prev_h_gt = state[0][-self._num_sent_nodes:]
    #（l,b,d)
    prev_h_wt = state[0][:-self._num_sent_nodes].masked_fill(seq_mask, 0)
    prev_c_gt = state[1][-self._num_sent_nodes:]
    prev_c_wt = state[1][:-self._num_sent_nodes].masked_fill(seq_mask, 0)

    # update sentence node
    h_hat = prev_h_wt.mean(dim=0)
    fg = func.sigmoid(func.linear(prev_h_gt, self.s_wg) +
                      func.linear(h_hat, self.s_ug) + self.s_bg)
    output_gate = func.sigmoid(func.linear(prev_h_gt, self.s_wo) +
                               func.linear(h_hat, self.s_uo) + self.s_bo)
    fi = func.sigmoid(func.linear(prev_h_gt, self.s_wf) +
                      func.linear(prev_h_wt, self.s_uf) +
                      self.s_bf).masked_fill(seq_mask, -1e25)
    fi_normalized = func.softmax(fi, dim=0)
    c_gt = fg.mul(prev_c_gt).add(fi_normalized.mul(prev_c_wt).sum(dim=0))
    h_gt = output_gate.mul(func.tanh(c_gt))

    # update word nodes
    epsilon = self._in_window_context(prev_h_wt, window_size=self._window_size)
    # epsilon: (l, b, d_word or emb_size * (2 * window_size + 1)
    input_gate = func.sigmoid(
      func.linear(epsilon, self.w_wi) +
      func.linear(src_seq, self.w_ui) +
      func.linear(prev_h_gt, self.w_vi) + self.w_bi
    )
    left_gate = func.sigmoid(
      func.linear(epsilon, self.w_wl) +
      func.linear(src_seq, self.w_ul) +
      func.linear(prev_h_gt, self.w_vl) + self.w_bl
    )
    right_gate = func.sigmoid(
      func.linear(epsilon, self.w_wr) +
      func.linear(src_seq, self.w_ur) +
      func.linear(prev_h_gt, self.w_vr) + self.w_br
    )
    forget_gate = func.sigmoid(
      func.linear(epsilon, self.w_wf) +
      func.linear(src_seq, self.w_uf) +
      func.linear(prev_h_gt, self.w_vf) + self.w_bf
    )
    sent_gate = func.sigmoid(
      func.linear(epsilon, self.w_ws) +
      func.linear(src_seq, self.w_us) +
      func.linear(prev_h_gt, self.w_vs) + self.w_bs
    )
    output_gate = func.sigmoid(
      func.linear(epsilon, self.w_wo) +
      func.linear(src_seq, self.w_uo) +
      func.linear(prev_h_gt, self.w_vo) + self.w_bo
    )
    current_update = func.tanh(
      func.linear(epsilon, self.w_wu) +
      func.linear(src_seq, self.w_uu) +
      func.linear(prev_h_gt, self.w_vu) + self.w_bu
    )

    gates = torch.stack(
      (left_gate, forget_gate, right_gate, sent_gate, input_gate), dim=0
    )
    # gates: (5*l,b,d)
    gates_normalized = func.softmax(gates.masked_fill(seq_mask, -1e25), dim=0)

    c_wt_l, prev_c_wt, c_wt_r = \
      self._in_window_context(prev_c_wt).chunk(3, dim=2) # split by dim 2
    # c_wt_: (l, b, d_word)
    c_mergered = torch.stack(
      (c_wt_l, prev_c_wt, c_wt_r, prev_c_gt.expand_as(prev_c_wt.data),
       current_update),
      dim=0
    )

    c_wt = gates_normalized.mul(c_mergered).sum(dim=0)
    c_wt = c_wt.masked_fill(seq_mask, 0)
    h_wt = output_gate.mul(func.tanh(c_wt))

    h_t = torch.cat((h_wt, h_gt), dim=0)
    c_t = torch.cat((c_wt, c_gt), dim=0)

    return h_t, c_t

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
               out_dropout: float=0):
    super(TranformerEncoder, self).__init__()

    self._layer_num = layer_num
    self._input_dense = Dense(nn.Linear(input_dim + pos_emb_dim, hidden_dim))
    self._pos_emb_dim = pos_emb_dim
    self._first_token = torch.nn.Parameter(torch.Tensor(input_dim))
    self._add_first_token = add_first_token
    assert pos_emb_dim > 0
    self._pos_emb_weight = torch.nn.Parameter(
      torch.Tensor(self.MAX_LEN + 1, pos_emb_dim)
    )

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

    pos_emb = self._pos_emb_weight[: x.size(1)]
    pos_emb = pos_emb.unsqueeze(0).expand(
      [x.size(0), x.size(1), pos_emb.size(1)]
    )
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
