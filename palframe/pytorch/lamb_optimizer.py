#!/usr/bin/env python
# coding: utf8
# author: Zongcheng Ji 
# date: 2020/10/27


import torch
from torch.optim.optimizer import Optimizer
from typing import Callable, Iterable, Tuple


class LambOptimizer(Optimizer):
  """Implements Lamb algorithm.

  It has been proposed in `Large Batch Optimization for Deep Learning:
  Training BERT in 76 minutes <https://arxiv.org/abs/1904.00962>`.

  Arguments:
      params: iterable of parameters to optimize or dicts defining
          parameter groups
      lr: learning rate (default: 1e-3)
      betas: coefficients used for computing
          running averages of gradient and its square (default: (0.9, 0.999))
      eps: term added to the denominator to improve
          numerical stability (default: 1e-8)
      weight_decay: decoupled weight decay (default: 0.0)
      clamp_value: clamp weight_norm in (0, clamp_value) (default: 10.0)
          set to a high value to avoid it (e.g 10e3)
      correct_bias: whether or not to correct bias in Adam (default: True)

  Example:
      # >>> optimizer = lamb_optimizer.Lamb(model.parameters())
      # >>> optimizer.zero_grad()
      # >>> loss_fn(model(input), target).backward()
      # >>> optimizer.step()

  Note:
      Reference codes in PyTorch:
      https://github.com/skyday123/pytorch-lamb
      https://github.com/jettify/pytorch-optimizer/blob/master/torch_optimizer/lamb.py

      Reference code in Tensorflow (the url is from Page 1 of the paper):
      https://github.com/tensorflow/addons/blob/master/tensorflow_addons/optimizers/lamb.py

      Reference articals:
      https://towardsdatascience.com/an-intuitive-understanding-of-the-lamb-optimizer-46f8c0ae4866
      https://developer.nvidia.com/blog/pretraining-bert-with-layer-wise-adaptive-learning-rates/
  """

  def __init__(
    self,
    params: Iterable[torch.nn.parameter.Parameter],
    lr: float=1e-3,
    betas: Tuple[float, float]=(0.9, 0.999),
    eps: float=1e-6,
    weight_decay: float=0.0,
    clamp_value: float=10.0,
    correct_bias: bool=True,
  ):
    if not 0.0 <= lr:
      raise ValueError('Invalid learning rate: {}'.format(lr))
    if not 0.0 <= betas[0] < 1.0:
      raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
    if not 0.0 <= betas[1] < 1.0:
      raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
    if not 0.0 <= eps:
      raise ValueError('Invalid epsilon value: {}'.format(eps))
    if not 0.0 <= weight_decay:
      raise ValueError('Invalid weight_decay value: {}'.format(weight_decay))
    if not 0.0 <= clamp_value:
      raise ValueError('Invalid clamp value: {}'.format(clamp_value))
    defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                    clamp_value=clamp_value, correct_bias=correct_bias)
    super(LambOptimizer, self).__init__(params, defaults)

  def step(self, closure: Callable=None):
    """Performs a single optimization step.

    Arguments:
      closure (callable, optional): A closure that reevaluates the model
        and returns the loss.
    """
    loss = None
    if closure is not None:
      loss = closure()

    for group in self.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue
        grad = p.grad.data
        if grad.is_sparse:
          raise RuntimeError('Lamb does not support sparse gradients, '
                             'please consider SparseAdam instead')

        state = self.state[p]

        # State initialization
        if len(state) == 0:
          state['step'] = 0
          # Exponential moving average of gradient values
          state['exp_avg'] = torch.zeros_like(p.data)
          # Exponential moving average of squared gradient values
          state['exp_avg_sq'] = torch.zeros_like(p.data)

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        beta1, beta2 = group['betas']

        state['step'] += 1

        # Decay the first and second moment running average coefficient
        # In-place operations to update the averages at the same time
        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

        # get "signal-to-noise ratio" (i.e., adam step) to adapte updates
        # The LAMB algorithm from paper v4 adds the use of bias-correction.
        # Appendix E of paper v4/v5 says "We can remove adam-correction from LAMB."
        # NVLAMB suggests to use bias-correction.
        if group['correct_bias']:
          bias_correction1 = 1.0 - beta1 ** state['step']
          bias_correction2 = 1.0 - beta2 ** state['step']
          exp_avg_hat = exp_avg / bias_correction1
          exp_avg_sq_hat = exp_avg_sq / bias_correction2
          adam_step = exp_avg_hat / (exp_avg_sq_hat.sqrt().add(group['eps']))
        else:
          adam_step = exp_avg / (exp_avg_sq.sqrt().add(group['eps']))

        if group['weight_decay'] > 0.0:
          adam_step.add_(p.data, alpha=group['weight_decay'])

        # get "trust ratio" to adapte learning rates
        weight_norm = torch.norm(p.data).clamp(0.0, group['clamp_value'])
        adam_norm = torch.norm(adam_step)
        trust_ratio = weight_norm / adam_norm \
          if weight_norm > 0.0 and adam_norm > 0.0 else 1.0

        # update weights with "signal-to-noise ratio" (i.e., adam step) and
        # "trust ratio"
        p.data.add_(adam_step, alpha=-group['lr'] * trust_ratio)

    return loss
