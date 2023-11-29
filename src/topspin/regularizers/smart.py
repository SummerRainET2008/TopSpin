# -*- coding: utf-8 -*-
# @Time : 2021-3-10 16:19
# @Author : by 周旋
# @File : smart.py
"""
方法来源: SMART:Robust and Efficient Fine-Tuning for Pre-trained Natural
           Language Models through Principled Regularized Optimization
代码从 https://github.com/namisan/mt-dnn 移植过来
"""

import torch
import torch.nn.functional as F


def stable_kl(logit, target, epsilon=1e-6, reduce=True):
  """
    计算kl散度
    :param logit:
    :param target:
    :param epsilon:
    :param reduce:
    :return:
    """
  logit = logit.view(-1, logit.size(-1)).float()
  target = target.view(-1, target.size(-1)).float()
  bs = logit.size(0)
  p = F.log_softmax(logit, 1).exp()
  y = F.log_softmax(target, 1).exp()
  rp = -(1.0 / (p + epsilon) - 1 + epsilon).detach().log()
  ry = -(1.0 / (y + epsilon) - 1 + epsilon).detach().log()
  if reduce:
    return (p * (rp - ry) * 2).sum() / bs
  else:
    return (p * (rp - ry) * 2).sum()


def generate_noise(embed, mask, epsilon=1e-5):
  noise = embed.data.new(embed.size()).normal_(0, 1) * epsilon
  noise.detach()
  noise.requires_grad_()
  return noise


class SmartPerturbation:
  """
    方法来源: SMART:Robust and Efficient Fine-Tuning for Pre-trained Natural
           Language Models through Principled Regularized Optimization
    """
  def __init__(self,
               epsilon=1e-6,
               step_size=1e-3,
               noise_var=1e-5,
               norm_p='inf',
               k=1,
               encoder_type=EncoderModelType.BERT,
               loss_map=[],
               norm_level=0):
    """

        :param epsilon: 输入x_i在范数norm_p下的变化半径
        :param step_size: 求解max优化对应的步长，论文中对应eta
        :param noise_var: 求解max优化对应初始值的高斯分布方差
        :param norm_p: 求解max对应的||\tilda{x_i}-x_i||对应的范数类型，默认为无穷范数
        :param k: 求解max需要迭代的步数
        :param encoder_type:
        :param loss_map:
        :param norm_level:
        """
    super(SmartPerturbation, self).__init__()
    self.epsilon = epsilon
    # eta
    self.step_size = step_size  #
    self.K = k
    # sigma
    self.noise_var = noise_var
    self.norm_p = norm_p
    self.encoder_type = encoder_type
    self.loss_map = loss_map
    self.norm_level = norm_level > 0
    assert len(loss_map) > 0

  def _norm_grad(self, grad, eff_grad=None, sentence_level=False):
    """

        :param grad:  v_i + g_i*eta
        :param eff_grad:  g_i*eta
        :param sentence_level:
        :return:
        """
    if self.norm_p == 'l2':
      if sentence_level:
        direction = grad / (torch.norm(grad, dim=(-2, -1), keepdim=True) +
                            self.epsilon)
      else:
        direction = grad / (torch.norm(grad, dim=-1, keepdim=True) +
                            self.epsilon)
    elif self.norm_p == 'l1':
      direction = grad.sign()
    else:
      if sentence_level:
        direction = grad / (grad.abs().max(
            (-2, -1), keepdim=True)[0] + self.epsilon)
      else:
        direction = grad / (grad.abs().max(-1, keepdim=True)[0] + self.epsilon)
        eff_direction = eff_grad / (grad.abs().max(-1, keepdim=True)[0] +
                                    self.epsilon)
    return direction, eff_direction

  def forward(self,
              model,
              logits,
              input_ids,
              token_type_ids,
              attention_mask,
              premise_mask=None,
              hyp_mask=None,
              task_id=0,
              task_type='Classification',
              pairwise=1):
    """

        :param model:
        :param logits:
        :param input_ids:
        :param token_type_ids:
        :param attention_mask:
        :param premise_mask:
        :param hyp_mask:
        :param task_id:
        :param task_type: 下游任务的分类，Classification，Regression，Ranking
        :param pairwise:
        :return:
        """
    # adv training
    vat_args = [
        input_ids, token_type_ids, attention_mask, premise_mask, hyp_mask,
        task_id, 1
    ]

    # init delta
    embed = model(*vat_args)  # 返回模型的embedding层结果，这里代码中实现的是三个embedding的求和
    # 得到 \tilda{embed}
    noise = generate_noise(embed, attention_mask, epsilon=self.noise_var)
    for step in range(0, self.K):
      vat_args = [
          input_ids, token_type_ids, attention_mask, premise_mask, hyp_mask,
          task_id, 2, embed + noise
      ]
      # 计算加入噪音的模型logits输出
      adv_logits = model(*vat_args)
      if task_type == 'Regression':
        adv_loss = F.mse_loss(adv_logits, logits.detach(), reduction='sum')
      else:
        if task_type == 'Ranking':
          adv_logits = adv_logits.view(-1, pairwise)
        # adv_embed 对应的交叉熵, 这里的logits为当前的模型的结果
        adv_loss = stable_kl(adv_logits, logits.detach(), reduce=False)
      # 计算adv_loss对于embed+ noise的导数，注意这里由于扰动是noise，所以直接计算noise的导数是一样的
      delta_grad, = torch.autograd.grad(adv_loss,
                                        noise,
                                        only_inputs=True,
                                        retain_graph=False)

      norm = delta_grad.norm()
      if (torch.isnan(norm) or torch.isinf(norm)):
        return 0
      #
      eff_delta_grad = delta_grad * self.step_size  # g_i*eta
      delta_grad = noise + delta_grad * self.step_size  # v_i + g_i*eta
      #
      noise, eff_noise = self._norm_grad(delta_grad,
                                         eff_grad=eff_delta_grad,
                                         sentence_level=self.norm_level)
      noise = noise.detach()
      noise.requires_grad_()
    vat_args = [
        input_ids, token_type_ids, attention_mask, premise_mask, hyp_mask,
        task_id, 2, embed + noise
    ]
    adv_logits = model(*vat_args)
    if task_type == 'Ranking':
      adv_logits = adv_logits.view(-1, pairwise)
    adv_lc = self.loss_map[task_id]
    adv_loss = adv_lc(logits, adv_logits, ignore_index=-1)
    return adv_loss, embed.detach().abs().mean(), eff_noise.detach().abs(
    ).mean()


if __name__ == '__main__':
  pass
