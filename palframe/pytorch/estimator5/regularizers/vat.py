# -*- coding: utf-8 -*- 
# @Time : 2021-3-14 15:03 
# @Author : by 周旋 
# @File : vat.py 
"""
vat loss 实现
"""

import torch
import torch.nn.functional as F


def kl_divergence_with_logit(q_logit, p_logit):
    q = F.softmax(q_logit, dim=1)
    qlogq = torch.mean(torch.sum(q * F.log_softmax(q_logit, dim=1), dim=1))
    qlogp = torch.mean(torch.sum(q * F.log_softmax(p_logit, dim=1), dim=1))
    return qlogq - qlogp


def get_normalized_vector(d):
    d_abs_max = torch.max(
        torch.abs(d.view(d.size(0), -1)), 1, keepdim=True)[0].view(
        d.size(0), 1, 1)
    # print(d_abs_max.size())
    d /= (1e-12 + d_abs_max)
    d /= torch.sqrt(1e-6 + torch.sum(
        torch.pow(d, 2.0), tuple(range(1, len(d.size()))), keepdim=True))
    # print(torch.norm(d.view(d.size(0), -1), dim=1))
    return d


def generate_virtual_adversarial_perturbation(x, logit, model, n_power, XI,
                                              epsilon):
    d = torch.randn_like(x)
    #print(d.shape,logit.shape)
    # power iteration, 目的是计算hessian矩阵的最大特征值值对应的特征向量
    for _ in range(n_power):
        d = XI * get_normalized_vector(d).requires_grad_()
        #print(_,d.shape,logit_m.shape)
        logit_m,_ = model(x + d)
        dist = kl_divergence_with_logit(logit, logit_m)
        grad = torch.autograd.grad(dist, [d])[0]
        d = grad.detach()
    # 最后这里的返回值将梯度强制投影到了epsilon球面
    return epsilon * get_normalized_vector(d)


def vat_adversarial_loss(
        x,
        logit,
        model_fn,
        epsilon=2,
        xi=1e-6,
        power_iteration_num=1
):
    """

    :param x: 需要进行计算扰动的变量, bert类模型中为embedding层
    :param logits: 模型计算得到的logits
    :param model_fn: 一个函数，一般使用偏函数的形式给出，调用时，输入为perturb_var，输出为 logits
    :param epsilon: vat 中范数的允许半径,默认为2，这是一个重要的调试参数
    :param xi:利用有限差分逼近hessian矩阵对应的尺度
    :param power_iteration_num:  利用power iteration计算最大hessian矩阵的特征向量对应的迭代次数
    :return:
    """
    # print(x.shape,logit.shape)
    r_vadv = generate_virtual_adversarial_perturbation(x, logit, model_fn,
                                                       power_iteration_num, xi, epsilon)
    logit_p = logit.detach()
    logit_m,_ = model_fn(x + r_vadv)
    loss = kl_divergence_with_logit(logit_p, logit_m)
    return loss



if __name__ == '__main__':
    pass
