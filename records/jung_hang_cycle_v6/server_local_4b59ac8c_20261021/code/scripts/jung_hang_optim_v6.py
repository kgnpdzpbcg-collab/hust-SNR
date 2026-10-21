"""AdaMod：按原文作者给出的带参数级步长上界的 Adam 变体。

Jung et al. (TPEL 2025) Sec. V-A 明确使用 AdaMod 训练。实现对应
Ding et al., "An adaptive and momental bound method for stochastic learning"
(arXiv:1910.12249) 的更新式：

    m_t = b1*m_{t-1} + (1-b1)*g_t
    v_t = b2*v_{t-1} + (1-b2)*g_t^2
    eta_t = lr / (sqrt(v_t/(1-b2^t)) + eps)      # 参数级步长
    s_t = b3*s_{t-1} + (1-b3)*eta_t              # 步长滑动上界
    p_t = p_{t-1} - min(eta_t, s_t) * m_t/(1-b1^t)

与 V5 的实现逐行一致，V6 只是把它放进本轮 scripts 目录，便于部署快照自洽。
"""
import torch
from torch.optim import Optimizer


class AdaMod(Optimizer):
    """参数级步长自适应上界优化器；betas=(b1, b2, b3)。"""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999, 0.999),
                 eps=1e-8, weight_decay=0.0):
        if lr <= 0 or eps <= 0 or weight_decay < 0:
            raise ValueError("invalid optimizer parameters")
        if len(betas) != 3 or any(not 0 <= value < 1 for value in betas):
            raise ValueError("betas must all be in [0,1)")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                      weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            b1, b2, b3 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdaMod does not support sparse gradients")
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    state["s"] = torch.zeros_like(p)
                state["step"] += 1
                t = state["step"]
                if group["weight_decay"]:
                    grad = grad.add(p, alpha=group["weight_decay"])
                state["m"].mul_(b1).add_(grad, alpha=1 - b1)
                state["v"].mul_(b2).addcmul_(grad, grad, value=1 - b2)
                m_hat = state["m"] / (1 - b1 ** t)
                v_hat = state["v"] / (1 - b2 ** t)
                eta = group["lr"] / (torch.sqrt(v_hat) + group["eps"])
                state["s"].mul_(b3).add_(eta, alpha=1 - b3)
                p.add_(-torch.minimum(eta, state["s"]) * m_hat)
        return loss
