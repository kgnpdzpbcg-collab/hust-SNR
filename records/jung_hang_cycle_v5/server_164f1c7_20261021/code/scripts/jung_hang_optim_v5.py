"""AdaMod: adapted momental bounds on per-parameter Adam step sizes.

Jung et al. (TPEL 2025) explicitly report AdaMod optimizer. This standalone
implementation follows the published AdaMod bounded effective learning rate.
"""
import math
import torch
from torch.optim import Optimizer


class AdaMod(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999, 0.999),
                 eps=1e-8, weight_decay=0.0):
        if lr <= 0 or eps <= 0 or weight_decay < 0:
            raise ValueError("invalid optimizer parameters")
        if len(betas) != 3 or any(not 0 <= x < 1 for x in betas):
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
                mhat = state["m"] / (1 - b1 ** t)
                vhat = state["v"] / (1 - b2 ** t)
                eta = group["lr"] / (torch.sqrt(vhat) + group["eps"])
                state["s"].mul_(b3).add_(eta, alpha=1 - b3)
                bounded = torch.minimum(eta, state["s"])
                p.add_(-bounded * mhat)
        return loss
