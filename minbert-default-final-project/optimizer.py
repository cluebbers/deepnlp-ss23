from typing import Callable, Iterable, Tuple
import math

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]

                # Complete the implementation of AdamW here, reading and saving
                # your state in the `state` dictionary above.
                # The hyperparameters can be read from the `group` dictionary
                # (they are lr, betas, eps, weight_decay, as saved in the constructor).
                #
                # 1- Update first and second moments of the gradients
                # 2- Apply bias correction
                #    (using the "efficient version" given in https://arxiv.org/abs/1412.6980;
                #     also given in the pseudo-code in the project description).
                # 3- Update parameters (p.data).
                # 4- After that main gradient-based update, update again using weight decay
                #    (incorporating the learning rate again).

                ### TODO
                # see project description page 16
                # given
                # alpha is given above
                beta_1, beta_2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                
                # Initialize 
                # theta_t = p.data                
                if "step" not in state:
                    state["step"] = 0
                if "m_t" not in state:
                    state["m_t"] = torch.zeros_like(p.data) # dim like parameter tensor
                if "v_t" not in state:
                    state["v_t"] = torch.zeros_like(p.data) 
                
                # repeat
                state["step"] += 1 
                t = state["step"]
                m_t = state["m_t"]
                v_t = state["v_t"]                
                
                # 1- Update first and second moments of the gradients
                # here and below all operations are ELEMENT-WISE
                m_t.mul_(beta_1).add_(grad, alpha = 1 - beta_1)
                v_t.mul_(beta_2).add_(grad.pow(2), alpha = 1 - beta_2)
                
                # 2- Apply bias correction
                # normal:
                # m_t_hat = m_t / (1 - beta_1 ** t)
                # v_t_hat = v_t / (1 - beta_2 ** t)
                # efficiency version:
                # but: no alpha schedule according to structure.md
                alpha_t = alpha * math.sqrt(1 - beta_2 ** t) / (1- beta_1 ** t)
                
                # 3- Update parameters (p.data).
                # normal:
                # p.data = p.data - alpha * m_t_hat / (v_t_hat.sqrt() + eps)
                # efficiency version:
                p.data.addcdiv_(m_t, v_t.sqrt().add_(eps), value=-alpha_t)
                
                # 4- After that main gradient-based update, update again using weight decay
                # see "DECOUPLED WEIGHT DECAY REGULARIZATION" page 3
                # alpha or alpha_t?
                p.data.mul_(1 - alpha_t * weight_decay)
                
                # raise NotImplementedError


        return loss
