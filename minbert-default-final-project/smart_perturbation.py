# Copyright (c) Microsoft. All rights reserved.

from copy import deepcopy
import torch
import logging
import random
from torch.nn import Parameter
from functools import wraps
import torch.nn.functional as F
from smart_utils import TaskType
from smart_utils import EncoderModelType
from smart_utils import stable_kl

logger = logging.getLogger(__name__)


def generate_noise(embed, epsilon=1e-5):
    noise = embed.data.new(embed.size()).normal_(0, 1) * epsilon
    noise.detach()
    noise.requires_grad_()
    return noise


class SmartPerturbation:
    def __init__(
        self,
        epsilon=1e-6,
        multi_gpu_on=False,
        step_size=1e-3,
        noise_var=1e-5,
        norm_p="inf",
        k=1,
        fp16=False,
        encoder_type=EncoderModelType.BERT,
        loss_map=[],
        norm_level=0,
    ):
        super(SmartPerturbation, self).__init__()
        self.epsilon = epsilon
        # eta
        self.step_size = step_size
        self.multi_gpu_on = multi_gpu_on
        self.fp16 = fp16
        self.K = k
        # sigma
        self.noise_var = noise_var
        self.norm_p = norm_p
        self.encoder_type = encoder_type
        self.loss_map = loss_map
        self.norm_level = norm_level > 0
        assert len(loss_map) > 0

    def _norm_grad(self, grad, eff_grad=None, sentence_level=False):
        if self.norm_p == "l2":
            if sentence_level:
                direction = grad / (
                    torch.norm(grad, dim=(-2, -1), keepdim=True) + self.epsilon
                )
            else:
                direction = grad / (
                    torch.norm(grad, dim=-1, keepdim=True) + self.epsilon
                )
        elif self.norm_p == "l1":
            direction = grad.sign()
        else:
            if sentence_level:
                direction = grad / (
                    grad.abs().max((-2, -1), keepdim=True)[0] + self.epsilon
                )
            else:
                direction = grad / (grad.abs().max(-1, keepdim=True)[0] + self.epsilon)
                eff_direction = eff_grad / (
                    grad.abs().max(-1, keepdim=True)[0] + self.epsilon
                )
        return direction

    def forward(
        self,
        model,
        logits,
        input_ids_1,
        attention_mask_1,
        input_ids_2=None,
        attention_mask_2=None,
        task_id=0,
        task_type=TaskType.Classification,
        pairwise=1,
    ):
        # adv training
        assert task_type in set(
            [TaskType.Classification, TaskType.Ranking, TaskType.Regression]
        ), "Donot support {} yet".format(task_type)
        vat_args = [
            input_ids_1,
            attention_mask_1,
            input_ids_2,
            attention_mask_2,
            task_id,
            1,
        ]

        # init delta
        embed_1, embed_2 = model(*vat_args)
        noise_1 = generate_noise(embed_1, epsilon=self.noise_var)
        embednoise_1 = embed_1 + noise_1
        
        if input_ids_2 is not None:
            noise_2 = generate_noise(embed_2, epsilon=self.noise_var)
            embednoise_2 = embed_2 + noise_2
        else:
            embednoise_2 = None
        
        for step in range(0, self.K):
            vat_args = [
                input_ids_1,                
                attention_mask_1,
                input_ids_2,
                attention_mask_2,
                task_id,
                2,
                embednoise_1,
                embednoise_2
            ]
            adv_logits = model(*vat_args)
            if task_type == TaskType.Regression:
                adv_loss = F.mse_loss(adv_logits, logits.detach(), reduction="mean")
            else:
                adv_loss = stable_kl(adv_logits, logits.detach(), reduce=False)
                
            if input_ids_2 is not None:
                delta_grad_1, delta_grad_2 = torch.autograd.grad(
                    adv_loss, (noise_1, noise_2), only_inputs=True, retain_graph=False)
            else:
                delta_grad_1, = torch.autograd.grad(
                    adv_loss, noise_1, only_inputs=True, retain_graph=False)
                delta_grad_2 = None
            
            # Calculating norm for first gradient
            norm_1 = delta_grad_1.norm()
            if torch.isnan(norm_1) or torch.isinf(norm_1):
                return 0
            eff_delta_grad_1 = delta_grad_1 * self.step_size
            delta_grad_1 = noise_1 + eff_delta_grad_1
            noise_1 = self._norm_grad(
                delta_grad_1, eff_grad=eff_delta_grad_1, sentence_level=self.norm_level
            )
            noise_1 = noise_1.detach()
            noise_1.requires_grad_()

            # If the second gradient is present, do the same for it
            if delta_grad_2 is not None:
                norm_2 = delta_grad_2.norm()
                if torch.isnan(norm_2) or torch.isinf(norm_2):
                    return 0
                eff_delta_grad_2 = delta_grad_2 * self.step_size
                delta_grad_2 = noise_2 + eff_delta_grad_2
                noise_2 = self._norm_grad(
                    delta_grad_2, eff_grad=eff_delta_grad_2, sentence_level=self.norm_level
                )
                noise_2 = noise_2.detach()
                noise_2.requires_grad_()

        vat_args = [
            input_ids_1,
            attention_mask_1,
            input_ids_2,
            attention_mask_2,
            task_id,
            2,
            embednoise_1,
            embednoise_2
        ]
        adv_logits = model(*vat_args)
        if task_type == TaskType.Ranking:
            adv_logits = adv_logits.view(-1, pairwise)
        adv_lc = self.loss_map[task_id]
        adv_loss = adv_lc(logits, adv_logits, ignore_index=-1)
        return adv_loss
