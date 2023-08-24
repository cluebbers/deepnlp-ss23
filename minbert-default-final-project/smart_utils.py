from enum import IntEnum
import torch.nn.functional as F
import torch
from torch import nn
from shared_classifier import *

BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5

class TaskType(IntEnum):
    Classification = 1
    Regression = 2
    Ranking = 3
    SpanClassification = 4  # squad v1
    SpanClassificationYN = 5  # squad v2
    SeqenceLabeling = 6
    MaskLM = 7
    SpanSeqenceLabeling = 8
    SeqenceGeneration = 9
    SeqenceGenerationMRC = 10
    EncSeqenceGeneration = 11
    ClozeChoice = 12
    
class EncoderModelType(IntEnum):
    BERT = 1
    ROBERTA = 2
    XLNET = 3
    SAN = 4
    XLM = 5
    DEBERTA = 6
    ELECTRA = 7
    T5 = 8
    T5G = 9
    MSRT5G = 10
    MSRT5 = 11
    MSRLONGT5G = 12
    MSRLONGT5 = 13
    
def stable_kl(logit, target, epsilon=1e-6, reduce=True):
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
    
class SymKlCriterion():
    def __init__(self, alpha=1.0, name="KL Div Criterion"):
        super().__init__()
        self.alpha = alpha
        self.name = name

    def forward(
        self, input, target, weight=None, ignore_index=-1, reduction="batchmean"
    ):
        """input/target: logits"""
        input = input.float()
        target = target.float()
        loss = F.kl_div(
            F.log_softmax(input, dim=-1, dtype=torch.float32),
            F.softmax(target.detach(), dim=-1, dtype=torch.float32),
            reduction=reduction,
        ) + F.kl_div(
            F.log_softmax(target, dim=-1, dtype=torch.float32),
            F.softmax(input.detach(), dim=-1, dtype=torch.float32),
            reduction=reduction,
        )
        loss = loss * self.alpha
        return loss
    
class MseCriterion():
    def __init__(self, alpha=1.0, name="MSE Regression Criterion"):
        super().__init__()
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1):
        """weight: sample weight"""
        if weight:
            loss = torch.mean(
                F.mse_loss(input.squeeze(), target, reduce=False)
                * weight.reshape((target.shape[0], 1))
            )
        else:
            loss = F.mse_loss(input.squeeze(), target)
        loss = loss * self.alpha
        return loss