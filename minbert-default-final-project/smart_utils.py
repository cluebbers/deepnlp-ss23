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
    
class SmartMultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super().__init__()
        self.bert = load_bert_model(config)

        self.hidden_size = BERT_HIDDEN_SIZE
        self.num_labels = N_SENTIMENT_CLASSES
        
        # see bert.BertModel.embed
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        
        # linear sentiment classifier
        self.sentiment_classifier= torch.nn.Linear(self.hidden_size, self.num_labels)
        
        # paraphrase classifier
        # double hidden size do concatenate both sentences
        self.paraphrase_classifier = torch.nn.Linear(self.hidden_size*2, 1)
        
        #cosine similarity classifier
        self.similarity_classifier = torch.nn.CosineSimilarity()

    def forward(self,
        input_ids_1,        
        attention_mask_1,
        input_ids_2=None,
        attention_mask_2=None,
        task_id=0,
        task_na=0,
        embednoise_1=None,
        embednoise_2=None):
        
        if task_na==2:
            embed_1 = embednoise_1
            embed_2 = embednoise_2 if embednoise_2 is not None else None
        else:          
            # input embeddings
            embed_1 = self.bert(input_ids_1, attention_mask_1)['pooler_output']            
            embed_2 = self.bert(input_ids_2, attention_mask_2)['pooler_output'] if input_ids_2 is not None else None
            
        embed_1 = self.dropout(embed_1)
        embed_2 = self.dropout(embed_2) if input_ids_2 is not None else None
        
        if task_na == 1:
            return embed_1, embed_2
        
        if task_id == 0: # Sentiment classification
            return self.predict_sentiment(embed_1)
        elif task_id == 1: # Paraphrase detection
            return self.predict_paraphrase(embed_1, embed_2)
        elif task_id == 2: # Semantic Textual Similarity
            return self.predict_similarity(embed_1, embed_2)
        else:
            raise ValueError("Invalid task_id")

    def predict_sentiment(self, embed_1):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''          
        # and then projecting it using a linear layer.
        sentiment_logit = self.sentiment_classifier(embed_1)
        
        return sentiment_logit

    def predict_paraphrase(self, embed_1, embed_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''      
        # Element-wise difference
        diff = torch.abs(embed_1 - embed_2)        
        # Element-wise product
        prod = embed_1 * embed_2
        # Concatenate difference and product
        pooled = torch.cat([diff, prod], dim=-1)
        
        paraphrase_logit = self.paraphrase_classifier(pooled).view(-1)
        
        return paraphrase_logit

    def predict_similarity(self, embed_1, embed_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''    
        # use cosine similarity as in
        # Agirre et al "SEM 2013 shared task: Semantic Textual Similarity" section 4.2
        # cosine_similarity has ouput [-1, 1], so it needs rescaling
        # +1 to get to [0, 2]
        # /2 to get to [0, 1]
        # *5 to get [0, 5] like in the dataset
        #similarity = (F.cosine_similarity(pooled_1, pooled_2, dim=1) + 1) * 2.5
        similarity = (self.similarity_classifier(embed_1, embed_2)+1)*2.5
        
        return similarity