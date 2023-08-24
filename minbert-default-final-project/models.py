import torch.nn.functional as F
import torch
from torch import nn
from shared_classifier import *

BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5

class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    
    Baseline model
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

        # raise NotImplementedError


    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO
        # the same as the first part in classifier.BertSentimentClassifier.forward
        pooled = self.bert(input_ids, attention_mask)['pooler_output']
        pooled = self.dropout(pooled)
        
        return pooled
    
        # raise NotImplementedError


    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        ### TODO
        # the same as in classifier.BertSentimentClassifier.forward        
        # input embeddings
        pooled = self.forward(input_ids, attention_mask)     
        
        # The class will then classify the sentence by applying dropout on the pooled output
        #pooled = self.dropout(pooled)
            
        # and then projecting it using a linear layer.
        sentiment_logit = self.sentiment_classifier(pooled)

        
        return sentiment_logit
        # raise NotImplementedError


    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        ### TODO
        # input embeddings
        pooled_1 = self.forward(input_ids_1, attention_mask_1)
        pooled_2 = self.forward(input_ids_2, attention_mask_2)
        
        # Fernando and Stevenson, 2008
        # paraphrase is just like similarity
        # similarity = F.cosine_similarity(pooled_1, pooled_2, dim=1)
        
        # cosine_similarity has ouput [-1, 1], so it needs rescaling
        # Reshape the similarity to fit the input shape of paraphrase_classifier
        # similarity = similarity.view(-1, 1)  
       
        # Generate the logit
        # paraphrase_logit = self.paraphrase_classifier(similarity)   
        # Remove the extra dimension added by paraphrase_classifier
        # paraphrase_logit = paraphrase_logit.view(-1)
        
        # Element-wise difference
        diff = torch.abs(pooled_1 - pooled_2)
        
        # Element-wise product
        prod = pooled_1 * pooled_2

        # Concatenate difference and product
        pooled = torch.cat([diff, prod], dim=-1)
        
        paraphrase_logit = self.paraphrase_classifier(pooled).view(-1)
        
        return paraphrase_logit
        # raise NotImplementedError


    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        ### TODO
        # input embeddings
        pooled_1 = self.forward(input_ids_1, attention_mask_1)
        pooled_2 = self.forward(input_ids_2, attention_mask_2)
        
        # use cosine similarity as in
        # Agirre et al "SEM 2013 shared task: Semantic Textual Similarity" section 4.2
        # cosine_similarity has ouput [-1, 1], so it needs rescaling
        # +1 to get to [0, 2]
        # /2 to get to [0, 1]
        # *5 to get [0, 5] like in the dataset
        #similarity = (F.cosine_similarity(pooled_1, pooled_2, dim=1) + 1) * 2.5
        similarity = (self.similarity_classifier(pooled_1,pooled_2)+1)*2.5
        # without scaling
        # similarity = F.cosine_similarity(pooled_1, pooled_2, dim=1) 
        
        return similarity
        # raise NotImplementedError

class SharedMultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    prediction and similarity share an additional layer
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
        self.paraphrase_classifier = torch.nn.Linear(self.hidden_size, 1)
        
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
        
        # use SMART noised embeddings if present
        if task_na==2:
            embed_1 = embednoise_1
            embed_2 = embednoise_2 if embednoise_2 is not None else None
        else:          
            # input embeddings
            embed_1 = self.bert(input_ids_1, attention_mask_1)['pooler_output']            
            embed_2 = self.bert(input_ids_2, attention_mask_2)['pooler_output'] if input_ids_2 is not None else None
            
        embed_1 = self.dropout(embed_1)
        embed_2 = self.dropout(embed_2) if input_ids_2 is not None else None
        
        # return embeddings for smart or proceed with logits
        if task_na == 1:
            return embed_1, embed_2
        
        if task_id == 0: # Sentiment classification
            sentiment_logit = self.sentiment_classifier(embed_1)        
            return sentiment_logit

        else: 
            similarity = (self.similarity_classifier(embed_1, embed_2)+1)*2.5      
            if task_id == 2: # Semantic Textual Similarity
                return similarity
            elif task_id == 1: 
                return self.predict_similarity(embed_1, embed_2)
            else:
                raise ValueError("Invalid task_id")
            
class SmartMultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    
    This is optimized to work with SMART
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