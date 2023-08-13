from shared_classifier import *

import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from datasets import SentenceClassificationDataset, SentencePairDataset, \
    load_multitask_data, load_multitask_test_data

from evaluation import test_model_multitask

# CLL import multitask evaluation
from evaluation import model_eval_multitask
# tensorboard
from torch.utils.tensorboard import SummaryWriter
# SOPHIA
from optimizer_sophia import SophiaG
# profiling
from torch.profiler import profile, record_function, ProfilerActivity

TQDM_DISABLE=False



BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class MultitaskBERT(nn.Module):
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
        pooled = self.dropout(pooled)
            
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
        similarity = (F.cosine_similarity(pooled_1, pooled_2, dim=1) + 1) * 2.5
        
        # without scaling
        similarity = F.cosine_similarity(pooled_1, pooled_2, dim=1)
        
        return similarity
        # raise NotImplementedError




def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


def train_multitask(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
       
    # Load data
    # Create the data and its corresponding datasets and dataloader
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)
    
    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)
    
    # CLL added other datasets and loaders
    # see evaluation.py line 229
    # paraphrasing
    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)
    
    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=para_dev_data.collate_fn)
    # similarity
    sts_train_data = SentencePairDataset(sts_train_data, args)
    sts_dev_data = SentencePairDataset(sts_dev_data, args)
    
    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_data.collate_fn)   
    # CLL end

    # Init model
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option,
              'local_files_only': args.local_files_only}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    # TODO: maybe different learning rate for different tasks?
    lr = args.lr
    weight_decay = args.weight_decay

    if args.optimizer == "sophiag":
        optimizer = SophiaG(model.parameters(), lr=lr, betas=(0.965, 0.99), rho = 0.01, weight_decay=weight_decay)
        #how often to update the hessian?
        k = args.k_for_sophia
    else:
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # tensorboard writer
    writer = SummaryWriter()
    
    # profiler
    profiler = args.profiler
    if profiler:            
        prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
                on_trace_ready=torch.profiler.tensorboard_trace_handler("runs/profiler"),
                record_shapes=True,
                profile_memory=True,
                with_stack=False)
        
    best_para_dev_acc = 0
    best_sst_dev_acc = 0
    best_sts_dev_cor = 0
                  
    # Run for the specified number of epochs
    for epoch in range(args.epochs):
                
        #train on semantic textual similarity (sts)
        
        # profiler start
        if profiler:
            prof.start()
        
        model.train()
        total_loss = 0
        num_batches = 0
        
        # TODO only trains for one batch? 755 iterations
        # maybe only combine sst and para since they perform much better
        for (sst_batch, para_batch, sts_batch) in tqdm(zip(sst_train_dataloader, para_train_dataloader, sts_train_dataloader),
                                                       desc=f'train-{epoch}', disable=TQDM_DISABLE):
            
            
            #train on semantic textual similarity (sts)
            b_ids1, b_mask1, b_ids2, b_mask2, b_labels = (sts_batch['token_ids_1'].to(device),
                                                          sts_batch['attention_mask_1'].to(device),                                                          
                                                          sts_batch['token_ids_2'].to(device),
                                                          sts_batch['attention_mask_2'].to(device),
                                                          sts_batch['labels'].to(device))

            similarity_logit = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2)
            similarity_loss = F.mse_loss(similarity_logit, b_labels.view(-1).float(), reduction='mean')
            
            # train on sentiment analysis (sst)
            b_ids, b_mask, b_labels = (sst_batch['token_ids'].to(device),
                                       sst_batch['attention_mask'].to(device), 
                                       sst_batch['labels'].to(device))
            
            sentiment_logits = model.predict_sentiment(b_ids, b_mask)
            sentiment_loss = F.cross_entropy(sentiment_logits, b_labels.view(-1), reduction='sum') / args.batch_size
             
            # train on paraphrasing
            b_ids1, b_mask1, b_ids2, b_mask2, b_labels = (para_batch['token_ids_1'].to(device), 
                                                          para_batch['attention_mask_1'].to(device),
                                                          para_batch['token_ids_2'].to(device),
                                                          para_batch['attention_mask_2'].to(device),
                                                          para_batch['labels'].to(device))
            
            paraphrase_logits = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2)
            paraphrase_loss = F.binary_cross_entropy_with_logits(paraphrase_logits, b_labels.view(-1).float(), reduction='sum') / args.batch_size
            
            # combine losses
            # TODO: different wweights?
            loss = sentiment_loss + paraphrase_loss + similarity_loss/10            
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            num_batches += 1
            
        paraphrase_accuracy, para_y_pred, para_sent_ids, sentiment_accuracy,sst_y_pred, sst_sent_ids, sts_corr, sts_y_pred, sts_sent_ids = model_eval_multitask(sst_dev_dataloader,
                         para_dev_dataloader,
                         sts_dev_dataloader,
                         model, device)

        total_loss = total_loss / num_batches
        
        # TODO: better metric
        dev_acc = (paraphrase_accuracy + sentiment_accuracy + sts_corr)/3
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)
        
        print(f"Epoch {epoch}: sst-loss: {sentiment_loss:.3f}, sts-loss: {similarity_loss:.3f}, para-loss: {paraphrase_loss:.3f}")

def test_model(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        test_model_multitask(args, model, device)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    # hyper parameters
    parser.add_argument("--batch_size", help='sst: 64 can fit a 12GB GPU', type=int, default=64)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-3)
    parser.add_argument("--local_files_only", action='store_true'),
    parser.add_argument("--optimizer", type=str, help="adamw or sophiag", choices=("adamw", "sophiag"), default="adamw"),
    parser.add_argument("--weight_decay", help="default for 'adamw': 0.01", type=float, default=0),
    parser.add_argument("--k_for_sophia", type=int, help="how often to update the hessian? default is 10", default=10),
    parser.add_argument("--profiler", action="store_true")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.option}-{args.epochs}-{args.lr}-multitask.pt' # save path
    seed_everything(args.seed)  # fix the seed for reproducibility    
    train_multitask(args)
    
    # TODO: uncomment for finalizing part 2
    # test_model(args)
