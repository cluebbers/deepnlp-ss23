from shared_classifier import *
from datasets import *

import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F

from torch.autograd import Variable

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from evaluation import test_model_multitask

# CLL import multitask evaluation
from evaluation import model_eval_multitask
# tensorboard
from torch.utils.tensorboard import SummaryWriter
# SOPHIA
from optimizer_sophia import SophiaG
# profiling
from torch.profiler import profile, record_function, ProfilerActivity
from custom_attention import *

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
        self.config = config
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

    @staticmethod
    def from_config(args, device, num_labels):
        config = SimpleNamespace(
            hidden_dropout_prob = args.hidden_dropout_prob,
            num_labels = num_labels,
            hidden_size = 768,
            data_dir = '.',
            option = args.option,
            local_files_only = args.local_files_only,
            attention_module = eval(args.custom_attention)
        )
        model = MultitaskBERT(config)
        model = model.to(device)
        return model


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


def load_optimizer(model, args):
    if args.optimizer == "sophiag":
        return SophiaG(model.parameters(), lr = args.lr, weight_decay = args.weight_decay,
                       betas = (0.965, 0.99), rho = 0.01)
    else:
        return AdamW(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)


def update_optimizer(optimizer, args, num_batches):
    if args.optimizer != "sophiag":
        return
    k = args.k_for_sophia
    if num_batches % k != k - 1:
        return
    optimizer.update_hessian()


def train_multitask(args):
    device      = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    dataloaders = MultitaskDataloader(args, device)
    model       = MultitaskBERT.from_config(args, device, dataloaders.num_labels)
    optimizer   = load_optimizer(model, args)
    writer      = SummaryWriter(comment = args.logdir)

    writer.add_hparams({
        "epochs": args.epochs,
        "optimizer": args.optimizer, 
        "lr": args.lr, 
        "weight_decay": args.weight_decay,
        "k_for_sophia": args.k_for_sophia,
        "hidden_dropout_prob": args.hidden_dropout_prob,
        "batch_size":args.batch_size,
        "custom_attention":args.custom_attention
    }, {})

    best_para_dev_acc = 0
    best_sst_dev_acc = 0
    best_sts_dev_cor = 0
    best_dev_acc = 0

    for epoch in range(args.epochs):
        #train on semantic textual similarity (sts)
        model.train()
        train_loss = 0
        num_batches = 0        
         
        for b_ids1, b_mask1, b_ids2, b_mask2, b_labels in dataloaders.iter_train_sts(epoch):

            optimizer.zero_grad(set_to_none=True)
            similarity = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2)
            # we need a loss function for similarity
            # there are different degrees of similarity
            # So maybe the mean squared error is a suitable loss function for the beginning,
            # since it punishes a prediction that is far away from the truth disproportionately
            # more than a prediction that is close to the truth
            loss = F.mse_loss(similarity, b_labels.view(-1).float(), reduction='mean')
            if args.option == "pretrain":
                loss.requires_grad = True
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1
            update_optimizer(optimizer, args, num_batches)

        train_loss = train_loss / num_batches
        # tensorboard
        writer.add_scalar("sts/train_loss", train_loss, epoch)
        
        print(f"Epoch {epoch}: Semantic Textual Similarity -> train loss: {train_loss:.3f}")

        # train on sentiment analysis

        model.train()
        train_loss = 0
        num_batches = 0
        for b_ids, b_mask, b_labels in dataloaders.iter_train_sst(epoch):

            optimizer.zero_grad(set_to_none=True)
            logits = model.predict_sentiment(b_ids, b_mask)
            loss = F.cross_entropy(logits, b_labels.view(-1), reduction='mean')

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1
            update_optimizer(optimizer, args, num_batches)

        train_loss = train_loss / (num_batches)
        
        # tensorboard
        writer.add_scalar("sst/train_loss", train_loss, epoch)

        print(f"Epoch {epoch}: Sentiment classification -> train loss :: {train_loss :.3f}")

        # train on paraphrasing
        # CLL add training on other tasks
        # paraphrasing
        # see evaluation.py line 72

        
        model.train()
        train_loss = 0
        num_batches = 0
        
        # datasets.py line 145
        for b_ids1, b_mask1, b_ids2, b_mask2, b_labels in dataloaders.iter_train_para(epoch):
            
            optimizer.zero_grad(set_to_none=True)
            logits = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2)
            
            # we need a loss function that handles logits. maybe this one?
            # paraphrasing is a binary task, so binary
            # we get logits, so logits
            # this one also has a sigmoid activation function
            loss = F.binary_cross_entropy_with_logits(logits, b_labels.view(-1).float(), reduction='mean')

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1
            update_optimizer(optimizer, args, num_batches)

        train_loss = train_loss / num_batches
        
        # tensorboard
        writer.add_scalar("para/train_loss", train_loss, epoch)
        
        print(f"Epoch {epoch}: Paraphrase Detection -> train loss: {train_loss:.3f}")
        

        if args.profiler:
            break

        # evaluation
        
        (_,train_para_acc, _, _, train_para_prec, train_para_rec, train_para_f1,
         _,train_sst_acc, _, _, train_sst_prec, train_sst_rec, train_sst_f1,
         _,train_sts_corr, *_ ) = model_eval_multitask(model, device, dataloaders, dev = False)
         
        # tensorboard   
        writer.add_scalar("para/train-acc", train_para_acc, epoch)
        writer.add_scalar("para/train-prec", train_para_prec, epoch)
        writer.add_scalar("para/train-rec", train_para_rec, epoch)
        writer.add_scalar("para/train-f1", train_para_f1, epoch)
        
        writer.add_scalar("sst/train-acc", train_sst_acc, epoch)
        for i, class_precision in enumerate(train_sst_prec):
            writer.add_scalar(f"sst/train-prec/class_{i}", class_precision, epoch)
        for i, class_recall in enumerate(train_sst_rec):
            writer.add_scalar(f"sst/train-rec/class_{i}", class_recall, epoch)  
        for i, class_f1 in enumerate(train_sst_f1):
            writer.add_scalar(f"sst/train-f1/class_{i}", class_f1, epoch)   
            
        writer.add_scalar("sts/train-cor", train_sts_corr, epoch) 
        
        (para_loss,dev_para_acc, _, _, dev_para_prec, dev_para_rec, dev_para_f1,
         sst_loss,dev_sst_acc, _, _, dev_sst_prec, dev_sst_rec, dev_sst_f1,
         sts_loss,dev_sts_cor, *_ )= model_eval_multitask(model, device, dataloaders, dev = True)        

        # tensorboard
        writer.add_scalar("para/dev_loss", para_loss, epoch)
        writer.add_scalar("sst/dev_loss", sst_loss, epoch)
        writer.add_scalar("sts/dev_loss", sts_loss, epoch)
        
        writer.add_scalar("para/dev-acc", dev_para_acc, epoch) 
        writer.add_scalar("para/dev-prec", dev_para_prec, epoch)
        writer.add_scalar("para/dev-rec", dev_para_rec, epoch)
        writer.add_scalar("para/dev-f1", dev_para_f1, epoch) 
                                 
        writer.add_scalar("sst/dev-acc", dev_sst_acc, epoch)        
        for i, class_precision in enumerate(dev_sst_prec):
            writer.add_scalar(f"sst/dev-prec/class_{i}", class_precision, epoch)       
        for i, class_recall in enumerate(dev_sst_rec):
            writer.add_scalar(f"sst/dev-rec/class_{i}", class_recall, epoch)          
        for i, class_f1 in enumerate(dev_sst_f1):
            writer.add_scalar(f"sst/dev-f1/class_{i}", class_f1, epoch)        
        
        writer.add_scalar("sts/dev-cor", dev_sts_cor, epoch)
        
        # store best results    
        if dev_para_acc > best_para_dev_acc:
            best_para_dev_acc = dev_para_acc
        if dev_sst_acc > best_sst_dev_acc:
            best_sst_dev_acc = dev_sst_acc
        if dev_sts_cor > best_sts_dev_cor:
            best_sts_dev_cor = dev_sts_cor 
        
        # TODO: save model 
        # and maybe use writer.add_hparams() to save parameters and accuracy in tensorboard
        # take average of the two accuraies,for para and sst, and the correlation of sts
        # normalize all three values to the interval (0,1) before taking average
        # note that accuracies are already normalized
        
        dev_sts_cor_norm = (dev_sts_cor+1)/2 #correlation coefficient lies in (-1,1)
        dev_acc = (dev_sts_cor_norm+dev_sst_acc+dev_para_acc)/3
        
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            #writer.add_hparams()
            save_model(model, optimizer, args, model.config, args.filepath)
          
        # save each epoch of the trained model for detailed error analysis
        if  args.save:
            save_model(model,optimizer, args, model.config,"Models/epoch"+str(epoch)+"-"+f'{args.option}-{args.lr}-multitask.pt')

        # cool down GPU    
        if epoch %10 ==9:
            time.sleep(60*5)                     
        
    # tensorboard
    # collect all information of run    
    writer.add_hparams({"epochs":args.epochs,
                        "optimizer":args.optimizer, 
                        "lr":args.lr, 
                        "weight_decay":args.weight_decay,
                        "k_for_sophia":args.k_for_sophia,
                        "hidden_dropout_prob": args.hidden_dropout_prob,
                        "batch_size":args.batch_size,
                        "custom_attention":args.custom_attention},
                        {"para-dev-acc":best_para_dev_acc,
                        "sst-dev-acc":best_sst_dev_acc,
                        "sts-dev-cor":best_sts_dev_cor})

    # close tensorboard writer
    writer.flush()
    writer.close()

    
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
    parser.add_argument("--epochs", type=int, default=6)
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
    parser.add_argument("--num_batches_para", help='sst: 64 can fit a 12GB GPU', type=int, default=float('nan'))
    parser.add_argument("--num_batches_sst", help='sst: 64 can fit a 12GB GPU', type=int, default=float('nan'))
    parser.add_argument("--num_batches_sts", help='sst: 64 can fit a 12GB GPU', type=int, default=float('nan'))
    
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-3)
    parser.add_argument("--local_files_only", action='store_true', default = True),
    parser.add_argument("--optimizer", type=str, help="adamw or sophiag", choices=("adamw", "sophiag"), default="adamw"),
    parser.add_argument("--weight_decay", help="default for 'adamw': 0.01", type=float, default=0),
    parser.add_argument("--k_for_sophia", type=int, help="how often to update the hessian? default is 10", default=10),
    parser.add_argument("--profiler", action="store_true")
    parser.add_argument("--custom_attention", type=str, choices = CUSTOM_ATTENTION_CHOICES,
                        help="Which custom attention should be used?", default = "BertSelfAttention")
    
    parser.add_argument("--save", type=bool, default=True)
    parser.add_argument("--logdir", type=str, default='')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    args.filepath = f'Models/{args.option}-{args.lr}-multitask.pt' # save path for model
    seed_everything(args.seed)  # fix the seed for reproducibility    
    
    train_multitask(args)

    if not args.profiler:
        test_model(args)
