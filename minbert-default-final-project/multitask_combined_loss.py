from shared_classifier import *

import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable

from bert import BertModel
from torch.optim import AdamW
from tqdm import tqdm

from datasets import SentenceClassificationDataset, SentencePairDataset, \
    load_multitask_data

# CLL import multitask evaluation
from evaluation import smart_eval
import itertools
# tensorboard
from torch.utils.tensorboard import SummaryWriter
# SOPHIA
from optimizer_sophia import SophiaG
# SMART regularization
from smart_perturbation import SmartPerturbation
import smart_utils as smart
# PCGrad
from pcgrad import PCGrad

TQDM_DISABLE=False



BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5

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
    

    
    # Init model
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option,
              'local_files_only': args.local_files_only}

    config = SimpleNamespace(**config)

    model = smart.SmartMultitaskBERT(config)
    model = model.to(device)

    lr = args.lr
    weight_decay = args.weight_decay    

    if args.optimizer == "sophiag":
        optimizer = SophiaG(model.parameters(), lr=lr, betas=(0.965, 0.99), weight_decay=weight_decay)
        #how often to update the hessian?
        k = args.k_for_sophia
    elif args.optimizer == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        print("no known optimizer")
        
    # PCGrad
    if args.pcgrad:
        optimizer = PCGrad(optimizer)
    
    # tensorboard writer
    writer = SummaryWriter(comment = args.comment)
    
    # SMART    
    if args.smart:
        smart_loss_sst = smart.SymKlCriterion().forward
        smart_loss_qqp = smart.SymKlCriterion().forward
        smart_loss_sts = smart.MseCriterion().forward
        smart_perturbation = SmartPerturbation(loss_map={0:smart_loss_sst, 1:smart_loss_qqp, 2:smart_loss_sts})
        
    best_para_dev_acc = 0
    best_sst_dev_acc = 0
    best_sts_dev_cor = 0
    best_dev_acc = 0
    
    n_iter= len(para_train_dataloader)
                  
    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        # create cycles so we can iterate over the long dataset
        combined_iterator = itertools.zip_longest(sts_train_dataloader, sst_train_dataloader, para_train_dataloader)
        length_of_zip_longest = max(len(sts_train_dataloader), len(sst_train_dataloader), len(para_train_dataloader))

        model.train()
        total_loss = 0
        num_batches = 0        
           
        for sts_batch, sst_batch, para_batch in tqdm(combined_iterator, total=length_of_zip_longest, desc=f'train-combined-{epoch}', disable=TQDM_DISABLE):
     
            optimizer.zero_grad()
            combined_loss = 0
            
            #train on semantic textual similarity (sts)
            if sts_batch:
                b_ids1, b_mask1, b_ids2, b_mask2, b_labels = (sts_batch['token_ids_1'].to(device),
                                                            sts_batch['attention_mask_1'].to(device),                                                          
                                                            sts_batch['token_ids_2'].to(device),
                                                            sts_batch['attention_mask_2'].to(device),
                                                            sts_batch['labels'].to(device))

                similarity_logit = model(b_ids1, b_mask1, b_ids2, b_mask2, task_id=2)
                
                if args.smart:
                    adv_loss = smart_perturbation.forward(
                        model=model,
                        logits=similarity_logit,
                        input_ids_1=b_ids1,                
                        attention_mask_1=b_mask1,
                        input_ids_2=b_ids2,
                        attention_mask_2=b_mask2,
                        task_id=2,
                        task_type=smart.TaskType.Regression) 
                else:
                    adv_loss = 0
                    
                loss_sts  = F.mse_loss(similarity_logit, b_labels.view(-1).float(), reduction='mean')
                loss_sts  = loss_sts + adv_loss
                combined_loss += loss_sts 
            
            # train on sentiment analysis (sst)
            if sst_batch:
                b_ids, b_mask, b_labels = (sst_batch['token_ids'].to(device),
                                        sst_batch['attention_mask'].to(device), 
                                        sst_batch['labels'].to(device))
                
                sentiment_logits = model(b_ids, b_mask, task_id=0)
                
                # SMART
                if args.smart:
                    adv_loss = smart_perturbation.forward(
                        model=model,
                        logits=sentiment_logits,
                        input_ids_1=b_ids,                
                        attention_mask_1=b_mask,
                        task_id=0,
                        task_type=smart.TaskType.Classification) 
                else:
                    adv_loss = 0    
                    
                loss_sst = F.cross_entropy(sentiment_logits, b_labels.view(-1), reduction='mean') 
                loss_sst = loss_sst + adv_loss
                combined_loss += loss_sst
             
            # train on paraphrasing
            if para_batch:
                b_ids1, b_mask1, b_ids2, b_mask2, b_labels = (para_batch['token_ids_1'].to(device), 
                                                            para_batch['attention_mask_1'].to(device),
                                                            para_batch['token_ids_2'].to(device),
                                                            para_batch['attention_mask_2'].to(device),
                                                            para_batch['labels'].to(device))
                
                paraphrase_logits = model(b_ids1, b_mask1, b_ids2, b_mask2, task_id = 1)
                
                # SMART
                if args.smart:
                    adv_loss = smart_perturbation.forward(
                        model=model,
                        logits=paraphrase_logits,
                        input_ids_1=b_ids1,                
                        attention_mask_1=b_mask1,
                        input_ids_2=b_ids2,
                        attention_mask_2=b_mask2,
                        task_id=1,
                        task_type=smart.TaskType.Classification) 
                else:
                    adv_loss = 0
                    
                loss_para = F.binary_cross_entropy_with_logits(paraphrase_logits, b_labels.view(-1).float(), reduction='mean')
                loss_para = loss_para + adv_loss
                combined_loss += loss_para
            
            if args.pcgrad:
                losses = [loss_para, loss_sts, loss_sst]
                optimizer.pc_backward(losses)
            else:                     
                combined_loss.backward()  
                          
            optimizer.step()         
            total_loss += combined_loss.item()
            num_batches += 1
            
            # SOPHIA
            # update hession EMA
            if args.optimizer == "sophiag" and num_batches % k == k - 1:                  
                optimizer.update_hessian()
                optimizer.zero_grad(set_to_none=True)
            
            if n_iter <= num_batches:
                    break             
            
        # evaluation          
        avg_loss = total_loss / max(num_batches, 1)            
        print(f"Epoch {epoch}: sst-loss: {loss_sst:.3f}, sts-loss: {loss_sts:.3f}, para-loss: {loss_para:.3f}")
        print(f"avg loss: {avg_loss}") 
         
        (_,train_para_acc, _, _, train_para_prec, train_para_rec, train_para_f1,
         _,train_sst_acc, _, _, train_sst_prec, train_sst_rec, train_sst_f1,
         _,train_sts_corr, *_ )= smart_eval(sst_train_dataloader,
                                                    para_train_dataloader,
                                                    sts_train_dataloader,
                                                    model, device, n_iter)
         
        # tensorboard   
        writer.add_scalar("avg_loss", avg_loss, epoch)
        writer.add_scalar("sts/train_loss", loss_sts, epoch)
        writer.add_scalar("sst/train_loss", loss_sst, epoch)
        writer.add_scalar("para/train_loss", loss_para, epoch)
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
         sts_loss,dev_sts_cor, *_ )= smart_eval(sst_dev_dataloader,
                                                 para_dev_dataloader,
                                                 sts_dev_dataloader,
                                                 model, device, n_iter)        

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
    
    writer.add_hparams({"comment": args.comment,
                        "epochs":args.epochs,
                        "optimizer":args.optimizer, 
                        "lr":args.lr, 
                        "weight_decay":args.weight_decay,
                        "k_for_sophia":args.k_for_sophia,
                        "hidden_dropout_prob": args.hidden_dropout_prob,
                        "batch_size":args.batch_size},
                        {"para-dev-acc":best_para_dev_acc,
                        "sst-dev-acc":best_sst_dev_acc,
                        "sts-dev-cor":best_sts_dev_cor})
    # close tensorboard writer
    writer.flush()
    writer.close()

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
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="finetune")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    # hyper parameters
    parser.add_argument("--batch_size", help='sst: 64 can fit a 12GB GPU', type=int, default=2)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)
    parser.add_argument("--local_files_only", action='store_true'),
    parser.add_argument("--optimizer", type=str, help="adamw or sophiag", choices=("adamw", "sophiag"), default="sophiag")
    parser.add_argument("--weight_decay", help="default for 'adamw': 0.01", type=float, default=0)
    parser.add_argument("--k_for_sophia", type=int, help="how often to update the hessian? default is 10", default=10)
    parser.add_argument("--smart", action="store_true")   
    parser.add_argument("--comment", type=str, default="")
    parser.add_argument("--pcgrad", action="store_true")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.option}-{args.epochs}-{args.lr}-multitask.pt' # save path
    seed_everything(args.seed)  # fix the seed for reproducibility    
    train_multitask(args)
