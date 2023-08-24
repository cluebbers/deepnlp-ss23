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
# SMART regularization
from smart_perturbation import SmartPerturbation
import smart_utils as smart
import torch.optim.lr_scheduler as lrscheduler

TQDM_DISABLE=False


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

    model = smart.SmartMultitaskBERT(config)
    model = model.to(device)
    
    # optimizer choice 
    # AdamW or SophiaG
    lr = args.lr
    weight_decay = args.weight_decay

    if args.optimizer == "sophiag":
        optimizer = SophiaG(model.parameters(), lr=lr, betas=(0.965, 0.99), rho = 0.01, weight_decay=weight_decay)
        #how often to update the hessian?
        k = args.k_for_sophia
    else:
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # SMART    
    if args.smart:
        smart_loss_sst = smart.SymKlCriterion().forward
        smart_loss_qqp = smart.SymKlCriterion().forward
        smart_loss_sts = smart.MseCriterion().forward
        smart_perturbation = SmartPerturbation(loss_map={0:smart_loss_sst, 1:smart_loss_qqp, 2:smart_loss_sts})
    
    # tensorboard writer
    writer = SummaryWriter(comment = args.comment)
   
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
    best_dev_acc = 0
                  
    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        #train on semantic textual similarity (sts)
        
        # profiler start
        if profiler:
            prof.start()
        
        model.train()
        train_loss = 0
        num_batches = 0        
         
        for batch in tqdm(sts_train_dataloader, desc=f'train-sts-{epoch}', disable=TQDM_DISABLE):#
            
            b_ids1, b_mask1, b_ids2, b_mask2, b_labels = (batch['token_ids_1'], batch['attention_mask_1'],
                                                          batch['token_ids_2'], batch['attention_mask_2'],
                                                          batch['labels'])
            
            b_ids1 = b_ids1.to(device)
            b_mask1 = b_mask1.to(device)
            b_ids2 = b_ids2.to(device)
            b_mask2 = b_mask2.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(b_ids1, b_mask1, b_ids2, b_mask2, task_id=2)
            
            # SMART
            if args.smart:
                adv_loss = smart_perturbation.forward(
                    model=model,
                    logits=logits,
                    input_ids_1=b_ids1,                
                    attention_mask_1=b_mask1,
                    input_ids_2=b_ids2,
                    attention_mask_2=b_mask2,
                    task_id=2,
                    task_type=smart.TaskType.Regression) 
            else:
                adv_loss = 0
            
            original_loss = F.mse_loss(logits, b_labels.view(-1).float(), reduction='sum')
            loss = original_loss + adv_loss
            
            if args.option == "pretrain":
                loss.requires_grad = True
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1
            
            # SOPHIA
            # update hession EMA
            if args.optimizer == "sophiag" and num_batches % k == k - 1:                  
                optimizer.update_hessian()
                optimizer.zero_grad(set_to_none=True)
            
            # profiling step
            if profiler:
                prof.step()
                 # stop after wait + warmup +active *repeat
                if num_batches >= (1+1+3):
                    break   
            
            # TODO for testing
            if num_batches >= args.num_batches_sts:
                break

        train_loss = train_loss / num_batches
        # tensorboard
        writer.add_scalar("sts/train_loss", train_loss, epoch)
        
        print(f"Epoch {epoch}: Semantic Textual Similarity -> train loss: {train_loss:.3f}")
        
        # profiler
        if profiler:
            prof.stop()      

        # train on sentiment analysis sst
        
        # profiler
        if profiler:
            prof.start()
        
        model.train()
        train_loss = 0
        num_batches = 0
        for batch in tqdm(sst_train_dataloader, desc=f'train-sst-{epoch}', disable=TQDM_DISABLE):
            
            b_ids, b_mask, b_labels = (batch['token_ids'],
                                       batch['attention_mask'], batch['labels'])

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            logits = model(b_ids, b_mask, task_id=0)
            
            # SMART
            if args.smart:
                adv_loss = smart_perturbation.forward(
                    model=model,
                    logits=logits,
                    input_ids_1=b_ids,                
                    attention_mask_1=b_mask,
                    task_id=0,
                    task_type=smart.TaskType.Classification) 
            else:
                adv_loss = 0            
            
            original_loss = F.cross_entropy(logits, b_labels.view(-1), reduction='mean')
            loss = original_loss + adv_loss

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1
            
            # SOPHIA
            # update hession EMA
            if args.optimizer == "sophiag" and num_batches % k == k - 1:                  
                optimizer.update_hessian()
                optimizer.zero_grad()
                
            # profiling step
            if profiler:
                prof.step()
                 # stop after wait + warmup +active +repeat
                if num_batches >= (1 + 1 + 3):
                    break   
                
            # TODO for testing
            if num_batches>=args.num_batches_sst:
                break

        train_loss = train_loss / (num_batches)
        
        # tensorboard
        writer.add_scalar("sst/train_loss", train_loss, epoch)

        print(f"Epoch {epoch}: Sentiment classification -> train loss :: {train_loss :.3f}")
        
        # profiler
        if profiler:
            prof.stop()
        
        # train on paraphrasing Quora Question Pairs qqp       
        # profiler
        if profiler:
            prof.start()
        
        model.train()
        train_loss = 0
        num_batches = 0

        for batch in tqdm(para_train_dataloader, desc=f'train-para-{epoch}', disable=TQDM_DISABLE):            
            
            b_ids1, b_mask1, b_ids2, b_mask2, b_labels = (batch['token_ids_1'], batch['attention_mask_1'],
                                                          batch['token_ids_2'], batch['attention_mask_2'],
                                                          batch['labels'])
            
            b_ids1 = b_ids1.to(device)
            b_mask1 = b_mask1.to(device)
            b_ids2 = b_ids2.to(device)
            b_mask2 = b_mask2.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            logits = model(b_ids1, b_mask1, b_ids2, b_mask2, task_id = 1)
            
            # SMART
            if args.smart:
                adv_loss = smart_perturbation.forward(
                    model=model,
                    logits=logits,
                    input_ids_1=b_ids1,                
                    attention_mask_1=b_mask1,
                    input_ids_2=b_ids2,
                    attention_mask_2=b_mask2,
                    task_id=1,
                    task_type=smart.TaskType.Classification) 
            else:
                adv_loss = 0
                
            original_loss = F.cross_entropy(logits, b_labels.view(-1).float(), reduction='mean')
            loss = original_loss + adv_loss

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1
            
            # SOPHIA
            # update hession EMA
            if args.optimizer == "sophiag" and num_batches % k == k - 1:                  
                optimizer.update_hessian()
                optimizer.zero_grad(set_to_none=True)
            
            # profiling step
            if profiler:
                prof.step()
                 # stop after wait + warmup +active +repeat
                if num_batches >= (1 + 1 + 3):
                    break   
                
            # TODO for testing
            if num_batches >= args.num_batches_para:
                 break            

        train_loss = train_loss / num_batches
        
        # tensorboard
        writer.add_scalar("para/train_loss", train_loss, epoch)
        
        print(f"Epoch {epoch}: Paraphrase Detection -> train loss: {train_loss:.3f}")
        
        # profiler stop after one epoch
        if profiler:
            prof.stop()
            break
        
        # evaluation
        
        (_,train_para_acc, _, _, train_para_prec, train_para_rec, train_para_f1,
         _,train_sst_acc, _, _, train_sst_prec, train_sst_rec, train_sst_f1,
         _,train_sts_corr, *_ )= model_eval_multitask(sst_train_dataloader,
                                                    para_train_dataloader,
                                                    sts_train_dataloader,
                                                    model, device)
         
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
         sts_loss,dev_sts_cor, *_ )= model_eval_multitask(sst_dev_dataloader,
                                                 para_dev_dataloader,
                                                 sts_dev_dataloader,
                                                 model, device)        

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
            #save_model(model, optimizer, args, config, args.filepath)
            if  args.save:
                save_model(model,optimizer,args,config,"Models/epoch"+str(epoch)+"-"+f'{args.option}-{args.lr}-multitask.pt')     

        # cool down GPU    
        if epoch % 5 == 4:
            time.sleep(60*5)                     
        
    # tensorboard
    # collect all information of run    
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

    
    
def test_model(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = smart.SmartMultitaskBERT(config)
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
                        choices=('pretrain', 'finetune'), default="finetune")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    # hyper parameters
    parser.add_argument("--batch_size", help='sst: 64 can fit a 12GB GPU', type=int, default=10)
    parser.add_argument("--num_batches_para", help='sst: 64 can fit a 12GB GPU', type=int, default=float('nan'))
    parser.add_argument("--num_batches_sst", help='sst: 64 can fit a 12GB GPU', type=int, default=float('nan'))
    parser.add_argument("--num_batches_sts", help='sst: 64 can fit a 12GB GPU', type=int, default=float('nan'))
    
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)
    parser.add_argument("--local_files_only", action='store_true', default = True),
    parser.add_argument("--optimizer", type=str, help="adamw or sophiag", choices=("adamw", "sophiag"), default="adamw"),
    parser.add_argument("--weight_decay", help="default for 'adamw': 0.01", type=float, default=0.01),
    parser.add_argument("--k_for_sophia", type=int, help="how often to update the hessian? default is 10", default=10),
    parser.add_argument("--profiler", action="store_true")
    
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--logdir", type=str, default='')
    parser.add_argument("--comment", type=str, default="")
    parser.add_argument("--smart", action="store_true")
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    args.filepath = f'Models/{args.option}-{args.lr}-multitask.pt' # save path for model
    seed_everything(args.seed)  # fix the seed for reproducibility    
    
    train_multitask(args)

    # if not args.profiler:
    #     test_model(args)