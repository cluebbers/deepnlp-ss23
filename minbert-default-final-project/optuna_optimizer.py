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
from evaluation import optuna_eval
# SOPHIA
from optimizer_sophia import SophiaG
# SMART regularization
import smart_utils as smart
# Optuna
import optuna
from optuna.trial import TrialState
import matplotlib.pyplot as plt
from optuna.visualization.matplotlib import plot_contour
from optuna.visualization.matplotlib  import plot_edf
from optuna.visualization.matplotlib  import plot_intermediate_values
from optuna.visualization.matplotlib  import plot_optimization_history
from optuna.visualization.matplotlib  import plot_parallel_coordinate
from optuna.visualization.matplotlib  import plot_param_importances
from optuna.visualization.matplotlib  import plot_rank
from optuna.visualization.matplotlib  import plot_slice
from optuna.visualization.matplotlib  import plot_timeline

TQDM_DISABLE=False


def train_multitask(args):       
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    
    # Load data    
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')
    
    # sst
    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)    
    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                    collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)

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
    
    n_iter= len(sst_train_dataloader)
    config = SimpleNamespace(**config)

    model = smart.SmartMultitaskBERT(config)
    model = model.to(device)
    
    # OPTUNA
    for _ in range(args.n_trials):
        # optimizer choice 
        trial = study.ask()
        pruned_trial = False
        optimizer_name = trial.suggest_categorical("Optimizer", ["AdamW", "SophiaG"])
        
        
        # AdamW or SophiaG
        if optimizer_name == "SophiaG":            
            lr_sophia = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
            weight_decay_sophia = trial.suggest_float("weight_decay", 1e-5, 1, log=True)
            rho = trial.suggest_float("rho", 0.1, 0.5) 
            k = trial.suggest_int("k", 5, 20, step=5)  
            optimizer = SophiaG(model.parameters(), lr=lr_sophia, betas=(0.965, 0.99), rho = rho, weight_decay=weight_decay_sophia)
        else:
            lr_adam = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
            weight_decay_adam = trial.suggest_float("weight_decay", 1e-5, 1, log=True)
            optimizer = AdamW(model.parameters(), lr=lr_adam, weight_decay=weight_decay_adam)
             
        for epoch in range(args.epochs):
            #train on semantic textual similarity (sts)
            model.train()
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
                
                loss = F.mse_loss(logits, b_labels.view(-1).float(), reduction='sum')
                loss.backward()
                optimizer.step()

                num_batches += 1
                
                # SOPHIA
                # update hession EMA
                if optimizer_name == "SophiaG" and num_batches % k == k - 1:                  
                    optimizer.update_hessian()
                    optimizer.zero_grad(set_to_none=True)
                    
                if num_batches >= n_iter:
                    break

            # train on sentiment analysis sst        
            model.train()
            num_batches = 0
            for batch in tqdm(sst_train_dataloader, desc=f'train-sst-{epoch}', disable=TQDM_DISABLE):
                
                b_ids, b_mask, b_labels = (batch['token_ids'],
                                        batch['attention_mask'], batch['labels'])

                b_ids = b_ids.to(device)
                b_mask = b_mask.to(device)
                b_labels = b_labels.to(device)

                optimizer.zero_grad()
                logits = model(b_ids, b_mask, task_id=0)     
                
                loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum')
                loss.backward()
                optimizer.step()

                num_batches += 1
                
                # SOPHIA
                # update hession EMA
                if optimizer_name == "SophiaG" and num_batches % k == k - 1:                  
                    optimizer.update_hessian()
                    optimizer.zero_grad()
                    
                if num_batches >= n_iter:
                    break
            
            # train on paraphrasing Quora Question Pairs qqp       
            model.train()
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
                    
                loss = F.cross_entropy(logits, b_labels.view(-1).float(), reduction='mean')
                loss.backward()
                optimizer.step()

                num_batches += 1
                
                # SOPHIA
                # update hession EMA
                if optimizer_name == "SophiaG" and num_batches % k == k - 1:                  
                    optimizer.update_hessian()
                    optimizer.zero_grad(set_to_none=True)    
                    
                if num_batches >= n_iter:
                    break     
            
            # evaluation                
            (paraphrase_accuracy, sts_corr, sentiment_accuracy)= optuna_eval(sst_dev_dataloader,
                                                    para_dev_dataloader,
                                                    sts_dev_dataloader,
                                                    model, device, n_iter) 
            if np.isnan(sts_corr):
                sts_corr = 0
            epoch_acc = (paraphrase_accuracy + sts_corr + sentiment_accuracy) / 3 
            trial.report(epoch_acc, epoch)
            if trial.should_prune():
                pruned_trial = True
                break
            
        if pruned_trial:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
        else:       
            study.tell(trial, epoch_acc, state=TrialState.COMPLETE)       

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
    parser.add_argument("--epochs", type=int, default=3)
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
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--local_files_only", action='store_true', default = True)
    parser.add_argument("--n_trials", type=int, default=100)
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)  # fix the seed for reproducibility    
    study = optuna.create_study(direction="minimize", study_name="Optimizer",
                                pruner =  optuna.pruners.HyperbandPruner(min_resource=1,
                                                                        max_resource=3))
    train_multitask(args)
    
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial
    
    if not os.path.exists('optuna'):
        os.makedirs('optuna')
    fig = plot_optimization_history(study)
    plt.savefig("optuna/optimizer-history.png")
    fig = plot_intermediate_values(study)
    plt.savefig("optuna/optimizer-intermediate.png")
    fig = plot_parallel_coordinate(study)
    plt.savefig("optuna/optimizer-parallel.png")
    fig = plot_contour(study)
    plt.savefig("optuna/optimizer-contour.png")
    fig = plot_slice(study)
    plt.savefig("optuna/optimizer-slice.png")
    fig = plot_param_importances(study)
    plt.savefig("optuna/optimizer-parameter.png")
    fig = plot_edf(study)
    plt.savefig("optuna/optimizer-edf.png")
    fig = plot_rank(study)
    plt.savefig("optuna/optimizer-rank.png")
    fig = plot_timeline(study)
    plt.savefig("optuna/optimizer-timeline.png")
                
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

