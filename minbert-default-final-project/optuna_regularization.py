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
from models import *
# SOPHIA
from optimizer import SophiaG
import smart_utils as smart
# Optuna
import logging
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
        
    # OPTUNA
    for n_tri in range(args.n_trials):
        print(f"Trial {n_tri}")
        trial = study.ask()
        pruned_trial = False    
        
        # Init model
        hidden_dropout_prob = trial.suggest_float("hidden_dropout_prob", 0, 1)
        hidden_dropout_prob2 = trial.suggest_float("hidden_dropout_prob", 0, 1)
    
        config = {'hidden_dropout_prob': hidden_dropout_prob,
                  'hidden_dropout_prob2': hidden_dropout_prob2,
                'num_labels': num_labels,
                'hidden_size': 768,
                'data_dir': '.',
                'option': args.option,
                'local_files_only': args.local_files_only}
        
        n_iter= len(sst_train_dataloader)
        config = SimpleNamespace(**config)

        model = SmartMultitaskBERT(config)
        model = model.to(device)     
        
        # SophiaG         
        lr = 1e-5
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
        optimizer = AdamW(model.parameters(), lr=lr, betas=(0.965, 0.99), weight_decay=weight_decay)

        for epoch in range(args.epochs):
            #train on semantic textual similarity (sts)
            model.train()
            num_batches = 0        

            for batch in tqdm(sts_train_dataloader, desc=f'train-sts-{epoch}', disable=TQDM_DISABLE):
                
                
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
                
                loss = F.mse_loss(logits, b_labels.view(-1).float(), reduction='mean')
                loss.backward()
                optimizer.step()

                num_batches += 1
                    
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
                
                loss = F.cross_entropy(logits, b_labels.view(-1), reduction='mean')
                loss.backward()
                optimizer.step()

                num_batches += 1
                    
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
    if not os.path.exists('optuna'):
        os.makedirs('optuna')
        
    logger = logging.getLogger('optuna')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('optuna/regularization_log.txt')
    logger.addHandler(file_handler)
    
    study = optuna.create_study(direction="maximize", study_name="Regularization",
                                pruner =  optuna.pruners.HyperbandPruner(min_resource=1,
                                                                        max_resource=3))  

    train_multitask(args)
    
    file_handler.close()
    
    number_trials = len(study.trials)    
    ntrial_string = f"Number of finished trials: {number_trials}"
    pruned_trials = len(study.get_trials(deepcopy=False, states=[TrialState.PRUNED]))
    pruned_string = f"Number of pruned trials: {pruned_trials}"
    complete_trials = len(study.get_trials(deepcopy=False, states=[TrialState.COMPLETE]))
    complete_string = f"Number of complete trials: {complete_trials}"
    param_string = "Best value: {} (params: {})\n".format(study.best_value, study.best_params)
    lines = [ntrial_string, pruned_string, complete_string, param_string] 
   
    with open('optuna/regularization.txt', 'w') as f:
        f.write('\n'.join(lines))          
    
    fig = plot_optimization_history(study)
    plt.savefig("optuna/regularization-history.png")
    fig = plot_intermediate_values(study)
    plt.savefig("optuna/regularization-intermediate.png")
    fig = plot_parallel_coordinate(study)
    plt.savefig("optuna/regularization-parallel.png")
    fig = plot_contour(study)
    plt.savefig("optuna/regularization-contour.png")
    fig = plot_slice(study)
    plt.savefig("optuna/regularization-slice.png")
    fig = plot_param_importances(study)
    plt.savefig("optuna/regularization-parameter.png")
    fig = plot_edf(study)
    plt.savefig("optuna/regularization-edf.png")
    fig = plot_rank(study)
    plt.savefig("optuna/regularization-rank.png")
    fig = plot_timeline(study)
    plt.savefig("optuna/regularization-timeline.png")