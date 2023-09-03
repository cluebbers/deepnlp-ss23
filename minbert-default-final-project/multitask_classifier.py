from shared_classifier import *
from datasets import *

import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F

from torch.autograd import Variable

from bert import BertModel
from optimizer import *
from tqdm import tqdm

from datasets import SentenceClassificationDataset, SentencePairDataset, \
    load_multitask_data, load_multitask_test_data

from models import *
from optimizer import SophiaG

from evaluation import *

# CLL import multitask evaluation
from evaluation import smart_eval
# tensorboard
from torch.utils.tensorboard import SummaryWriter

# SMART regularization
from smart_perturbation import SmartPerturbation
import smart_utils as smart
# profiling
from torch.profiler import profile, record_function, ProfilerActivity
from custom_attention import *

from functools import *
from itertools import *

BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5

def load_model(args, device, num_labels):
    config = SimpleNamespace(
        hidden_dropout_prob = args.hidden_dropout_prob,
        hidden_dropout_prob2 = args.hidden_dropout_prob2,
        hidden_dropout_prob_para = args.hidden_dropout_prob_para,
        hidden_dropout_prob_sst = args.hidden_dropout_prob_sst,
        hidden_dropout_prob_sts = args.hidden_dropout_prob_sts,
        num_labels = num_labels,
        hidden_size = 768,
        data_dir = '.',
        option = args.option,
        local_files_only = args.local_files_only,
        attention_module = eval(args.custom_attention)
    )

    if args.shared:
        model = SharedMultitaskBERT(config)
    else:
        model = SmartMultitaskBERT(config)
    
    if args.skip_para or args.option == 'pretrain': #load model trained on para set to continue training mainly on sst and sts set
        print(args.option)
        saved = torch.load(args.filepath)
        config = saved['model_config']
        model = SmartMultitaskBERT(config)
        model.load_state_dict(saved['model'])

    model = model.to(device)
    return model


def load_optimizers(model, args):

    # optimizer choice 
    lr = args.lr

    if args.optimizer == "sophiag":
        if args.para_sep: # those parameters where found with an optuna study for the para dataset
            optimizer_para= SophiaG(model.parameters(), lr=1e-5, betas=(0.959, 0.92), rho = 0.04, weight_decay=0.25)
        else: #try different para_optimizer on the last epochs during training on all datasets
            optimizer_para= SophiaG(model.parameters(), lr=args.lr_para, rho = args.rho_para, weight_decay=args.weight_decay_para)
    
        optimizer_sst = SophiaG(model.parameters(), lr=args.lr_sst,rho=args.rho_sst, weight_decay=args.weight_decay_sst)
        optimizer_sts = SophiaG(model.parameters(), lr=args.lr_sts,rho=args.rho_sts, weight_decay=args.weight_decay_sts)

    else:
        optimizer_para = AdamW(model.parameters(), lr=args.lr_para,weight_decay=args.weight_decay_para)
        optimizer_sst = AdamW(model.parameters(), lr=args.lr_sst,weight_decay=args.weight_decay_sst)
        optimizer_sts = AdamW(model.parameters(), lr=args.lr_sts,weight_decay=args.weight_decay_sts)
        
    return optimizer_para, optimizer_sst, optimizer_sts


def update_optimizer(optimizer, args, num_batches):
    if args.optimizer != "sophiag":
        return
    k = args.k_for_sophia
    if num_batches % k != k - 1:
        return
    optimizer.update_hessian()


def train_step_sts_generator(model, dataloaders, optimizer, epoch, args, perturbation):
    model.train()
    for i, (b_ids1, b_mask1, b_ids2, b_mask2, b_labels) in enumerate(dataloaders.iter_train_sts(epoch)):
        optimizer.zero_grad(set_to_none=True)

        if args.one_embed:
            b_ids = torch.cat([b_ids1,b_ids2],dim=1)
            b_mask = torch.cat([b_mask1,b_mask2],dim=1)
            #logits are unnormalized probabilities, the more negative a logit is the lower the similarity prediction of the model is
            logits = model(b_ids, b_mask, task_id = 2, add_layers = args.add_layers)
        else:
            #return cosine similarity between two embeddings
            logits = model(b_ids1, b_mask1, b_ids2, b_mask2, task_id=2, add_layers=args.add_layers)
        
        # SMART
        if args.smart:
            adv_loss = perturbation.forward(
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

        # we need a loss function for similarity
        # there are different degrees of similarity
        # So maybe the mean squared error is a suitable loss function for the beginning,
        # since it punishes a prediction that is far away from the truth disproportionately
        # more than a prediction that is close to the truth
        if args.one_embed:
            # since similarity is between (0,5) the labels have to be normalized for the cross entropy such that
            # similarity is in (0,5)*0.2=(0,1)
            original_loss = F.binary_cross_entropy_with_logits(logits, b_labels*0.2)
        else:
            original_loss = F.mse_loss(logits, b_labels.view(-1).float(), reduction='mean')

        loss = original_loss + adv_loss
        if args.option == "pretrain":
            loss.requires_grad = True
        loss.backward()
        optimizer.step()
        train_loss = loss.item()
        update_optimizer(optimizer, args, i)
        yield train_loss

def train_step_sst_generator(model, dataloaders, optimizer, epoch, args, perturbation):
    model.train()
    for i, (b_ids, b_mask, b_labels) in enumerate(dataloaders.iter_train_sst(epoch)):
        optimizer.zero_grad(set_to_none=True)
        logits = model(b_ids, b_mask, task_id=0,add_layers=args.add_layers)
        
        # SMART
        if args.smart:
            adv_loss = perturbation.forward(
                model=model,
                logits=logits,
                input_ids_1=b_ids,                
                attention_mask_1=b_mask,
                task_id=0,
                task_type=smart.TaskType.Classification) 
        else:
            adv_loss = 0  
        
        if args.weights:
            w = torch.FloatTensor([2.1,1,1.3,1,1.8]).to(dataloaders.device)
            original_loss = F.cross_entropy(logits, b_labels.view(-1), reduction='mean', weight=w)#,label_smoothing=0.1)
        else:
            original_loss = F.cross_entropy(logits, b_labels.view(-1), reduction='mean')
        
        loss = original_loss + adv_loss
        loss.backward()
        optimizer.step()
        train_loss = loss.item()
        update_optimizer(optimizer, args, i)
        yield train_loss

def train_step_para_generator(model, dataloaders, optimizer, epoch, args, perturbation):
    model.train()
    for i, (b_ids1, b_mask1, b_ids2, b_mask2, b_labels) in enumerate(dataloaders.iter_train_para(epoch)):
        optimizer.zero_grad(set_to_none=True)
        
        if args.one_embed:
            b_ids = torch.cat([b_ids1,b_ids2],dim=1)
            b_mask = torch.cat([b_mask1,b_mask2],dim=1)
            logits = model(b_ids, b_mask, task_id = 1,add_layers= args.add_layers)        
        else:
            logits = model(b_ids1, b_mask1, b_ids2, b_mask2, task_id = 1,add_layers= args.add_layers)

        # SMART
        if args.smart:
            adv_loss = perturbation.forward(
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

        # we need a loss function that handles logits. maybe this one?
        # paraphrasing is a binary task, so binary
        # we get logits, so logits
        # this one also has a sigmoid activation function
        # to balance train data add weights in loss function 
        #there are roughly 1.66 times more non_paraphrase samples as is_paraphrase samples
        #thus weight a positive sample with weight 1.66
        if args.weights:
            w_p = torch.FloatTensor([1.66]).to(dataloaders.device)
            original_loss = F.binary_cross_entropy_with_logits(logits, b_labels.view(-1).float(), reduction='mean', pos_weight=w_p)
        else: 
            original_loss = F.binary_cross_entropy_with_logits(logits, b_labels.view(-1).float(), reduction='mean')
        
        loss = original_loss + adv_loss
        loss.backward()
        optimizer.step()
        train_loss = loss.item()
        update_optimizer(optimizer, args, i)
        yield train_loss

def load_smart_perturbation(args):
    if not args.smart:
        return None, None, None

    smart_loss_sst = smart.SymKlCriterion().forward
    smart_loss_qqp = smart.SymKlCriterion().forward
    smart_loss_sts = smart.MseCriterion().forward
    loss_map = {0: smart_loss_sst, 1: smart_loss_qqp, 2: smart_loss_sts}
    
    if not args.multi_smart:
        params_para = dict()
        params_sst  = dict()
        params_sts  = dict()
    else:
        params_para = dict(
            epsilon=1.7e-7,
            step_size=0.0012,
            noise_var=1.13e-5,
            norm_p='L2',
        )
        params_sst  = dict(
            epsilon=3.9e-6,
            step_size=1.1e-4,
            noise_var=4.2e-6,
            norm_p='inf',
        )
        params_sts  = dict(
            epsilon=4.4e-7,
            step_size=2.4e-3,
            noise_var=1.7e-5,
            norm_p='L2',
        )

    perturbation_para = SmartPerturbation(loss_map = loss_map, **params_para)
    perturbation_sst  = SmartPerturbation(loss_map = loss_map, **params_sst)
    perturbation_sts  = SmartPerturbation(loss_map = loss_map, **params_sts)
    return perturbation_para, perturbation_sst, perturbation_sts
    

def train_multitask(args):
    device      = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    dataloaders = MultitaskDataloader(args, device)
    model       = load_model(args, device, dataloaders.num_labels)
    optimizer_para, optimizer_sst, optimizer_sts = load_optimizers(model, args)
    smart_perturbation_para, smart_perturbation_sst, smart_perturbation_sts = load_smart_perturbation(args)
    writer      = SummaryWriter(comment = args.logdir)

    best_para_dev_acc = 0
    best_sst_dev_acc = 0
    best_sts_dev_cor = 0
    best_dev_acc = 0

    n_iter = dataloaders.para_train_dataloader_size 

    for epoch in range(args.epochs):
        # Mixed training (para), (sst, para), (sts, para, sst)
        sts_generator = train_step_sts_generator(model, dataloaders, optimizer_sts, epoch, args, smart_perturbation_sts)
        sst_generator = train_step_sst_generator(model, dataloaders, optimizer_sst, epoch, args, smart_perturbation_sst)
        para_generator = train_step_para_generator(model, dataloaders, optimizer_para, epoch, args, smart_perturbation_para)

        size_language_training = (dataloaders.para_train_dataloader_size-dataloaders.sst_train_dataloader_size)
        size_language_pretrain = (dataloaders.sst_train_dataloader_size-dataloaders.sts_train_dataloader_size)
        size_language_finetune = dataloaders.sts_train_dataloader_size

        sst_loss = 0
        sst_loss_count = 0
        sts_loss = 0
        sts_loss_count = 0
        para_loss = 0
        para_loss_count = 0

        for loss in islice(para_generator, size_language_training):
            para_loss += loss
            para_loss_count += 1

        for loss in islice(sst_generator, size_language_pretrain):
            sst_loss += loss
            sst_loss_count += 1

        for loss in islice(para_generator, size_language_pretrain):
            para_loss += loss
            para_loss_count += 1

        if args.cyclic_finetuning:
            for sts, para, sst in zip(sts_generator, para_generator, sst_generator):
                sts_loss += sts
                sts_loss_count += 1
                para_loss += para
                para_loss_count += 1
                sst_loss += sst
                sst_loss_count += 1
        else:
            for loss in islice(sts_generator, size_language_finetune):
                sts_loss += loss
                sts_loss_count += 1
            
            for loss in islice(para_generator, size_language_finetune):
                para_loss += loss
                para_loss_count += 1

            for loss in islice(sst_generator, size_language_finetune):
                sst_loss += loss
                sst_loss_count += 1 

        sts_loss = sts_loss / sts_loss_count
        sst_loss = sst_loss / sst_loss_count
        para_loss = para_loss / para_loss_count

        # tensorboard
        writer.add_scalar("sts/train_loss", sts_loss, epoch)
        print(f"Epoch {epoch}: Semantic Textual Similarity -> train loss: {sts_loss:.3f}")
        writer.add_scalar("sst/train_loss", sst_loss, epoch)
        print(f"Epoch {epoch}: Sentiment classification -> train loss :: {sst_loss :.3f}")
        writer.add_scalar("para/train_loss", para_loss, epoch)
        print(f"Epoch {epoch}: Paraphrase Detection -> train loss: {para_loss:.3f}")

        # evaluation
        (_,train_para_acc, _, _, train_para_prec, train_para_rec, train_para_f1,
         _,train_sst_acc, _, _, train_sst_prec, train_sst_rec, train_sst_f1,
         _,train_sts_corr, *_ )= smart_eval(dataloaders.sst_train_dataloader,
                                            dataloaders.para_train_dataloader,
                                            dataloaders.sts_train_dataloader,
                                            model, device, n_iter,one_embed=args.one_embed, add_layers = args.add_layers, args = args)
         
        # tensorboard   
        writer.add_scalar("para/train-acc", train_para_acc, epoch)
        writer.add_scalar("para/train-prec", train_para_prec, epoch)
        writer.add_scalar("para/train-rec", train_para_rec, epoch)
        writer.add_scalar("para/train-f1", train_para_f1, epoch, n_iter)
        
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
         sts_loss,dev_sts_cor, *_ )= smart_eval(dataloaders.sst_dev_dataloader,
                                                dataloaders.para_dev_dataloader,
                                                dataloaders.sts_dev_dataloader,
                                                model, device, n_iter, one_embed=args.one_embed,add_layers = args.add_layers, args = args)        

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
            save_model(model, optimizer_para, args, model.config, args.filepath)
          
        # save each epoch of the trained model for detailed error analysis
        if  args.save:
            save_path = f'Models/epoch-{epoch}-{args.option}-{args.custom_attention}-{args.lr}-multitask.pt'
            save_model(model, optimizer_para, args, model.config, save_path)

        # cool down GPU
        if epoch %10 ==9:
            time.sleep(60*5)                     
        
    # tensorboard
    # collect all information of run    
    writer.add_hparams({"comment": args.comment,
                        "epochs":args.epochs,
                        "optimizer":args.optimizer, 
                        "SMART":args.smart,
                        "lr":args.lr_para, 
                        "weight_decay_para":args.weight_decay_para,
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

        model = SmartMultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        test_model_smart(args, model, device)

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
    parser.add_argument("--batch_size", help='sst: 64 can fit a 12GB GPU', type=int, default=64)
    parser.add_argument("--num_batches_para", help='sst: 64 can fit a 12GB GPU', type=int, default=float('nan'))
    parser.add_argument("--num_batches_sst", help='sst: 64 can fit a 12GB GPU', type=int, default=float('nan'))
    parser.add_argument("--num_batches_sts", help='sst: 64 can fit a 12GB GPU', type=int, default=float('nan'))
    
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.0)
    parser.add_argument("--hidden_dropout_prob_para", type=float, default=0)
    parser.add_argument("--hidden_dropout_prob_sst", type=float, default=0)
    parser.add_argument("--hidden_dropout_prob_sts", type=float, default=0)
    parser.add_argument("--hidden_dropout_prob2", type=float, default=None)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)
    parser.add_argument("--lr_para", type=float,default=1e-5)
    parser.add_argument("--lr_sst", type=float,default=1e-5)
    parser.add_argument("--lr_sts", type=float,default=1e-5)
    parser.add_argument("--local_files_only", action='store_true', default = True),
    # optimizer
    # default parameters are from the sophia optimizer papaer
    parser.add_argument("--optimizer", type=str, help="adamw or sophiag", choices=("adamw", "sophiag"), default="sophiag")
    parser.add_argument("--weight_decay_para", help="default for 'adamw': 0.01", type=float, default=0)
    parser.add_argument("--weight_decay_sst", help="default for 'adamw': 0.01", type=float, default=0)
    parser.add_argument("--weight_decay_sts", help="default for 'adamw': 0.01", type=float, default=0)
    parser.add_argument("--beta1_para", help="first beta parameter of adam/sophia optimizer" ,type=float,default=0.965)
    parser.add_argument("--beta1_sst", help="first beta parameter of adam/sophia optimizer" ,type=float,default=0.965)
    parser.add_argument("--beta1_sts", help="first beta parameter of adam/sophia optimizer" ,type=float,default=0.965)
    parser.add_argument("--beta2_para", help="second beta parameter of adam/sophia optimizer" ,type=float,default=0.99)
    parser.add_argument("--beta2_sst", help="second beta parameter of adam/sophia optimizer" ,type=float,default=0.99)
    parser.add_argument("--beta2_sts", help="second beta parameter of adam/sophia optimizer" ,type=float,default=0.99)
    parser.add_argument("--rho_para", help="rho parameter of sophia optimizer",type=float,default=0.04)
    parser.add_argument("--rho_sst", help="rho parameter of sophia optimizer",type=float,default=0.04)
    parser.add_argument("--rho_sts", help="rho parameter of sophia optimizer",type=float,default=0.04)
    
    parser.add_argument("--k_for_sophia", type=int, help="how often to update the hessian? default is 10", default=10)    
    # tensorboard    
    parser.add_argument("--logdir", type=str, default='')
    parser.add_argument("--comment", type=str, default="")
    parser.add_argument("--profiler", action="store_true")
    parser.add_argument("--custom_attention", type=str, choices = CUSTOM_ATTENTION_CHOICES,
                        help="Which custom attention should be used?", default = "BertSelfAttention")
    parser.add_argument("--cyclic_finetuning", type=bool, default=False)
    parser.add_argument("--shared", action="store_true") # shared model
    parser.add_argument("--smart", action="store_true") # SMART
    parser.add_argument("--multi_smart", help="if True every task uses different smart parameters during training", type=bool, default=False) 
    parser.add_argument("--para_sep", help="if True model is  only trained on para data set", type=bool, default=False) 
    parser.add_argument("--skip_para", help="if True model is only trained on sst and sts data set", type=bool, default=False) 
    parser.add_argument("--weights", help="balance loss function with weights in para and sst", type=bool, default=False)
    parser.add_argument("--add_layers", help="add additional layers in model.predict_para/sts/sst", type=bool, default=False)
    parser.add_argument("--one_embed", help="produce one bert embedding for a sentence pair instead of two seperate ones", type=bool, default=False)
    parser.add_argument("--freeze_bert", help="freeze bert at end of training and only train the tasks classifier", type=bool, default=False)
    #Model saving
    parser.add_argument("--save",help= "save model of every epoch", type=bool, default=False)
    parser.add_argument("--filepath", help= "path where model is saved",type=str, default='')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    if args.filepath == '':
        args.filepath = f'Models/{args.optimizer}-{args.custom_attention}-one_embed-{args.one_embed}-para_sep-{args.para_sep}-weights-{args.weights}-onelayer-{args.add_layers}-multitask.pt' # save path for model
    seed_everything(args.seed)  # fix the seed for reproducibility 
    if args.smart:
        args.comment = "smart"
        train_multitask(args)
    elif args.freeze_bert: #train only the tasks classifiers on the last epochs
    #if add_layers true the model trains the last layers all the time 
    #if add_layers false the model is trained with linear classifiers first
    #those are discarded and replaced by the nn classifier in the last epochs
        args.filepath = f'Models/{args.optimizer}-{args.custom_attention}-one_embed-{args.one_embed}-add_layers-{args.add_layers}-freeze-{args.freeze_bert}-multitask.pt' # save path for model
        train_multitask(args)
        args.epochs = 10
        args.option = 'pretrain' #pretrain option freezes bert parameters
        args.add_layers = True
        train_multitask(args)
        
    else:
        if args.para_sep:
            #train first epochs on para and then train 10 epochs on all three datasets but only a tiny fraction of the para set is used
            train_multitask(args)
            args.skip_para = True
            args.epochs = 10
            args.comment = "para_skip""_weighted_loss"+str(args.weights)+"add_layers"+str(args.add_layers)
            args.para_sep = False
            train_multitask(args)
        else:
            args.comment = "sophia"
            train_multitask(args)
            
    # if not args.profiler:
    test_model(args)
