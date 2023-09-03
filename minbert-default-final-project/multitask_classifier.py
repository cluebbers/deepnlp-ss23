from shared_classifier import *

import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
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
    sts_train_data = SentencePairDataset(sts_train_data, args,isRegression =True)
    sts_dev_data = SentencePairDataset(sts_dev_data, args,isRegression =True)
    
    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_data.collate_fn)   
    # CLL end

    # Init model
    
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'hidden_dropout_prob2': args.hidden_dropout_prob2,
              'hidden_dropout_prob_para': args.hidden_dropout_prob_para,
              'hidden_dropout_prob_sst': args.hidden_dropout_prob_sst,
              'hidden_dropout_prob_sts': args.hidden_dropout_prob_sts,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option,
              'local_files_only': args.local_files_only}
    
    n_iter= len(sst_train_dataloader)

    config = SimpleNamespace(**config)
    
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
    
    # optimizer choice 
    lr = args.lr
    #weight_decay = args.weight_decay

    if args.optimizer == "sophiag":
        if args.para_sep: # those parameters where found with an optuna study for the para dataset
            optimizer_para= SophiaG(model.parameters(), lr=1e-5, betas=(0.959, 0.92), rho = 0.04, weight_decay=0.25)
        else: #try different para_optimizer on the last epochs during training on all datasets
            optimizer_para= SophiaG(model.parameters(), lr=args.lr_para, rho = args.rho_para, weight_decay=args.weight_decay_para)
    
        optimizer_sst = SophiaG(model.parameters(), lr=args.lr_sst,rho=args.rho_sst, weight_decay=args.weight_decay_sst)
        optimizer_sts = SophiaG(model.parameters(), lr=args.lr_sts,rho=args.rho_sts, weight_decay=args.weight_decay_sts)
        #how often to update the hessian?
        k = args.k_for_sophia
        
    else:
        optimizer_para = AdamW(model.parameters(), lr=args.lr_para,weight_decay=args.weight_decay_para)
        optimizer_sst = AdamW(model.parameters(), lr=args.lr_sst,weight_decay=args.weight_decay_sst)
        optimizer_sts = AdamW(model.parameters(), lr=args.lr_sts,weight_decay=args.weight_decay_sts)
        
    # SMART    
    if args.smart:
        smart_loss_sst = smart.SymKlCriterion().forward
        smart_loss_qqp = smart.SymKlCriterion().forward
        smart_loss_sts = smart.MseCriterion().forward
        if args.multi_smart: #those values are from an optuna study on smart
            smart_perturbation_para = SmartPerturbation(epsilon=1.7e-7,
                                                   step_size=0.0012,
                                                   noise_var=1.13e-5,
                                                   norm_p='L2',
                                                   loss_map={0:smart_loss_sst, 1:smart_loss_qqp, 2:smart_loss_sts})
            
            smart_perturbation_sst = SmartPerturbation(epsilon=3.9e-6,
                                                   step_size=1.1e-4,
                                                   noise_var=4.2e-6,
                                                   norm_p='inf',
                                                   loss_map={0:smart_loss_sst, 1:smart_loss_qqp, 2:smart_loss_sts})
            smart_perturbation_sts = SmartPerturbation(epsilon=4.4e-7,
                                                   step_size=2.4e-3,
                                                   noise_var=1.7e-5,
                                                   norm_p='L2',
                                                   loss_map={0:smart_loss_sst, 1:smart_loss_qqp, 2:smart_loss_sts})
        else:
            smart_perturbation_para = SmartPerturbation(loss_map={0:smart_loss_sst, 1:smart_loss_qqp, 2:smart_loss_sts})
            smart_perturbation_sst = SmartPerturbation(loss_map={0:smart_loss_sst, 1:smart_loss_qqp, 2:smart_loss_sts})
            smart_perturbation_sts = SmartPerturbation(loss_map={0:smart_loss_sst, 1:smart_loss_qqp, 2:smart_loss_sts})
       
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
    
    n_iter= len(sst_train_dataloader)
                  
    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        # train on paraphrasing
        # CLL add training on other tasks
        # paraphrasing
        # see evaluation.py line 72
        
        # profiler
        if profiler:
            prof.start()
        
        model.train()
        train_loss = 0
        num_batches = 0
        
        # datasets.py line 145
        for batch in tqdm(para_train_dataloader, desc=f'train-para-{epoch}', disable=TQDM_DISABLE):            
            if args.skip_para: #train the last epochs only on a small fraction of the para data
                rand = np.random.uniform()
                if rand >= len(sst_train_dataloader)/(2*len(para_train_dataloader)): #train on batch only with a certain probability 
                #-> the mean of trained batches is half of the number of batches as in the sst set(the smallest dataset of all three)
                    continue
                
            b_ids1, b_mask1, b_ids2, b_mask2, b_labels = (batch['token_ids_1'], batch['attention_mask_1'],
                                                          batch['token_ids_2'], batch['attention_mask_2'],
                                                          batch['labels'])
            
            b_ids1 = b_ids1.to(device)
            b_mask1 = b_mask1.to(device)
            b_ids2 = b_ids2.to(device)
            b_mask2 = b_mask2.to(device)
            b_labels = b_labels.to(device)
            
            optimizer_para.zero_grad()
            
            if args.one_embed:
                b_ids = torch.cat([b_ids1,b_ids2],dim=1)
                b_mask = torch.cat([b_mask1,b_mask2],dim=1)
                logits = model(b_ids, b_mask, task_id = 1,add_layers= args.add_layers)
            
            else:
                logits = model(b_ids1, b_mask1, b_ids2, b_mask2, task_id = 1,add_layers= args.add_layers)
            
            # SMART
            if args.smart:
                adv_loss = smart_perturbation_para.forward(
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
                w_p = torch.FloatTensor([1.66]).to(device)
                original_loss = F.binary_cross_entropy_with_logits(logits, b_labels.view(-1).float(), reduction='mean', pos_weight=w_p)
            else: 
                original_loss = F.binary_cross_entropy_with_logits(logits, b_labels.view(-1).float(), reduction='mean')
           
            loss = original_loss + adv_loss

            loss.backward()
            optimizer_para.step()

            train_loss += loss.item()
            num_batches += 1
            
            # SOPHIA
            # update hession EMA
            if args.optimizer == "sophiag" and num_batches % k == k - 1:  
                optimizer_para.update_hessian()
                optimizer_para.zero_grad()         
            
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
        
        
        #train on semantic textual similarity (sts)
        
        # profiler start
        if profiler:
            prof.start()
        
        model.train()
        train_loss = 0
        num_batches = 0        
         
        for batch in tqdm(sts_train_dataloader, desc=f'train-sts-{epoch}', disable=TQDM_DISABLE):
            if args.para_sep: #train only on para set 
                break
            b_ids1, b_mask1, b_ids2, b_mask2, b_labels = (batch['token_ids_1'], batch['attention_mask_1'],
                                                          batch['token_ids_2'], batch['attention_mask_2'],
                                                          batch['labels'])
            
            b_ids1 = b_ids1.to(device)
            b_mask1 = b_mask1.to(device)
            b_ids2 = b_ids2.to(device)
            b_mask2 = b_mask2.to(device)
            b_labels = b_labels.to(device)

            optimizer_sts.zero_grad(set_to_none=True)
            
            if args.one_embed:
                b_ids = torch.cat([b_ids1,b_ids2],dim=1)
                b_mask = torch.cat([b_mask1,b_mask2],dim=1)
                #logits are unnormalized probabilities, the more negative a logit is the lower the similarity prediction of the model is
                logits = model(b_ids, b_mask, task_id = 2,add_layers= args.add_layers)
    
            else:
                #return cosine similarity between two embeddings
                similarity = model(b_ids1, b_mask1, b_ids2, b_mask2, task_id=2,add_layers=args.add_layers)
            
            # SMART
            if args.smart:
                adv_loss = smart_perturbation_sts.forward(
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
                original_loss = F.mse_loss(similarity, b_labels.view(-1).float(), reduction='mean')
            
            loss = original_loss + adv_loss          
                
            #if args.option == "pretrain":
                #loss.requires_grad = True
            loss.backward()
            optimizer_sts.step()

            train_loss += loss.item()
            num_batches += 1
            
            # SOPHIA
            # update hession EMA
            if args.optimizer == "sophiag" and num_batches % k == k - 1:  
                optimizer_sts.update_hessian()
                optimizer_sts.zero_grad()       
            
            # profiling step
            if profiler:
                prof.step()
                 # stop after wait + warmup +active *repeat
                if num_batches >= (1+1+3):
                    break   
            
            # TODO for testing
            if num_batches >= args.num_batches_sts:
                break
        if not args.para_sep:  
            train_loss = train_loss / num_batches
        else:
            train_loss = 0
        # tensorboard
        writer.add_scalar("sts/train_loss", train_loss, epoch)
        
        print(f"Epoch {epoch}: Semantic Textual Similarity -> train loss: {train_loss:.3f}")
        
        # profiler
        if profiler:
            prof.stop()      

        # train on sentiment analysis
        
        # profiler
        if profiler:
            prof.start()
        
        model.train()
        train_loss = 0
        num_batches = 0
        for batch in tqdm(sst_train_dataloader, desc=f'train-sst-{epoch}', disable=TQDM_DISABLE):
            if args.para_sep: #train only on para set 
                break
            b_ids, b_mask, b_labels = (batch['token_ids'],
                                       batch['attention_mask'], batch['labels'])

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            b_labels = b_labels.to(device)

            optimizer_sst.zero_grad()
            logits = model(b_ids, b_mask, task_id=0,add_layers=args.add_layers)
            
            # SMART
            if args.smart:
                adv_loss = smart_perturbation_sst.forward(
                    model=model,
                    logits=logits,
                    input_ids_1=b_ids,                
                    attention_mask_1=b_mask,
                    task_id=0,
                    task_type=smart.TaskType.Classification) 
            else:
                adv_loss = 0  
                
            #to balance train data add weights in loss function
            # sentiment class 1 has most samples
            # class 1 has roughly 2.1 times more samples as class 0
            # class 1 has roughly 1.3 times more samples as class 2
            # class 1 and class 3 are almost of the same size
            # class 1 has roughly 1.8 times more samples as class 4
            w = torch.FloatTensor([2.1,1,1.3,1,1.8]).to(device)
            if args.weights:
                original_loss = F.cross_entropy(logits, b_labels.view(-1), reduction='mean', weight=w)#,label_smoothing=0.1)
            else:
                original_loss = F.cross_entropy(logits, b_labels.view(-1), reduction='mean')
           
            loss = original_loss + adv_loss

            loss.backward()
            optimizer_sst.step()

            train_loss += loss.item()
            num_batches += 1
            
            # SOPHIA
            # update hession EMA
            if args.optimizer == "sophiag" and num_batches % k == k - 1:  
                optimizer_sst.update_hessian()
                optimizer_sst.zero_grad()         
                
            # profiling step
            if profiler:
                prof.step()
                 # stop after wait + warmup +active +repeat
                if num_batches >= (1 + 1 + 3):
                    break   
                
            # TODO for testing
            if num_batches>=args.num_batches_sst:
                break
        if not args.para_sep: 
            train_loss = train_loss / (num_batches)
        else:
            train_loss = 0
        
        # tensorboard
        writer.add_scalar("sst/train_loss", train_loss, epoch)

        print(f"Epoch {epoch}: Sentiment classification -> train loss :: {train_loss :.3f}")
        
        # profiler
        if profiler:
            prof.stop()
        '''
        # train on paraphrasing
        # CLL add training on other tasks
        # paraphrasing
        # see evaluation.py line 72
        
        # profiler
        if profiler:
            prof.start()
        
        model.train()
        train_loss = 0
        num_batches = 0
        
        # datasets.py line 145
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
            
            # we need a loss function that handles logits. maybe this one?
            # paraphrasing is a binary task, so binary
            # we get logits, so logits
            # this one also has a sigmoid activation function
            original_loss = F.binary_cross_entropy_with_logits(logits, b_labels.view(-1).float(), reduction='mean')
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
        '''
        # evaluation
        
        (_,train_para_acc, _, _, train_para_prec, train_para_rec, train_para_f1,
         _,train_sst_acc, _, _, train_sst_prec, train_sst_rec, train_sst_f1,
         _,train_sts_corr, *_ )= smart_eval(sst_train_dataloader,
                                                    para_train_dataloader,
                                                    sts_train_dataloader,
                                                    model, device, n_iter,one_embed=args.one_embed, add_layers = args.add_layers)
         
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
         sts_loss,dev_sts_cor, *_ )= smart_eval(sst_dev_dataloader,
                                                 para_dev_dataloader,
                                                 sts_dev_dataloader,
                                                 model, device, n_iter,one_embed=args.one_embed,add_layers = args.add_layers)        

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
            save_model(model, optimizer_para, args, config, args.filepath)
          
        # save each epoch of the trained model for detailed error analysis
        if  args.save:
            save_model(model,optimizer_para,args,config,"Models/epoch"+str(epoch)+"-"+f'{args.option}-{args.lr_para}-multitask.pt')      
        
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
                        "hidden_dropout_prob_para": args.hidden_dropout_prob,
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

        model = SmartMultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        para_acc,sst_acc,sts_cor,embed,labels = test_model_smart(args, model, device)
    return model,para_acc,sst_acc,sts_cor,embed,labels

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
    
    #training and model adjustments
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
        args.filepath = f'Models/{args.optimizer}-one_embed-{args.one_embed}-para_sep-{args.para_sep}-weights-{args.weights}-onelayer-{args.add_layers}-multitask.pt' # save path for model
    seed_everything(args.seed)  # fix the seed for reproducibility 
    if args.smart:
        args.comment = "smart"
        train_multitask(args)
    elif args.freeze_bert: #train only the tasks classifiers on the last epochs
    #if add_layers true the model trains the last layers all the time 
    #if add_layers false the model is trained with linear classifiers first
    #those are discarded and replaced by the nn classifier in the last epochs
        args.filepath = f'Models/{args.optimizer}-one_embed-{args.one_embed}-add_layers-{args.add_layers}-freeze-{args.freeze_bert}-multitask.pt' # save path for model
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
