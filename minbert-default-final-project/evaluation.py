#!/usr/bin/env python3

'''
Model evaluation functions.

When training your multitask model, you will find it useful to run
model_eval_multitask to be able to evaluate your model on the 3 tasks in the
development set.

Before submission, your code needs to call test_model_multitask(args, model, device) to generate
your predictions. We'll evaluate these predictions against our labels on our end,
which is how the leaderboard will be updated.
The provided test_model() function in multitask_classifier.py **already does this for you**,
so unless you change it you shouldn't need to call anything from here
explicitly aside from model_eval_multitask.
'''

import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score, recall_score, accuracy_score, precision_score
from tqdm import tqdm
import numpy as np
from datasets import *


TQDM_DISABLE = False

# Evaluate a multitask model for accuracy.on SST only.
def model_eval_sst(dataloader, model, device):
    model.eval()  # switch to eval model, will turn off randomness like dropout
    y_true = []
    y_pred = []
    sents = []
    sent_ids = []
    for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
        b_ids, b_mask, b_labels, b_sents, b_sent_ids = batch['token_ids'],batch['attention_mask'],  \
                                                        batch['labels'], batch['sents'], batch['sent_ids']

        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)

        logits = model.predict_sentiment(b_ids, b_mask)
        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1).flatten()

        b_labels = b_labels.flatten()
        y_true.extend(b_labels)
        y_pred.extend(preds)
        sents.extend(b_sents)
        sent_ids.extend(b_sent_ids)

    f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)

    return acc, f1, y_pred, y_true, sents, sent_ids

# Perform model evaluation in terms by averaging accuracies across tasks.
def model_eval_multitask(model, device, dataloaders, dev):
    model.eval()  # switch to eval model, will turn off randomness like dropout

    with torch.no_grad():
        para_y_true = []
        para_y_pred = []
        para_sent_ids = []
        para_loss = 0
        num_batches = 0

        # Evaluate paraphrase detection.
        for step, (b_ids1, b_mask1, b_ids2, b_mask2, b_labels, b_sent_ids) in enumerate(dataloaders.iter_eval_para(dev)):

            logits = model(b_ids1, b_mask1, b_ids2, b_mask2, task_id = 1)
            #logits = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2)
            
            #dev_loss 
            
            loss = F.binary_cross_entropy_with_logits(logits, b_labels.float(), reduction='mean')
            para_loss += loss.item()
            num_batches+=1
            
            y_hat = logits.sigmoid().round().flatten().cpu().numpy()
            b_labels = b_labels.flatten().cpu().numpy()

            para_y_pred.extend(y_hat)
            para_y_true.extend(b_labels)
            para_sent_ids.extend(b_sent_ids)
            
        para_loss = para_loss/num_batches #normalize loss

        # paraphrase_accuracy = np.mean(np.array(para_y_pred) == np.array(para_y_true))
        paraphrase_accuracy = accuracy_score(para_y_true, para_y_pred)
        paraphrase_precision = precision_score(para_y_true, para_y_pred, average="binary")
        paraphrase_recall = recall_score(para_y_true, para_y_pred, average="binary")
        paraphrase_f1 = f1_score(para_y_true, para_y_pred, average="binary")

        sts_y_true = []
        sts_y_pred = []
        sts_sent_ids = []
        sts_loss = 0
        num_batches = 0


        # Evaluate semantic textual similarity. (sts)
        for step, (b_ids1, b_mask1, b_ids2, b_mask2, b_labels, b_sent_ids) in enumerate(dataloaders.iter_eval_sts(dev)):

            logits = model(b_ids1, b_mask1, b_ids2, b_mask2, task_id=2)
            
            #dev loss
            loss = F.mse_loss(logits, b_labels.float(), reduction='mean')
            sts_loss += loss.item()
            num_batches+=1            
            y_hat = logits.flatten().cpu().numpy()
            b_labels = b_labels.flatten().cpu().numpy()

            sts_y_pred.extend(y_hat)
            sts_y_true.extend(b_labels)
            sts_sent_ids.extend(b_sent_ids)
            
        sts_loss = sts_loss/num_batches #normalize loss
        
        pearson_mat = np.corrcoef(sts_y_pred,sts_y_true)
        sts_corr = pearson_mat[1][0]


        sst_y_true = []
        sst_y_pred = []
        sst_sent_ids = []
        sst_loss = 0
        num_batches = 0

        # Evaluate sentiment classification. (sst)
        embed = torch.empty(64,768).to(device) #bert embeddings of the sentences, return them for error-analysis
        labels = torch.empty(64).to(device) #labels of the sentences
        for step, (b_ids, b_mask, b_labels, b_sent_ids) in enumerate(dataloaders.iter_eval_sst(dev)):
        
            if num_batches <8: #collect bert_embbedings of the first batches
                embed_new = model.bert(b_ids, b_mask)['pooler_output']
                embed = torch.cat([embed,embed_new])
                labels = torch.cat([labels,b_labels])
            
            logits = model(b_ids, b_mask, task_id=0)
            #logits = model.predict_sentiment(b_ids, b_mask)
            
            #Dev loss
            loss = F.cross_entropy(logits, b_labels.view(-1), reduction='mean')
            sst_loss += loss.item()
            num_batches+=1
            
            y_hat = logits.argmax(dim=-1).flatten().cpu().numpy()
            b_labels = b_labels.flatten().cpu().numpy()

            sst_y_pred.extend(y_hat)
            sst_y_true.extend(b_labels)
            sst_sent_ids.extend(b_sent_ids)

        sst_loss = sst_loss/ num_batches #normalize loss

        # sentiment_accuracy = np.mean(np.array(sst_y_pred) == np.array(sst_y_true))
        sentiment_accuracy = accuracy_score(sst_y_true, sst_y_pred)
        sentiment_precision = precision_score(sst_y_true, sst_y_pred, average=None)
        sentiment_recall = recall_score(sst_y_true, sst_y_pred, average=None)
        sentiment_f1 = f1_score(sst_y_true, sst_y_pred, average=None)

        print(f'Paraphrase detection accuracy: {paraphrase_accuracy:.3f}')
        print(f'Sentiment classification accuracy: {sentiment_accuracy:.3f}')
        print(f'Semantic Textual Similarity correlation: {sts_corr:.3f}')

        return (para_loss,paraphrase_accuracy, para_y_pred, para_sent_ids, paraphrase_precision, paraphrase_recall, paraphrase_f1, 
                sst_loss,sentiment_accuracy,sst_y_pred, sst_sent_ids, sentiment_precision, sentiment_recall, sentiment_f1,
                sts_loss,sts_corr, sts_y_pred, sts_sent_ids,embed,labels)

# Perform model evaluation in terms by averaging accuracies across tasks.
def model_eval_test_multitask(sentiment_dataloader,
                         paraphrase_dataloader,
                         sts_dataloader,
                         model, device):
    model.eval()  # switch to eval model, will turn off randomness like dropout

    with torch.no_grad():

        para_y_pred = []
        para_sent_ids = []
        # Evaluate paraphrase detection.
        for step, batch in enumerate(tqdm(paraphrase_dataloader, desc=f'eval', disable=TQDM_DISABLE)):
            (b_ids1, b_mask1,
             b_ids2, b_mask2,
             b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'],
                          batch['token_ids_2'], batch['attention_mask_2'],
                          batch['sent_ids'])

            b_ids1 = b_ids1.to(device)
            b_mask1 = b_mask1.to(device)
            b_ids2 = b_ids2.to(device)
            b_mask2 = b_mask2.to(device)

            logits = model(b_ids1, b_mask1, b_ids2, b_mask2, task_id = 1)
            #logits = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2)
            y_hat = logits.sigmoid().round().flatten().cpu().numpy()

            para_y_pred.extend(y_hat)
            para_sent_ids.extend(b_sent_ids)


        sts_y_pred = []
        sts_sent_ids = []


        # Evaluate semantic textual similarity.
        for step, batch in enumerate(tqdm(sts_dataloader, desc=f'eval', disable=TQDM_DISABLE)):
            (b_ids1, b_mask1,
             b_ids2, b_mask2,
             b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'],
                          batch['token_ids_2'], batch['attention_mask_2'],
                          batch['sent_ids'])

            b_ids1 = b_ids1.to(device)
            b_mask1 = b_mask1.to(device)
            b_ids2 = b_ids2.to(device)
            b_mask2 = b_mask2.to(device)

            logits = model(b_ids1, b_mask1, b_ids2, b_mask2, task_id = 2)
            #logits = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2)
            y_hat = logits.flatten().cpu().numpy()

            sts_y_pred.extend(y_hat)
            sts_sent_ids.extend(b_sent_ids)


        sst_y_pred = []
        sst_sent_ids = []

        # Evaluate sentiment classification.
        for step, batch in enumerate(tqdm(sentiment_dataloader, desc=f'eval', disable=TQDM_DISABLE)):
            b_ids, b_mask, b_sent_ids = batch['token_ids'], batch['attention_mask'],  batch['sent_ids']

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            
            logits = model(b_ids, b_mask, task_id=0)
            #logits = model.predict_sentiment(b_ids, b_mask)
            y_hat = logits.argmax(dim=-1).flatten().cpu().numpy()

            sst_y_pred.extend(y_hat)
            sst_sent_ids.extend(b_sent_ids)

            

        return (para_y_pred, para_sent_ids,
                sst_y_pred, sst_sent_ids,
                sts_y_pred, sts_sent_ids)


def test_model_multitask(args, model, device):
    dataloaders = MultitaskDataloader(args, device, enable_test = True)

    _,dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids,_,_,_, \
        _,dev_sentiment_accuracy,dev_sst_y_pred, dev_sst_sent_ids,_,_,_,_,dev_sts_corr, \
        dev_sts_y_pred, dev_sts_sent_ids, embed, labels = model_eval_multitask(model, device, dataloaders, dev = True)

    test_para_y_pred, test_para_sent_ids, test_sst_y_pred, \
        test_sst_sent_ids, test_sts_y_pred, test_sts_sent_ids = \
            model_eval_test_multitask(dataloaders.sst_test_dataloader,
                                      dataloaders.para_test_dataloader,
                                      dataloaders.sts_test_dataloader, model, device)

    with open(args.sst_dev_out, "w+") as f:
        print(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}")
        f.write(f"id,Predicted_Sentiment\n")
        for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
            f.write(f"{p},{s}\n")

    with open(args.sst_test_out, "w+") as f:
        f.write(f"id,Predicted_Sentiment")
        for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
            f.write(f"{p},{s}\n")

    with open(args.para_dev_out, "w+") as f:
        print(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}")
        f.write(f"id,Predicted_Is_Paraphrase \n")
        for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
            f.write(f"{p},{s}\n")

    with open(args.para_test_out, "w+") as f:
        f.write(f"id,Predicted_Is_Paraphrase \n")
        for p, s in zip(test_para_sent_ids, test_para_y_pred):
            f.write(f"{p},{s}\n")

    with open(args.sts_dev_out, "w+") as f:
        print(f"dev sts corr :: {dev_sts_corr :.3f}")
        f.write(f"id,Predicted_Similiary \n")
        for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
            f.write(f"{p},{s}\n")

    with open(args.sts_test_out, "w+") as f:
        f.write(f"id,Predicted_Similiary \n")
        for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
            f.write(f"{p},{s}\n")

    return dev_paraphrase_accuracy,dev_sentiment_accuracy,dev_sts_corr, embed,labels

def optuna_eval(sentiment_dataloader,
                         paraphrase_dataloader,
                         sts_dataloader,
                         model, device, n_iter, one_embed = False, args = None,add_layers = False):
    model.eval()  # switch to eval model, will turn off randomness like dropout
    
    with torch.no_grad():
        # Evaluate paraphrase detection. qqp
        para_y_true = []
        para_y_pred = []        
        num_batches = 0
        
        for step, batch in enumerate(tqdm(paraphrase_dataloader, desc=f'eval-para', disable=False)):
            (b_ids1, b_mask1,
             b_ids2, b_mask2,
             b_labels) = (batch['token_ids_1'], batch['attention_mask_1'],
                                      batch['token_ids_2'], batch['attention_mask_2'],
                                      batch['labels'])

            b_ids1 = b_ids1.to(device)
            b_mask1 = b_mask1.to(device)
            b_ids2 = b_ids2.to(device)
            b_mask2 = b_mask2.to(device)
            b_labels = b_labels.to(device)

            if one_embed:
                b_ids = torch.cat([b_ids1,b_ids2],dim=1)
                b_mask = torch.cat([b_mask1,b_mask2],dim=1)
                logits = model(b_ids, b_mask, task_id = 1,add_layers=add_layers)
            else:
                logits = model(b_ids1, b_mask1, b_ids2, b_mask2, task_id=1,add_layers=add_layers) 
            y_hat = logits.sigmoid().round().flatten().cpu().numpy()
            b_labels = b_labels.flatten().cpu().numpy()

            para_y_pred.extend(y_hat)
            para_y_true.extend(b_labels)
            
            num_batches +=1
            if num_batches >= n_iter:
                break

            if args and num_batches >= args.num_batches_para:
                break

        paraphrase_accuracy = accuracy_score(para_y_true, para_y_pred)
        
        # Evaluate semantic textual similarity. (sts)
        sts_y_true = []
        sts_y_pred = []
        
        for step, batch in enumerate(tqdm(sts_dataloader, desc=f'eval-sts', disable=False)):
            (b_ids1, b_mask1,
             b_ids2, b_mask2,
             b_labels) = (batch['token_ids_1'], batch['attention_mask_1'],
                          batch['token_ids_2'], batch['attention_mask_2'],
                          batch['labels'])

            b_ids1 = b_ids1.to(device)
            b_mask1 = b_mask1.to(device)
            b_ids2 = b_ids2.to(device)
            b_mask2 = b_mask2.to(device)
            b_labels = b_labels.to(device)

            if one_embed:
                b_ids = torch.cat([b_ids1,b_ids2],dim=1)
                b_mask = torch.cat([b_mask1,b_mask2],dim=1)
                #logits are unnormalized probabilities, the more negative a logit is the lower the similarity prediction of the model is
                logits = model(b_ids, b_mask, task_id = 2,add_layers=add_layers)
    
            else:
                similarity = model(b_ids1, b_mask1, b_ids2, b_mask2, task_id=2, add_layers=add_layers)

            if one_embed:
                y_hat = logits.sigmoid().flatten().cpu().numpy()*5 #unnormalize predictions after passing them through sigmoid function
            else:
                y_hat = similarity.flatten().cpu().numpy()
            b_labels = b_labels.flatten().cpu().numpy()

            sts_y_pred.extend(y_hat)
            sts_y_true.extend(b_labels)

            if args and step >= args.num_batches_sts:
                break
            
        pearson_mat = np.corrcoef(sts_y_pred,sts_y_true)
        sts_corr = pearson_mat[1][0]
            
        # Evaluate sentiment classification. (sst)
        sst_y_true = []
        sst_y_pred = []
        
        for step, batch in enumerate(tqdm(sentiment_dataloader, desc=f'eval-sst', disable=False)):
            b_ids, b_mask, b_labels = batch['token_ids'], batch['attention_mask'], batch['labels']

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            b_labels = b_labels.to(device)

            logits = logits = model(b_ids, b_mask, task_id=0, add_layers = add_layers)
 
            y_hat = logits.argmax(dim=-1).flatten().cpu().numpy()
            b_labels = b_labels.flatten().cpu().numpy()

            sst_y_pred.extend(y_hat)
            sst_y_true.extend(b_labels)

            if args and step >= args.num_batches_sst:
                break

        sentiment_accuracy = accuracy_score(sst_y_true, sst_y_pred)

        return (paraphrase_accuracy, sts_corr, sentiment_accuracy)
    
def smart_eval(sentiment_dataloader, paraphrase_dataloader, sts_dataloader, model,
               device, n_iter, one_embed = False,add_layers = False, args = None):
    #if one_embed is true, the prediction on para and sts is a bit different
    model.eval()  # switch to eval model, will turn off randomness like dropout

    with torch.no_grad():
        para_y_true = []
        para_y_pred = []
        para_sent_ids = []
        para_loss = 0
        num_batches = 0

        # Evaluate paraphrase detection. qqp
        for step, batch in enumerate(tqdm(paraphrase_dataloader, desc=f'eval-para', disable=TQDM_DISABLE)):
            (b_ids1, b_mask1,
             b_ids2, b_mask2,
             b_labels, b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'],
                          batch['token_ids_2'], batch['attention_mask_2'],
                          batch['labels'], batch['sent_ids'])

            b_ids1 = b_ids1.to(device)
            b_mask1 = b_mask1.to(device)
            b_ids2 = b_ids2.to(device)
            b_mask2 = b_mask2.to(device)
            b_labels = b_labels.to(device)
            
            if one_embed:
                b_ids = torch.cat([b_ids1,b_ids2],dim=1)
                b_mask = torch.cat([b_mask1,b_mask2],dim=1)
                logits = model(b_ids, b_mask, task_id = 1,add_layers=add_layers)
            else:
                logits = model(b_ids1, b_mask1, b_ids2, b_mask2, task_id=1,add_layers=add_layers)
            
            #dev_loss             
            loss = F.binary_cross_entropy_with_logits(logits, b_labels.float(), reduction='mean')
            para_loss += loss.item()
            num_batches+=1            
            
            y_hat = logits.sigmoid().round().flatten().cpu().numpy()
            b_labels = b_labels.flatten().cpu().numpy()

            para_y_pred.extend(y_hat)
            para_y_true.extend(b_labels)
            para_sent_ids.extend(b_sent_ids)
            
            if num_batches >= n_iter:
                break

            if args and num_batches >= args.num_batches_para:
                break

        para_loss = para_loss/num_batches #normalize loss

        # paraphrase_accuracy = np.mean(np.array(para_y_pred) == np.array(para_y_true))
        paraphrase_accuracy = accuracy_score(para_y_true, para_y_pred)
        paraphrase_precision = precision_score(para_y_true, para_y_pred, average="binary")
        paraphrase_recall = recall_score(para_y_true, para_y_pred, average="binary")
        paraphrase_f1 = f1_score(para_y_true, para_y_pred, average="binary")

        sts_y_true = []
        sts_y_pred = []
        sts_sent_ids = []
        sts_loss = 0
        num_batches = 0

        # Evaluate semantic textual similarity. (sts)
        for step, batch in enumerate(tqdm(sts_dataloader, desc=f'eval-sts', disable=TQDM_DISABLE)):
            (b_ids1, b_mask1,
             b_ids2, b_mask2,
             b_labels, b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'],
                          batch['token_ids_2'], batch['attention_mask_2'],
                          batch['labels'], batch['sent_ids'])

            b_ids1 = b_ids1.to(device)
            b_mask1 = b_mask1.to(device)
            b_ids2 = b_ids2.to(device)
            b_mask2 = b_mask2.to(device)
            b_labels = b_labels.to(device)
            
            if one_embed:
                b_ids = torch.cat([b_ids1,b_ids2],dim=1)
                b_mask = torch.cat([b_mask1,b_mask2],dim=1)
                #logits are unnormalized probabilities, the more negative a logit is the lower the similarity prediction of the model is
                logits = model(b_ids, b_mask, task_id = 2,add_layers=add_layers)
    
            else:
                similarity = model(b_ids1, b_mask1, b_ids2, b_mask2, task_id=2, add_layers=add_layers)
            
            #dev loss
            if one_embed:
                # since similarity is between (0,5) the labels have to be normalized for the cross entropy such that
                # similarity is in (0,5)*0.2=(0,1)
                loss = F.binary_cross_entropy_with_logits(logits, b_labels*0.2)
            else:
                loss = F.mse_loss(similarity, b_labels.float(), reduction='mean')
                
            sts_loss += loss.item()
            num_batches+=1
            
            if one_embed:
                y_hat = logits.sigmoid().flatten().cpu().numpy()*5 #unnormalize predictions after passing them through sigmoid function
            else:
                y_hat = similarity.flatten().cpu().numpy()
            
            b_labels = b_labels.flatten().cpu().numpy()

            sts_y_pred.extend(y_hat)
            sts_y_true.extend(b_labels)
            sts_sent_ids.extend(b_sent_ids)
            
            if num_batches >= n_iter:
                break

            if args and num_batches >= args.num_batches_sts:
                break
            
        sts_loss = sts_loss/num_batches #normalize sts_loss
        
        pearson_mat = np.corrcoef(sts_y_pred,sts_y_true)
        sts_corr = pearson_mat[1][0]


        sst_y_true = []
        sst_y_pred = []
        sst_sent_ids = []
        sst_loss = 0
        num_batches = 0

        # Evaluate sentiment classification. (sst)
        for step, batch in enumerate(tqdm(sentiment_dataloader, desc=f'eval-sst', disable=TQDM_DISABLE)):
            b_ids, b_mask, b_labels, b_sent_ids = batch['token_ids'], batch['attention_mask'], batch['labels'], batch['sent_ids']

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            b_labels = b_labels.to(device)

            logits = logits = model(b_ids, b_mask, task_id=0, add_layers = add_layers)
            
            #Dev loss
            loss = F.cross_entropy(logits, b_labels.view(-1), reduction='mean')
            sst_loss += loss.item()
            num_batches+=1
            
            y_hat = logits.argmax(dim=-1).flatten().cpu().numpy()
            b_labels = b_labels.flatten().cpu().numpy()

            sst_y_pred.extend(y_hat)
            sst_y_true.extend(b_labels)
            sst_sent_ids.extend(b_sent_ids) 
            
            if num_batches >= n_iter:
                break 

            if args and num_batches >= args.num_batches_sst:
                break 
            
        sst_loss = sst_loss/num_batches #normalize loss

        # sentiment_accuracy = np.mean(np.array(sst_y_pred) == np.array(sst_y_true))
        sentiment_accuracy = accuracy_score(sst_y_true, sst_y_pred)
        sentiment_precision = precision_score(sst_y_true, sst_y_pred, average=None)
        sentiment_recall = recall_score(sst_y_true, sst_y_pred, average=None)
        sentiment_f1 = f1_score(sst_y_true, sst_y_pred, average=None)

        print(f'Paraphrase detection accuracy: {paraphrase_accuracy:.3f}')
        print(f'Sentiment classification accuracy: {sentiment_accuracy:.3f}')
        print(f'Semantic Textual Similarity correlation: {sts_corr:.3f}')

        return (para_loss,paraphrase_accuracy, para_y_pred, para_sent_ids, paraphrase_precision, paraphrase_recall, paraphrase_f1, 
                sst_loss,sentiment_accuracy,sst_y_pred, sst_sent_ids, sentiment_precision, sentiment_recall, sentiment_f1,
                sts_loss,sts_corr, sts_y_pred, sts_sent_ids)

def test_model_smart(args, model, device):
        sst_test_data, num_labels,para_test_data, sts_test_data = \
            load_multitask_data(args.sst_test,args.para_test, args.sts_test, split='test')

        sst_dev_data, num_labels,para_dev_data, sts_dev_data = \
            load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev,split='dev')

        sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_test_dataloader = DataLoader(sst_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sst_test_data.collate_fn)
        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sst_dev_data.collate_fn)

        para_test_data = SentencePairTestDataset(para_test_data, args)
        para_dev_data = SentencePairDataset(para_dev_data, args) 

        para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                          collate_fn=para_test_data.collate_fn)
        para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                         collate_fn=para_dev_data.collate_fn)

        sts_test_data = SentencePairTestDataset(sts_test_data, args)
        sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

        sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sts_test_data.collate_fn)
        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)

        _,dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids,_,_,_, \
            _,dev_sentiment_accuracy,dev_sst_y_pred, dev_sst_sent_ids,_,_,_,_,dev_sts_corr, \
            dev_sts_y_pred, dev_sts_sent_ids = smart_eval(sst_dev_dataloader,
                                                                    para_dev_dataloader,
                                                                    sts_dev_dataloader, model, device, 
                                                                    n_iter=9999999,one_embed=args.one_embed,add_layers=args.add_layers)

        test_para_y_pred, test_para_sent_ids, test_sst_y_pred, \
            test_sst_sent_ids, test_sts_y_pred, test_sts_sent_ids = \
                model_eval_test_smart(sst_test_dataloader,
                                          para_test_dataloader,
                                          sts_test_dataloader, model, device,
                                          n_iter=9999999,one_embed=args.one_embed,add_layers=args.add_layers)

        with open(args.sst_dev_out, "w+") as f:
            print(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sst_test_out, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_dev_out, "w+") as f:
            print(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}")
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_test_out, "w+") as f:
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(test_para_sent_ids, test_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_dev_out, "w+") as f:
            print(f"dev sts corr :: {dev_sts_corr :.3f}")
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_test_out, "w+") as f:
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
                f.write(f"{p} , {s} \n")
    
# Perform model evaluation in terms by averaging accuracies across tasks.
def model_eval_test_smart(sentiment_dataloader,
                         paraphrase_dataloader,
                         sts_dataloader,
                         model, device, n_iter,one_embed=False,add_layers=False):
    model.eval()  # switch to eval model, will turn off randomness like dropout

    with torch.no_grad():

        para_y_pred = []
        para_sent_ids = []
        # Evaluate paraphrase detection.
        for step, batch in enumerate(tqdm(paraphrase_dataloader, desc=f'eval', disable=TQDM_DISABLE)):
            (b_ids1, b_mask1,
             b_ids2, b_mask2,
             b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'],
                          batch['token_ids_2'], batch['attention_mask_2'],
                          batch['sent_ids'])

            b_ids1 = b_ids1.to(device)
            b_mask1 = b_mask1.to(device)
            b_ids2 = b_ids2.to(device)
            b_mask2 = b_mask2.to(device)
            
            if one_embed:
                b_ids = torch.cat([b_ids1,b_ids2],dim=1)
                b_mask = torch.cat([b_mask1,b_mask2],dim=1)
                logits = model(b_ids, b_mask, task_id = 1,add_layers=add_layers)
            else:
                logits = model(b_ids1, b_mask1, b_ids2, b_mask2, task_id=1,add_layers=add_layers)

            y_hat = logits.sigmoid().round().flatten().cpu().numpy()

            para_y_pred.extend(y_hat)
            para_sent_ids.extend(b_sent_ids)


        sts_y_pred = []
        sts_sent_ids = []


        # Evaluate semantic textual similarity.
        for step, batch in enumerate(tqdm(sts_dataloader, desc=f'eval', disable=TQDM_DISABLE)):
            (b_ids1, b_mask1,
             b_ids2, b_mask2,
             b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'],
                          batch['token_ids_2'], batch['attention_mask_2'],
                          batch['sent_ids'])

            b_ids1 = b_ids1.to(device)
            b_mask1 = b_mask1.to(device)
            b_ids2 = b_ids2.to(device)
            b_mask2 = b_mask2.to(device)

            if one_embed:
                b_ids = torch.cat([b_ids1,b_ids2],dim=1)
                b_mask = torch.cat([b_mask1,b_mask2],dim=1)
                #logits are unnormalized probabilities, the more negative a logit is the lower the similarity prediction of the model is
                logits = model(b_ids, b_mask, task_id = 2,add_layers=add_layers)
    
            else:
                similarity = model(b_ids1, b_mask1, b_ids2, b_mask2, task_id=2, add_layers=add_layers)
                
            if one_embed:
                y_hat = logits.sigmoid().flatten().cpu().numpy()*5 #unnormalize predictions after passing them through sigmoid function
            else:
                y_hat = similarity.flatten().cpu().numpy()

            sts_y_pred.extend(y_hat)
            sts_sent_ids.extend(b_sent_ids)


        sst_y_pred = []
        sst_sent_ids = []

        # Evaluate sentiment classification.
        for step, batch in enumerate(tqdm(sentiment_dataloader, desc=f'eval', disable=TQDM_DISABLE)):
            b_ids, b_mask, b_sent_ids = batch['token_ids'], batch['attention_mask'],  batch['sent_ids']

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            
            logits = logits = model(b_ids, b_mask, task_id=0, add_layers = add_layers)
            y_hat = logits.argmax(dim=-1).flatten().cpu().numpy()

            sst_y_pred.extend(y_hat)
            sst_sent_ids.extend(b_sent_ids)

            

        return (para_y_pred, para_sent_ids,
                sst_y_pred, sst_sent_ids,
                sts_y_pred, sts_sent_ids)