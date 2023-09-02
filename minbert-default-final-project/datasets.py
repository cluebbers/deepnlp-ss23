#!/usr/bin/env python3

'''
This module contains our Dataset classes and functions to load the 3 datasets we're using.

You should only need to call load_multitask_data to get the training and dev examples
to train your model.
'''


import csv

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tokenizer import BertTokenizer
from tqdm import tqdm


def preprocess_string(s):
    return ' '.join(s.lower()
                    .replace('.', ' .')
                    .replace('?', ' ?')
                    .replace(',', ' ,')
                    .replace('\'', ' \'')
                    .split())


class SentenceClassificationDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=args.local_files_only)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):

        sents = [x[0] for x in data]
        labels = [x[1] for x in data]
        sent_ids = [x[2] for x in data]

        encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])
        labels = torch.LongTensor(labels)

        return token_ids, attention_mask, labels, sents, sent_ids

    def collate_fn(self, all_data):
        token_ids, attention_mask, labels, sents, sent_ids= self.pad_data(all_data)

        batched_data = {
                'token_ids': token_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'sents': sents,
                'sent_ids': sent_ids
            }

        return batched_data


class SentenceClassificationTestDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=args.local_files_only)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sents = [x[0] for x in data]
        sent_ids = [x[1] for x in data]

        encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])

        return token_ids, attention_mask, sents, sent_ids

    def collate_fn(self, all_data):
        token_ids, attention_mask, sents, sent_ids= self.pad_data(all_data)

        batched_data = {
                'token_ids': token_ids,
                'attention_mask': attention_mask,
                'sents': sents,
                'sent_ids': sent_ids
            }

        return batched_data


class SentencePairDataset(Dataset):
    def __init__(self, dataset, args, isRegression =False):
        self.dataset = dataset
        self.p = args
        self.isRegression = isRegression
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=args.local_files_only)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sent1 = [x[0] for x in data]
        sent2 = [x[1] for x in data]
        labels = [x[2] for x in data]
        sent_ids = [x[3] for x in data]

        encoding1 = self.tokenizer(sent1, return_tensors='pt', padding=True, truncation=True)
        encoding2 = self.tokenizer(sent2, return_tensors='pt', padding=True, truncation=True)

        token_ids = torch.LongTensor(encoding1['input_ids'])
        attention_mask = torch.LongTensor(encoding1['attention_mask'])
        token_type_ids = torch.LongTensor(encoding1['token_type_ids'])

        token_ids2 = torch.LongTensor(encoding2['input_ids'])
        attention_mask2 = torch.LongTensor(encoding2['attention_mask'])
        token_type_ids2 = torch.LongTensor(encoding2['token_type_ids'])
        if self.isRegression:
            labels = torch.DoubleTensor(labels)
        else:
            labels = torch.LongTensor(labels)
            

        return (token_ids, token_type_ids, attention_mask,
                token_ids2, token_type_ids2, attention_mask2,
                labels,sent_ids)

    def collate_fn(self, all_data):
        (token_ids, token_type_ids, attention_mask,
         token_ids2, token_type_ids2, attention_mask2,
         labels, sent_ids) = self.pad_data(all_data)

        batched_data = {
                'token_ids_1': token_ids,
                'token_type_ids_1': token_type_ids,
                'attention_mask_1': attention_mask,
                'token_ids_2': token_ids2,
                'token_type_ids_2': token_type_ids2,
                'attention_mask_2': attention_mask2,
                'labels': labels,
                'sent_ids': sent_ids
            }

        return batched_data


class SentencePairTestDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=args.local_files_only)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sent1 = [x[0] for x in data]
        sent2 = [x[1] for x in data]
        sent_ids = [x[2] for x in data]

        encoding1 = self.tokenizer(sent1, return_tensors='pt', padding=True, truncation=True)
        encoding2 = self.tokenizer(sent2, return_tensors='pt', padding=True, truncation=True)

        token_ids = torch.LongTensor(encoding1['input_ids'])
        attention_mask = torch.LongTensor(encoding1['attention_mask'])
        token_type_ids = torch.LongTensor(encoding1['token_type_ids'])

        token_ids2 = torch.LongTensor(encoding2['input_ids'])
        attention_mask2 = torch.LongTensor(encoding2['attention_mask'])
        token_type_ids2 = torch.LongTensor(encoding2['token_type_ids'])


        return (token_ids, token_type_ids, attention_mask,
                token_ids2, token_type_ids2, attention_mask2,
               sent_ids)

    def collate_fn(self, all_data):
        (token_ids, token_type_ids, attention_mask,
         token_ids2, token_type_ids2, attention_mask2,
         sent_ids) = self.pad_data(all_data)

        batched_data = {
                'token_ids_1': token_ids,
                'token_type_ids_1': token_type_ids,
                'attention_mask_1': attention_mask,
                'token_ids_2': token_ids2,
                'token_type_ids_2': token_type_ids2,
                'attention_mask_2': attention_mask2,
                'sent_ids': sent_ids
            }

        return batched_data


def load_multitask_test_data():
    paraphrase_filename = f'data/quora-test.csv'
    sentiment_filename = f'data/ids-sst-test.txt'
    similarity_filename = f'data/sts-test.csv'

    sentiment_data = []

    with open(sentiment_filename, 'r', encoding='utf-8') as fp:
        for record in csv.DictReader(fp,delimiter = '\t'):
            sent = record['sentence'].lower().strip()
            sentiment_data.append(sent)

    print(f"Loaded {len(sentiment_data)} test examples from {sentiment_filename}")

    paraphrase_data = []
    with open(paraphrase_filename, 'r', encoding='utf-8') as fp:
        for record in csv.DictReader(fp,delimiter = '\t'):
            #if record['split'] != split:
            #    continue
            paraphrase_data.append((preprocess_string(record['sentence1']),
                                    preprocess_string(record['sentence2']),
                                    ))

    print(f"Loaded {len(paraphrase_data)} test examples from {paraphrase_filename}")

    similarity_data = []
    with open(similarity_filename, 'r', encoding='utf-8') as fp:
        for record in csv.DictReader(fp,delimiter = '\t'):
            similarity_data.append((preprocess_string(record['sentence1']),
                                    preprocess_string(record['sentence2']),
                                    ))

    print(f"Loaded {len(similarity_data)} test examples from {similarity_filename}")

    return sentiment_data, paraphrase_data, similarity_data



def load_multitask_data(sentiment_filename,paraphrase_filename,similarity_filename,split='train'):
    sentiment_data = []
    num_labels = {}
    if split == 'test':
        with open(sentiment_filename, 'r', encoding='utf-8') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent = record['sentence'].lower().strip()
                sent_id = record['id'].lower().strip()
                sentiment_data.append((sent,sent_id))
    else:
        with open(sentiment_filename, 'r', encoding='utf-8') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent = record['sentence'].lower().strip()
                sent_id = record['id'].lower().strip()
                label = int(record['sentiment'].strip())
                if label not in num_labels:
                    num_labels[label] = len(num_labels)
                sentiment_data.append((sent, label,sent_id))

    print(f"Loaded {len(sentiment_data)} {split} examples from {sentiment_filename}")

    paraphrase_data = []
    if split == 'test':
        with open(paraphrase_filename, 'r', encoding='utf-8') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent_id = record['id'].lower().strip()
                paraphrase_data.append((preprocess_string(record['sentence1']),
                                        preprocess_string(record['sentence2']),
                                        sent_id))

    else:
        with open(paraphrase_filename, 'r', encoding='utf-8') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                try:
                    sent_id = record['id'].lower().strip()
                    paraphrase_data.append((preprocess_string(record['sentence1']),
                                            preprocess_string(record['sentence2']),
                                            int(float(record['is_duplicate'])),sent_id))
                except:
                    pass

    print(f"Loaded {len(paraphrase_data)} {split} examples from {paraphrase_filename}")

    similarity_data = []
    if split == 'test':
        with open(similarity_filename, 'r', encoding='utf-8') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent_id = record['id'].lower().strip()
                similarity_data.append((preprocess_string(record['sentence1']),
                                        preprocess_string(record['sentence2'])
                                        ,sent_id))
    else:
        with open(similarity_filename, 'r', encoding='utf-8') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent_id = record['id'].lower().strip()
                similarity_data.append((preprocess_string(record['sentence1']),
                                        preprocess_string(record['sentence2']),
                                        float(record['similarity']),sent_id))

    print(f"Loaded {len(similarity_data)} {split} examples from {similarity_filename}")

    return sentiment_data, num_labels, paraphrase_data, similarity_data


class MultitaskDataloader:
    def __init__(self, args, device, enable_test: bool = False):
        
        common_dataloader_params = dict(
            batch_size = args.batch_size,
            num_workers = 2,
            pin_memory = True
        )

        sst_train_data, num_labels, para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
        sst_dev_data, num_labels, para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')
        sst_train_data = SentenceClassificationDataset(sst_train_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)
        sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, collate_fn=sst_train_data.collate_fn,
                                        **common_dataloader_params)
        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, collate_fn=sst_dev_data.collate_fn,
                                        **common_dataloader_params)
        
        para_train_data = SentencePairDataset(para_train_data, args)
        para_dev_data = SentencePairDataset(para_dev_data, args)
        para_train_dataloader = DataLoader(para_train_data, shuffle=True, collate_fn=para_train_data.collate_fn,
                                        **common_dataloader_params)
        para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, collate_fn=para_dev_data.collate_fn,
                                        **common_dataloader_params)
        
        sts_train_data = SentencePairDataset(sts_train_data, args, isRegression = True)
        sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression = True)
        sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, collate_fn=sts_train_data.collate_fn,
                                        **common_dataloader_params)
        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, collate_fn=sts_dev_data.collate_fn,
                                        **common_dataloader_params)   
        
        self.args                  = args
        self.sst_train_dataloader  = sst_train_dataloader
        self.sst_dev_dataloader    = sst_dev_dataloader    
        self.para_train_dataloader = para_train_dataloader
        self.para_dev_dataloader   = para_dev_dataloader
        self.sts_train_dataloader  = sts_train_dataloader
        self.sts_dev_dataloader    = sts_dev_dataloader
        self.num_labels            = num_labels
        self.device                = device
        self.TQDM_DISABLE          = False
        self.profiler              = None

        self.sts_train_dataloader_size = len(sts_train_dataloader) 
        self.sst_train_dataloader_size = len(sst_train_dataloader)
        self.para_train_dataloader_size = len(para_train_dataloader)

        if args.profiler:
            self.profiler = torch.profiler.profile(
                activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule = torch.profiler.schedule(wait = 1, warmup = 1, active = 3),
                on_trace_ready = torch.profiler.tensorboard_trace_handler("runs/profiler"),
                record_shapes = True,
                profile_memory = True,
                with_stack = False
            )
        
        if enable_test:
            sst_test_data, num_labels, para_test_data, sts_test_data = load_multitask_data(args.sst_test,args.para_test, args.sts_test, split='test')
            
            sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
            sst_test_dataloader = DataLoader(sst_test_data, shuffle=True, collate_fn=sst_test_data.collate_fn,
                                             **common_dataloader_params)
            
            para_test_data = SentencePairTestDataset(para_test_data, args)
            para_test_dataloader = DataLoader(para_test_data, shuffle=True, collate_fn=para_test_data.collate_fn,
                                              **common_dataloader_params)
            
            sts_test_data = SentencePairTestDataset(sts_test_data, args)
            sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, collate_fn=sts_test_data.collate_fn,
                                             **common_dataloader_params)

            self.sst_test_dataloader = sst_test_dataloader
            self.para_test_dataloader = para_test_dataloader
            self.sts_test_dataloader = sts_test_dataloader

    
    def iter_impl(self, dataloader, tqdm_desc, test_max_batch):

        if self.profiler:
            self.profiler.start()

        for i, batch in enumerate(tqdm(dataloader, desc = tqdm_desc, disable = self.TQDM_DISABLE)):
            yield batch

            if self.profiler:
                self.profiler.step()
                if i >= (1 + 1 + 3):
                    break

            if i >= test_max_batch:
                break

        if self.profiler:
            self.profiler.stop() 

    def iter_sts(self, dataloader, tqdm_desc, evaluate):
        for batch in self.iter_impl(dataloader, tqdm_desc, self.args.num_batches_sts):
            if self.args.para_sep:
                break
            output = [
                batch['token_ids_1'].to(self.device),
                batch['attention_mask_1'].to(self.device),
                batch['token_ids_2'].to(self.device),
                batch['attention_mask_2'].to(self.device),
                batch['labels'].to(self.device)
            ]
            if evaluate:
                output.append(batch['sent_ids'])
            yield output

    def iter_sst(self, dataloader, tqdm_desc, evaluate):
        for batch in self.iter_impl(dataloader, tqdm_desc, self.args.num_batches_sst):
            if self.args.para_sep:
                break
            output = [
                batch['token_ids'].to(self.device),
                batch['attention_mask'].to(self.device),
                batch['labels'].to(self.device)
            ]
            if evaluate:
                output.append(batch['sent_ids'])
            yield output

    def iter_para(self, dataloader, tqdm_desc, evaluate):
        for batch in self.iter_impl(dataloader, tqdm_desc, self.args.num_batches_para):
            #train the last epochs only on a small fraction of the para data
            if self.args.skip_para:
                rand = np.random.uniform()
                if rand >= self.sst_train_dataloader_size / (2 * self.para_train_dataloader_size): #train on batch only with a certain probability 
                #-> the mean of trained batches is half of the number of batches as in the sst set(the smallest dataset of all three)
                    continue
            output = [
                batch['token_ids_1'].to(self.device),
                batch['attention_mask_1'].to(self.device),
                batch['token_ids_2'].to(self.device),
                batch['attention_mask_2'].to(self.device),
                batch['labels'].to(self.device)
            ]
            if evaluate:
                output.append(batch['sent_ids'])
            yield output

    def iter_train_sts(self, epoch):
        yield from self.iter_sts(self.sts_train_dataloader, f'train-sts-{epoch}', False)

    def iter_eval_sts(self, dev):
        if dev:
            yield from self.iter_sts(self.sts_dev_dataloader, f'dev-sts', True)
        else:
            yield from self.iter_sts(self.sts_train_dataloader, f'eval-sts', True)
    
    def iter_train_sst(self, epoch):
        yield from self.iter_sst(self.sst_train_dataloader, f'train-sst-{epoch}', False)

    def iter_eval_sst(self, dev):
        if dev:
            yield from self.iter_sst(self.sst_dev_dataloader, f'dev-sst', True)
        else:
            yield from self.iter_sst(self.sst_train_dataloader, f'eval-sst', True)

    def iter_train_para(self, epoch):
        yield from self.iter_para(self.para_train_dataloader, f'train-para-{epoch}', False)

    def iter_eval_para(self, dev):
        if dev:
            yield from self.iter_para(self.para_dev_dataloader, f'dev-para', True)
        else:
            yield from self.iter_para(self.para_train_dataloader, f'eval-para', True)
