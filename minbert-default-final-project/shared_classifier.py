import csv
import numpy
import random
import torch
from bert import BertModel


def seed_everything(seed = 11711):
	random.seed(seed)
	numpy.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


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


def load_classifier_data(filename, flag='train'):
	num_labels = {}
	data = []
	if flag == 'test':
		with open(filename, 'r') as fp:
			for record in csv.DictReader(fp,delimiter = '\t'):
				sent = record['sentence'].lower().strip()
				sent_id = record['id'].lower().strip()
				data.append((sent,sent_id))
	else:
		with open(filename, 'r') as fp:
			for record in csv.DictReader(fp,delimiter = '\t'):
				sent = record['sentence'].lower().strip()
				sent_id = record['id'].lower().strip()
				label = int(record['sentiment'].strip())
				if label not in num_labels:
					num_labels[label] = len(num_labels)
				data.append((sent, label,sent_id))
		print(f"load {len(data)} data from {filename}")
	return (data, len(num_labels)) if flag == 'train' else data


def load_bert_model(config: dict):
	kwargs = dict(
		local_files_only = config.local_files_only
	)
	bert = BertModel.from_pretrained('bert-base-uncased', **kwargs)
	bert_grads = True if config.option == 'finetine' else False
	for param in self.bert.parameters():
		param.requires_grad = bert_grads
	return bert
