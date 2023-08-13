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


def load_bert_model(config: dict):
	kwargs = dict(
		local_files_only = config.local_files_only
	)
	bert = BertModel.from_pretrained('bert-base-uncased', **kwargs)
	bert_grads = True if config.option == 'finetine' else False
	for param in self.bert.parameters():
		param.requires_grad = bert_grads
	return bert
