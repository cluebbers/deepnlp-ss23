from typing import Dict, List, Optional, Union, Tuple, Callable
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from base_bert import BertPreTrainedModel
from utils import *


class BertSelfAttentionBase(ABC, torch.nn.Module):
	'''
	Base class for bert self attention calculations and
	custom derivated attention formulas. The goal is that
	we can change the formula without having to change
	too much of the code.
	'''
	def __init__(self, config: dict):
		super().__init__()
		self.num_attention_heads = config.num_attention_heads
		self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
		self.all_head_size = self.num_attention_heads * self.attention_head_size
		self.query = torch.nn.Linear(config.hidden_size, self.all_head_size)
		self.key = nn.Linear(config.hidden_size, self.all_head_size)
		self.value = nn.Linear(config.hidden_size, self.all_head_size)
		self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

	def transform(self, input: torch.Tensor, linear_layer: torch.nn.Linear):
		'''
		input: [bs, seq_len, hidden_size]
		linear_layer: [bs, seq_len, hidden_size, all_head_size]
		output: [bs, num_attention_heads, seq_len, attention_head_size]
		'''
		bs, seq_len = input.shape[:2]
		proj = linear_layer(input)
		proj = proj.view(bs, seq_len, self.num_attention_heads, self.attention_head_size)
		proj = proj.transpose(1, 2)
		return proj

	def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor):
		'''
		hidden_states: [bs, seq_len, hidden_size]
		attention_mask: [bs, 1, 1, seq_len]
		output: [bs, seq_len, hidden_size]
		'''
		key = self.transform(hidden_states, self.key)
		query = self.transform(hidden_states, self.value)
		value = self.transform(hidden_states, self.query)
		attention_value = self.attention(key, query, value, attention_mask)
		return attention_value

	@abstractmethod
	def attention(self, key: torch.Tensor, query: torch.Tensor, value: torch.Tensor, attention_mask: torch.Tensor):
		'''
		This is the method that we should overwrite.
		'''
		pass


class BertSelfAttention(BertSelfAttentionBase):
	'''
	The standard implementation of bert self attention, see
	eq (1) of https://arxiv.org/pdf/1706.03762.pdf.
	'''
	def attention(self, key: Tensor, query: Tensor, value: Tensor, attention_mask: Tensor):
		'''
		key: [bs, num_attention_heads, seq_len, attention_head_size]
		query: [bs, num_attention_heads, seq_len, attention_head_size]
		value: [bs, num_attention_heads, seq_len, attention_head_size]
		attention_mask: [bs, 1, 1, seq_len]
		score: [bs, self.num_attention_heads, seq_len, seq_len]
		output: [bs, seq_len, hidden_size]

		See paper eq (1) for the formula!!
		'''
		score = torch.matmul(query, key.transpose(2, 3))
		score = torch.div(score, math.sqrt(self.attention_head_size))
		score = torch.add(score, attention_mask)
		score = F.softmax(score, dim = 3)
		score = self.dropout(score)
		attention = torch.matmul(score, value)

		# before: [bs, self.num_attention_heads, seq_len, self.attention_head_size]
		# after: [bs, seq_len, num_attention_heads * attention_head_size = hidden_size]
		attention = attention.transpose(1, 2)
		attention = attention.reshape(attention.shape[0], attention.shape[1], self.all_head_size)
		return attention


class BertLayer(nn.Module):
	def __init__(self, config, attention_module: BertSelfAttentionBase = BertSelfAttention):
		super().__init__()

		from custom_attention import CenterMatrixLinearSelfAttentionWithSparsemax
		attention_module = CenterMatrixLinearSelfAttentionWithSparsemax

		# multi-head attention
		self.self_attention = attention_module(config)
		# add-norm
		self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
		self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
		self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
		# feed forward
		self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
		self.interm_af = F.gelu
		# another add-norm
		self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
		self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
		self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

	def add_norm(self, input, output, dense_layer, dropout, ln_layer):
		"""
		this function is applied after the multi-head attention layer or the feed forward layer
		input: the input of the previous layer
		output: the output of the previous layer
		dense_layer: used to transform the output
		dropout: the dropout to be applied 
		ln_layer: the layer norm to be applied
		"""
		# Hint: Remember that BERT applies to the output of each sub-layer, before it is added to the sub-layer input and normalized 
		### TODO
		# section 5.4 in "Attention is all you need"
		# dense_layer: used to transform the output
		norm = dense_layer(output)
		
		# dropout: the dropout to be applied
		norm = dropout(norm)
		
		# ln_layer: the layer norm to be applied
		norm = ln_layer(input + norm)
		
		return norm
		# raise NotImplementedError


	def forward(self, hidden_states, attention_mask):
		"""
		hidden_states: either from the embedding layer (first bert layer) or from the previous bert layer
		as shown in the left of Figure 1 of https://arxiv.org/pdf/1706.03762.pdf 
		each block consists of 
		1. a multi-head attention layer (BertSelfAttention)
		2. a add-norm that takes the input and output of the multi-head attention layer
		3. a feed forward layer
		4. a add-norm that takes the input and output of the feed forward layer
		"""
		### TODO
		# 1. a multi-head attention layer (BertSelfAttention)
		# see first TODO implementation
		attention = self.self_attention(hidden_states, attention_mask)
		
		# 2. a add-norm that takes the input and output of the multi-head attention layer
		# see second TODO implementation
		# function inputs are specidfied above in the class after # add_norm
		norm = self.add_norm(hidden_states, attention, self.attention_dense, self.attention_dropout, self.attention_layer_norm)
		
		# 3. a feed forward layer
		# see class definition
		# input is the normalized layer
		feed = self.interm_dense(norm)
		feed = self.interm_af(feed)
		
		# 4. a add-norm that takes the input and output of the feed forward layer   
		forward = self.add_norm(norm, feed, self.out_dense, self.out_dropout, self.out_layer_norm)
		
		return forward
		# raise NotImplementedError



class BertModel(BertPreTrainedModel):
	"""
	the bert model returns the final embeddings for each token in a sentence
	it consists
	1. embedding (used in self.embed)
	2. a stack of n bert layers (used in self.encode)
	3. a linear transformation layer for [CLS] token (used in self.forward, as given)
	"""
	def __init__(self, config, attention_module = BertSelfAttention):
		super().__init__(config)
		self.config = config

		# embedding
		self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
		self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
		self.tk_type_embedding = nn.Embedding(config.type_vocab_size, config.hidden_size)
		self.embed_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
		self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)
		# position_ids (1, len position emb) is a constant, register to buffer
		position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
		self.register_buffer('position_ids', position_ids)

		# bert encoder
		self.bert_layers = nn.ModuleList([BertLayer(config, attention_module) for _ in range(config.num_hidden_layers)])

		# for [CLS] token
		self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
		self.pooler_af = nn.Tanh()

		self.init_weights()

	def embed(self, input_ids):
		input_shape = input_ids.size()
		seq_length = input_shape[1]

		# Get word embedding from self.word_embedding into input_embeds.
		inputs_embeds = None
		### TODO
		# see class definition
		inputs_embeds = self.word_embedding(input_ids)
		# raise NotImplementedError


		# Get position index and position embedding from self.pos_embedding into pos_embeds.
		pos_ids = self.position_ids[:, :seq_length]

		pos_embeds = None
		### TODO
		# see class definition
		pos_ids = self.position_ids[:, :seq_length]

		pos_embeds = self.pos_embedding(pos_ids)
		# raise NotImplementedError


		# Get token type ids, since we are not consider token type, just a placeholder.
		tk_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
		tk_type_embeds = self.tk_type_embedding(tk_type_ids)

		# Add three embeddings together; then apply embed_layer_norm and dropout and return.
		### TODO
		embeds = torch.add(inputs_embeds, torch.add(tk_type_embeds, pos_embeds))
		embeds = self.embed_layer_norm(embeds)
		embeds = self.embed_dropout(embeds)
		
		return embeds
		# raise NotImplementedError


	def encode(self, hidden_states, attention_mask):
		"""
		hidden_states: the output from the embedding layer [batch_size, seq_len, hidden_size]
		attention_mask: [batch_size, seq_len]
		"""
		# get the extended attention mask for self attention
		# returns extended_attention_mask of [batch_size, 1, 1, seq_len]
		# non-padding tokens with 0 and padding tokens with a large negative number 
		extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, self.dtype)

		# pass the hidden states through the encoder layers
		for i, layer_module in enumerate(self.bert_layers):
			# feed the encoding from the last bert_layer to the next
			hidden_states = layer_module(hidden_states, extended_attention_mask)

		return hidden_states

	def forward(self, input_ids, attention_mask):
		"""
		input_ids: [batch_size, seq_len], seq_len is the max length of the batch
		attention_mask: same size as input_ids, 1 represents non-padding tokens, 0 represents padding tokens
		"""
		# get the embedding for each input token
		embedding_output = self.embed(input_ids=input_ids)

		# feed to a transformer (a stack of BertLayers)
		sequence_output = self.encode(embedding_output, attention_mask=attention_mask)

		# get cls token hidden state
		first_tk = sequence_output[:, 0]
		first_tk = self.pooler_dense(first_tk)
		first_tk = self.pooler_af(first_tk)

		return {'last_hidden_state': sequence_output, 'pooler_output': first_tk}
