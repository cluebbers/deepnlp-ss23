from bert import BertSelfAttentionBase, BertSelfAttention
import torch


class LinearSelfAttention(BertSelfAttention):
	'''
	The idea is that we allow linear combinations of
	Q and K with weights that we can learn. It generalizes
	the bert self attention since for specific values of
	the weights we get the original formula.
	'''
	def __init__(self, config):
		super().__init__(config)
		self.linear_alpha   = torch.nn.Linear(2, 1)
		self.linear_beta    = torch.nn.Linear(2, 1)
	
	def attention(self, key: torch.Tensor, query: torch.Tensor, value: torch.Tensor, attention_mask: torch.Tensor):
		'''
		key: [bs, num_attention_heads, seq_len, attention_head_size]
		query: [bs, num_attention_heads, seq_len, attention_head_size]
		value: [bs, num_attention_heads, seq_len, attention_head_size]
		attention_mask: [bs, 1, 1, seq_len]
		score: [bs, self.num_attention_heads, seq_len, seq_len]
		output: [bs, seq_len, hidden_size]

		See issue #53 for the formula !!
		'''
		KQ = torch.stack([key, query], dim = -1)
		WA = self.linear_alpha(KQ).squeeze()
		WB = self.linear_beta(KQ).squeeze()
		return super().attention(WA, WB, value, attention_mask)
