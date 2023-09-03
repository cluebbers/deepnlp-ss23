from bert import BertSelfAttentionBase, BertSelfAttention
import torch
import math


class LinearSelfAttention(BertSelfAttention):
	'''
	The idea is that we allow linear combinations of
	Q and K with weights that we can learn. It generalizes
	the bert self attention since for specific values of
	the weights we get the original formula.
	'''
	def __init__(self, config):
		super().__init__(config)
		self.linear_beta  = torch.nn.Linear(2, 1)
		self.linear_alpha = torch.nn.Linear(2, 1)
	
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
		QK = torch.stack([query, key], dim = -1)
		WA = self.linear_alpha(QK).squeeze()
		WB = self.linear_beta(QK).squeeze()
		return super().attention(WA, WB, value, attention_mask)


class SparsemaxSelfAttention(BertSelfAttention):
	'''
	The idea is to replace the softmax with sparsemax. The softmax
	produces a lot of small data, but is never zero. Maybe this
	additional data decreases the performance. Sparsemax has almost
	the same properties as softmax, but small values are set to zero.
	'''
	def attention(self, key: torch.Tensor, query: torch.Tensor, value: torch.Tensor, attention_mask: torch.Tensor):
		'''
		key: [bs, num_attention_heads, seq_len, attention_head_size]
		query: [bs, num_attention_heads, seq_len, attention_head_size]
		value: [bs, num_attention_heads, seq_len, attention_head_size]
		attention_mask: [bs, 1, 1, seq_len]
		score: [bs, self.num_attention_heads, seq_len, seq_len]
		output: [bs, seq_len, hidden_size]
		'''
		score = torch.matmul(query, key.transpose(2, 3))
		score = torch.div(score, math.sqrt(self.attention_head_size))
		score = torch.add(score, attention_mask)

		# The only change to BertSelfAttention is this line!
		score = Sparsemax(dim = 3)(score)

		score = self.dropout(score)
		attention = torch.matmul(score, value)
		attention = attention.transpose(1, 2)
		attention = attention.reshape(attention.shape[0], attention.shape[1], self.all_head_size)
		return attention


class CenterMatrixSelfAttention(BertSelfAttention):
	'''
	The idea is to have one matrix in between: Q W K^T (from lecture).
	'''
	def __init__(self, config):
		super().__init__(config)
		self.linear_center = torch.nn.Linear(self.attention_head_size, self.attention_head_size)
	
	def attention(self, key: torch.Tensor, query: torch.Tensor, value: torch.Tensor, attention_mask: torch.Tensor):
		'''
		key: [bs, num_attention_heads, seq_len, attention_head_size]
		query: [bs, num_attention_heads, seq_len, attention_head_size]
		value: [bs, num_attention_heads, seq_len, attention_head_size]
		attention_mask: [bs, 1, 1, seq_len]
		score: [bs, self.num_attention_heads, seq_len, seq_len]
		output: [bs, seq_len, hidden_size]
		'''
		key = self.linear_center(key)
		return super().attention(key, query, value, attention_mask)

################################################################################
# Combinations of the above ideas to see how well they work together.
################################################################################

class LinearSelfAttentionWithSparsemax(LinearSelfAttention, SparsemaxSelfAttention):
	pass

class CenterMatrixSelfAttentionWithSparsemax(CenterMatrixSelfAttention, SparsemaxSelfAttention):
	pass

class CenterMatrixLinearSelfAttention(LinearSelfAttention, CenterMatrixSelfAttention):
	pass

class CenterMatrixLinearSelfAttentionWithSparsemax(CenterMatrixLinearSelfAttention, SparsemaxSelfAttention):
	pass


################################################################################
# All custom attention choices as strings for the parameters.
################################################################################
CUSTOM_ATTENTION_CHOICES = [
	"BertSelfAttention",
	"LinearSelfAttention",
	"SparsemaxSelfAttention",
	"CenterMatrixSelfAttention",
	"LinearSelfAttentionWithSparsemax",
	"CenterMatrixSelfAttentionWithSparsemax",
	"CenterMatrixLinearSelfAttention",
	"CenterMatrixLinearSelfAttentionWithSparsemax"
]
