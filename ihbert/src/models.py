# -*- coding: utf-8 -*-
"""
Created on Fri 29 15:46:32 2022

inspired by the following:
https://github.com/codertimo/BERT-pytorch
https://github.com/coaxsoft/pytorch_bert

@author: tadahaya
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NewGELUActivation(nn.Module):
    """
    Google BERTで用いられているGELUを借用
    https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py
    https://arxiv.org/abs/1606.08415
    
    """
    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.hidden_size = hidden_size


    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            input tensor.
            (batch_size, sentence_size)

        """
        return self.embed(x)


class PositionalEmbedding(nn.Module):
    def __init__(self, hidden_size, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, hidden_size).float()
        pe.require_grad = False
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        d = (torch.arange(0, hidden_size, 2).float() * -(math.log(10000.0) / hidden_size)).exp()
        pe[:, 0::2] = torch.sin(pos * d)
        pe[:, 1::2] = torch.cos(pos * d)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    
    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            input tensor.
            (batch_size, sentence_size)

        """
        return self.pe[:, :x.size(1)]


class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_len=512, dropout=0.1):
        super().__init__()
        self.token = TokenEmbedding(vocab_size, hidden_size)
        self.position = PositionalEmbedding(hidden_size, max_len)
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            input tensor.
            (batch_size, sentence_size)

        """
        x = self.token(x) + self.position(x)
        return self.dropout(x)


class FastMultiHeadAttention(nn.Module):
    def __init__(
            self, hidden_size, num_attention_heads, qkv_bias,
            attn_dropout, output_dropout
            ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size # ~ hidden_size
        self.qkv_bias = qkv_bias
        self.qkv_projection = nn.Linear(hidden_size, 3 * self.all_head_size, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.output_projection = nn.Linear(self.all_head_size, hidden_size)
        # attention output -> hidden_size
        self.output_dropout = nn.Dropout(output_dropout)
    

    def forward(self, x, attention_mask:torch.Tensor=None, output_attentions:bool=False):
        """
        Parameters
        ----------
        x : torch.Tensor
            input tensor.
            (batch_size, sentence_size, hidden_size)
        
        attention_mask : torch.Tensor, optional
            attention mask tensor
            (batch_size, sentence_size, sentence_size)
            The default is None.
        
        output_attentions : bool, optional
            whether to return attention probabilities.
            The default is False.

        """
        # Q, K, Vをまとめて計算
        # (batch_size, sentence_size, hidden_size) -> (batch_size, sentence_size, 3 * all_head_size)
        qkv = self.qkv_projection(x)
        # Q, K, Vに分割
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # Q, K, Vの次元を変更
        # (batch_size, sentence_size, 3 * all_head_size)
        # -> (batch_size, sentence_size, num_attention_heads, attention_head_size)
        batch_size, seq_length, _ = q.size()
        q = q.view(
            batch_size, seq_length, self.num_attention_heads, self.attention_head_size
            ).transpose(1, 2)
        k = k.view(
            batch_size, seq_length, self.num_attention_heads, self.attention_head_size
            ).transpose(1, 2)
        v = v.view(
            batch_size, seq_length, self.num_attention_heads, self.attention_head_size
            ).transpose(1, 2)
        # attentionを計算
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / (self.attention_head_size ** 0.5)
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)
        # attentionを掛ける
        attention_output = torch.matmul(attention_probs, v)
        # attentionをreshape
        # (batch_size, num_attention_heads, sentence_size, attention_head_size)
        # -> (batch_size, sentence_size, all_head_size)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_length, self.all_head_size
            )
        # attention_outputをhidden_sizeに変換
        output = self.output_projection(attention_output)
        output = self.output_dropout(output)
        if output_attentions:
            return (output, attention_probs)
        return (output, None)


class FeedForward(nn.Module):
    # position-wise feed-forward networks
    def __init__(self, hidden_size, ffn_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, ffn_dim)
        self.activation = NewGELUActivation()
        self.fc2 = nn.Linear(ffn_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        """
        x : torch.Tensor
            input tensor.
            (batch_size, sentence_size, hidden_size)

        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """ a single transformer block """
    def __init__(
            self, hidden_size, num_attention_heads, qkv_bias,
            attn_dropout, output_dropout, ffn_dim, block_dropout
            ):
        super().__init__()
        self.attention = FastMultiHeadAttention(
            hidden_size, num_attention_heads, qkv_bias, attn_dropout, output_dropout
            )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = FeedForward(hidden_size, ffn_dim, block_dropout)
        self.norm2 = nn.LayerNorm(hidden_size)

    
    def forward(self, x, attention_mask:torch.Tensor=None, output_attentions:bool=False):
        """
        Parameters
        ----------
        x : torch.Tensor
            input tensor.
            (batch_size, sentence_size, hidden_size)
        
        """
        # self-attention
        attention_output, attention_probs = self.attention(
            x, attention_mask, output_attentions
            )
        # skip connection
        x = x + attention_output
        # feed-forward networks with skip connection
        x = x + self.ffn(self.norm2(x))
        if output_attentions:
            return (x, attention_probs)
        return (x, None)


class Encoder(nn.Module):
    def __init__(
            self, hidden_size, num_attention_heads, qkv_bias,
            attn_dropout, output_dropout, ff_dim, block_dropout, num_blocks
            ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            block = Block(
                hidden_size, num_attention_heads, qkv_bias,
                attn_dropout, output_dropout, ff_dim, block_dropout
                )
            self.blocks.append(block)


    def forward(self, x, attention_mask:torch.Tensor=None, output_attentions:bool=False):
        """
        Parameters
        ----------
        x : torch.Tensor
            input tensor.
            (batch_size, sentence_size, hidden_size)
        
        """
        attentions = []
        for block in self.blocks:
            x, attention = block(x, attention_mask, output_attentions)
            if attention is not None:
                attentions.append(attention)
        if output_attentions:
            # block数分のattentionを返す
            # (batch_size, num_attention_heads, sentence_size, sentence_size)
            return (x, attentions)
        return (x, None)


class BERT(nn.Module):
    def __init__(
            self, vocab_size, hidden_size, max_len, num_attention_heads, qkv_bias,
            attn_dropout, output_dropout, ff_dim, block_dropout, num_blocks
            ):
        super().__init__()
        self.embedding = BERTEmbedding(vocab_size, hidden_size, max_len)
        self.encoder = Encoder(
            hidden_size, num_attention_heads, qkv_bias,
            attn_dropout, output_dropout, ff_dim, block_dropout, num_blocks
            )
        self.pooler = nn.Linear(hidden_size, 1)
        self.pooler_activation = nn.Tanh()


    def forward(self, x, attention_mask:torch.Tensor=None, output_attentions:bool=False):
        """
        Parameters
        ----------
        x : torch.Tensor
            input tensor.
            (batch_size, sentence_size)
        
        attention_mask : torch.Tensor, optional
            attention mask tensor.
            (batch_size, sentence_size, sentence_size)
            The default is None.
        
        output_attentions : bool, optional
            whether to return attention probabilities.
            The default is False.
        """
        # padded token用にattention maskを作成
        if attention_mask is None:
            attention_mask = (x != 0).unsqueeze(1).expand(-1, x.size(1), -1).unsqueeze(1)
        # embedding
        x = self.embedding(x)
        # attention
        x, attentions = self.encoder(x, attention_mask, output_attentions)
        if output_attentions:
            return (x, attentions)
        return (x, None)
    

class MaskedLanguageModel(nn.Module):
    def __init__(self, bert, vocab_size):
        super(self).__init__()
        self.bert = bert
        self.decoder = nn.Linear(bert.embedding.hidden_size, vocab_size, bias=False)
        # biasはFalseにしておく
        # https://huggingface.co/transformers/model_doc/bert.html#transformers.BertForMaskedLM
        self.decoder.weight = self.bert.embedding.embed.weight
        # weightを共有する
        self.softmax = nn.LogSoftmax(dim=-1)
        

    def forward(self, x, attention_mask:torch.Tensor=None, output_attentions:bool=False):
        x, attentions = self.bert(x, attention_mask, output_attentions)
        x = self.decoder(x)
        x = self.softmax(x)
        if output_attentions:
            return (x, attentions)
        return (x, None)
    

# ToDo:
# mask部分の導入など, input tensorの形状についてもう少し考える必要あり
# trainerにヒントがあるはず
# Datasetの作成も必要, BERT用の形式が必要