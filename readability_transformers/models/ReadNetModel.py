import torch
import numpy as np
from torch import Tensor, nn, tensor
import math
import pandas as pd
import csv
from pathlib import Path


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, masked):
        super().__init__()
        assert d_model % num_heads == 0, "num_heads must evenly chunk d_model"
        self.num_heads = num_heads
        self.wq = nn.Linear(d_model, d_model, bias=False)  # QQ what if bias=True?
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.masked = masked
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        qs = self.wq(q).chunk(self.num_heads, dim=2)
        ks = self.wk(k).chunk(self.num_heads, dim=2)
        vs = self.wv(v).chunk(self.num_heads, dim=2)
        outs = []
        # TODO Use einsum instead of for loop
        for qi, ki, vi in zip(qs, ks, vs):
            attns = qi.bmm(ki.transpose(1, 2)) / (ki.shape[2] ** 0.5)
            if self.masked:
                attns = attns.tril()  # Zero out upper triangle so it can't look ahead
            attns = self.softmax(attns)
            outs.append(attns.bmm(vi))
        return torch.cat(outs, dim=2)

class AddNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x1, x2):
        return self.ln(x1+x2)

class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.l1 = nn.Linear(d_model, d_model)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(d_model, d_model)
    def forward(self, x):
        return self.l2(self.relu(self.l1(x)))

def pos_encode(x):
    pos, dim = torch.meshgrid(torch.arange(x.shape[1]), torch.arange(x.shape[2]))
    dim = 2 * (dim // 2)
    enc_base = pos/(10_000**(dim / x.shape[2]))
    addition = torch.zeros_like(x)
    for d in range(x.shape[2]):
        enc_func = torch.sin if d % 2 == 0 else torch.cos
        addition[:,:,d] = enc_func(enc_base[:,d])
    if x.is_cuda:
        addition = addition.cuda()
    return x + addition

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, masked=False)
        self.an1 = AddNorm(d_model)
        self.ff = FeedForward(d_model)
        self.an2 = AddNorm(d_model)

    def forward(self, x):
        x = self.an1(x, self.mha(q=x, k=x, v=x))
        return self.an2(x, self.ff(x))

class AttentionAggregation(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Linear(d_model, 1, bias=False)

    def forward(self, x):  # (b, s, m)
        attns = self.query(x).softmax(dim=1)  # (b, s, 1)
        enc = torch.bmm(attns.transpose(1, 2), x)  # (b, 1, m)
        return enc.squeeze(1)

class LinTanh(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.lin = nn.Linear(d_model, d_model)
        self.tanh = nn.Tanh()

    def forward(self, x):
        return self.tanh(self.lin(x))

class LinFeatConcat(nn.Module):
    def __init__(self, d_model, n_feats, n_out):
        super().__init__()
        self.n_feats = n_feats
        self.d_model = d_model
        self.lin = nn.Linear(d_model + n_feats, n_out, bias=False)  # TODO what if True?

    def forward(self, x, feats):
        concatenated = torch.cat([x, feats], dim=1)
        dense_out = self.lin(concatenated)
        return dense_out

class ReadNetBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_blocks, n_feats, n_out):
        super().__init__()
        self.blocks = nn.Sequential(*[EncoderBlock(d_model=d_model, num_heads=n_heads) for _ in range(n_blocks)])
        self.lin_tanh = LinTanh(d_model=d_model)
        self.attn_agg = AttentionAggregation(d_model=d_model)
        self.lin_feat_concat = LinFeatConcat(d_model=d_model, n_feats=n_feats, n_out=n_out)
        self.actf = nn.Tanh()

    def forward(self, x, feats):  # (b, s, m), (b, f)
        x = pos_encode(x)
        x = self.blocks(x)
        x = self.lin_tanh(x)
        x = self.attn_agg(x)
        return self.actf(self.lin_feat_concat(x, feats))

class SentenceReadNetBlock(nn.Module):
    def __init__(self, sentence_transformers, d_model, n_heads, n_blocks, n_feats, n_out):
        super().__init__()
        self.blocks = sentence_transformers
        self.lin_tanh = LinTanh(d_model=d_model)
        self.attn_agg = AttentionAggregation(d_model=d_model)
        self.lin_feat_concat = LinFeatConcat(d_model=d_model, n_feats=n_feats, n_out=n_out)

    def forward(self, x, feats):  # (b, s, m), (b, f)
        sents_enc = self.blocks(x)
        sents_enc = sents_enc["sentence_embedding"] # of "cls_token_embeddings". we can experiment with this later

        # sents_enc = self.lin_tanh(sents_enc)
        # sents_enc = self.attn_agg(sents_enc)
        # basically attn_agg * lin_tanh is all done by sentence_transformer already
  
        return self.lin_feat_concat(sents_enc, feats)


class ReadNetModel(nn.Module):
    def __init__(self, sentence_transformers, d_model, n_heads, n_blocks, n_feats_sent, n_feats_doc):
        super().__init__()
        self.sent_block = SentenceReadNetBlock(
            sentence_transformers=sentence_transformers, d_model=d_model, n_heads=n_heads, n_blocks=n_blocks, n_feats=n_feats_sent, n_out=d_model
        )
        self.d_model = d_model
        self.doc_block = ReadNetBlock(
            d_model=d_model, n_heads=n_heads, n_blocks=n_blocks, n_feats=n_feats_doc, n_out=d_model + n_feats_doc
        )
        self.head = nn.Sequential(
            nn.Linear(d_model + n_feats_doc, 1),
        )
        

    def forward(self, x, feats_sent=None, feats_doc=None):  # (b, d, s) tokens, (b, d, n_f_s), (b, n_f_d)
        if feats_sent is None: feats_sent = Tensor([])
        if feats_doc is None: feats_doc = Tensor([])
      
        n_docs, n_sents, n_tokens = x["input_ids"].size()
        reshaped_batch = dict()
        for key in x:
            reshaped_batch[key] = x[key].reshape(n_docs * n_sents, n_tokens)
        sents_enc = self.sent_block(reshaped_batch, feats_sent.reshape(n_docs * n_sents, -1))
        
        docs = sents_enc.reshape(n_docs, n_sents, self.d_model)
        docs_enc = self.doc_block(docs, feats_doc)
        out = self.head(docs_enc)
        return out.squeeze(1)