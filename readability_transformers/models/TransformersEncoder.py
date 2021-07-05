# Copyright 2021 One Theta. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import torch
from torch import nn
import torch.nn.functional as F

class TransformersEncoder(nn.Module):
    # A transformers head to go from embeddings -> attention embeddings
    # i.e. input is some sort of embedding already, like a concat of two embeddings.
    def __init__(
        self, 
        max_seq_len: int,
        embedding_size: int, 
        encoder_layers: int, 
        heads: int, 
        dropout: int,
        device: str
    ):
        super().__init__()
        self.encoder_layers = encoder_layers
        #self.posit_encoder = PositionalEncoder(embedding_size, max_seq_len, device)
        
        
        for layer_count in range(encoder_layers):
            norm_name = f"norm_{layer_count + 1}"
            encoder_layer_name = f"encoder_layer_{layer_count+1}"
            activation_name = f"relu_{layer_count + 1}"
            setattr(self, norm_name, nn.LayerNorm(embedding_size))
            setattr(self, encoder_layer_name, EncoderLayer(embedding_size, heads, dropout))
            setattr(self, activation_name, nn.ReLU())


    def forward(self, input_embedding):
        x = input_embedding
        #x = self.posit_encoder(x)

        for layer_count in range(self.encoder_layers):
            norm_name = f"norm_{layer_count + 1}"
            encoder_layer_name = f"encoder_layer_{layer_count+1}"
            activation_name = f"relu_{layer_count + 1}"
            norm = getattr(self, norm_name)
            layer = getattr(self, encoder_layer_name)
            activation = getattr(self, activation_name)

            x = norm(x)
            x = layer(x)
            x = activation(x)
        return x
        

class EncoderLayer(nn.Module):
    def __init__(self, embedding_size: int, heads: int, dropout=0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(embedding_size)
        self.norm_2 = nn.LayerNorm(embedding_size)
        self.attention_layer = MultiHeadAttention(heads, embedding_size)
        self.fc_layer = FeedForward(embedding_size)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

        self.activation_1 = nn.ReLU()
        self.activation_2 = nn.ReLU()
    
    def forward(self, x):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attention_layer(x2,x2,x2))
        x = self.activation_1(x)

        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.fc_layer(x2))
        x= self.activation_2(x)
        return x


# class Norm(nn.Module):
#     def __init__(self, d_model, eps = 1e-6):
#         super().__init__()
    
#         self.size = d_model
#         # create two learnable parameters to calibrate normalisation
#         self.alpha = nn.Parameter(torch.ones(self.size))
#         self.bias = nn.Parameter(torch.zeros(self.size))
#         self.eps = eps
#     def forward(self, x):
#         norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps + self.bias)
#         return norm


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__() 
        # We set d_ff as a default to 2048
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_ff)
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.activation_1 = nn.ReLU()
        self.activation_2 = nn.ReLU()
    def forward(self, x):
        x = self.layer_norm_1(x)
        x = self.dropout(self.activation_1(self.linear_1(x)))
        x = self.layer_norm_2(x)
        x = self.linear_2(x)
        x = self.activation_2(x)
        return x



class MultiHeadAttention(nn.Module):
    def __init__(self, heads: int, embedding_size: int, dropout: int = 0.1):
        super().__init__()
        self.embedding_size = embedding_size
        self.dim_per_head = embedding_size // heads
        self.heads = heads

        self.q_linear = nn.Linear(embedding_size, embedding_size)
        self.v_linear = nn.Linear(embedding_size, embedding_size)
        self.k_linear = nn.Linear(embedding_size, embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(embedding_size, embedding_size)
        self.activation = nn.ReLU()
    def forward(self, q, k, v):
        bs = q.size(0)
        
        k = self.k_linear(k).view(bs, -1, self.heads, self.dim_per_head)
        q = self.q_linear(q).view(bs, -1, self.heads, self.dim_per_head)
        v = self.v_linear(v).view(bs, -1, self.heads, self.dim_per_head)

        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        scores = self.attention(q, k, v, self.dim_per_head, self.dropout)
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.embedding_size)
        
        output = self.out(concat)
        output = self.activation(output)
    
        return output

    def attention(self, q, k, v, d_k, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
        
        scores = F.softmax(scores, dim=-1)
        
        if dropout is not None:
            scores = dropout(scores)
            
        output = torch.matmul(scores, v)
        return output


class PositionalEncoder(nn.Module):
    def __init__(self, embedding_size: int, max_seq_len: int, device: str):
        super(PositionalEncoder, self).__init__()
        self.embedding_size = embedding_size
        self.max_seq_len = max_seq_len
        self.device = device

        posit_matrix = torch.zeros(max_seq_len, embedding_size)
        for posit in range(max_seq_len):
            for i in range(0, embedding_size, 2):
                posit_matrix[posit, i] = math.sin(posit / (10000 ** ((2 * i)/embedding_size)))
                posit_matrix[posit, i + 1] = math.cos(posit / (10000 ** ((2 * (i + 1))/embedding_size)))

        posit_matrix = posit_matrix.unsqueeze(0)
        self.posit_matrix = posit_matrix

    def forward(self, x):
        x = x * math.sqrt(self.embedding_size)
        seq_len = x.size(1)
        x = x + torch.tensor(self.posit_matrix[:, :seq_len], requires_grad=False).to(self.device)
        return x