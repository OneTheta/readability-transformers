import torch
from torch import nn

from typing import List, Union

from .Prediction import ClassificationHead, RegressionHead, Prediction

class ResFCClassification(Prediction):
    def __init__(self, input_size: int, n_layers: int, h_size: int, dropout: int, n_labels: List, double: bool):
        super().__init__(input_size, double)

        self.input_size = input_size
        self.n_layers = n_layers
        self.h_size = h_size,
        self.dropout = dropout
        self.n_labels = n_labels
        self.double = double

        if self.double:
            torch.set_default_dtype(torch.float64)
        else:
            torch.set_default_dtype(torch.float32)

        self.resfc_linear = ResFCLinear(input_size, n_layers, h_size, dropout, double)
        self.classification_head = ClassificationHead(h_size, dropout, n_labels, double)

    def forward(self, x):
        assert x.size(1) == self.input_size
        
        if self.double:
            x = x.type(torch.float64)
        else:
            x = x.type(torch.float32)
        
        hidden_outputs = self.resfc_linear(x)
        classification_outputs = self.classification_head(hidden_outputs)
        return classification_outputs

    def config(self):
        return {
            "input_size": self.input_size,
            "n_layers": self.n_layers,
            "h_size": self.h_size,
            "dropout": self.dropout,
            "n_labels": self.n_labels,
            "double": self.double
        }
        


class ResFCRegression(Prediction):
    def __init__(self, input_size: int, n_layers: int, h_size: int, dropout: int, double: bool):
        """PREDICTION_MODEL_1: Fully-Connected NN. 
        This is MODEL_1 of the list of possible tail-end models for the ReadabilityTransformer,
        where it predicts the Readability score given the features extracted and the embedding from the SentenceTransformer.

        Args:
            input_size (int): # of values per one row of data.
            n_layers (int): number of layers for the fully connected NN.
            h_size (int): Size of hidden layer
        """
        super().__init__(input_size, double)
        self.input_size = input_size
        self.n_layers = n_layers
        self.h_size = h_size,
        self.dropout = dropout
        self.double = double

        if self.double:
            torch.set_default_dtype(torch.float64)
        else:
            torch.set_default_dtype(torch.float32)

        self.resfc_linear = nn.Sequential(
            nn.Linear(input_size, h_size),
            nn.BatchNorm1d(h_size),
            nn.ReLU(),
            *[ResFCHiddenLinear(
                hidden_dim=h_size,
                dropout=dropout,
                double=double
            ) for n in range(n_layers - 1)]
        )
        self.regression_head = RegressionHead(h_size, dropout, double)

    def forward(self, x):
        assert x.size(1) == self.input_size
        
        if self.double:
            x = x.type(torch.float64)
        else:
            x = x.type(torch.float32)
        
        hidden_outputs = self.resfc_linear(x)
        regression_outputs = self.regression_head(hidden_outputs)
        regression_outputs = torch.squeeze(regression_outputs, dim=-1)
        return regression_outputs

    def config(self):
        return {
            "input_size": self.input_size,
            "n_layers": self.n_layers,
            "h_size": self.h_size,
            "dropout": self.dropout,
            "double": self.double
        }


class ResFCHiddenLinear(nn.Module):
    def __init__(self, hidden_dim: int, layernorm_eps:float=1e-12, dropout: float=0.1, double: bool=False):
        super(ResFCHiddenLinear, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.double = double

        if self.double:
            torch.set_default_dtype(torch.float64)
        else:
            torch.set_default_dtype(torch.float32)

        self.dense = nn.Linear(hidden_dim, hidden_dim)
        self.layernorm = nn.LayerNorm(hidden_dim, eps=layernorm_eps)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, hidden_states):
        hidden_states_out = self.dense(hidden_states)
        hidden_states_out = self.dropout(hidden_states_out)
        hidden_states_out = self.activation(hidden_states_out)
        hidden_states_out = self.layernorm(hidden_states + hidden_states_out)
        return hidden_states_out
        