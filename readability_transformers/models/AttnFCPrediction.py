import torch
from torch import nn

from typing import List, Union

from .Prediction import ClassificationHead, RegressionHead, Prediction
from .ResidualFCPrediction import ResFCHiddenLinear

class AttnFCClassification(Prediction):
    def __init__(self, input_size: int, n_layers: int, h_size: int, dropout: int, n_labels: List, double: bool):
        super().__init__(input_size, double)

        self.input_size = input_size
        self.n_layers = n_layers
        self.h_size = h_size
        self.dropout = dropout
        self.n_labels = n_labels
        self.double = double

        if self.double:
            torch.set_default_dtype(torch.float64)
        else:
            torch.set_default_dtype(torch.float32)

        self.attention_head = AttentionHead(self.input_size, self.h_size)

        self.fc_linear = nn.Sequential(
            *[FCHiddenLinear(
                hidden_dim=h_size,
                dropout=dropout,
                double=double
            ) for n in range(n_layers - 1)]
        )

        self.classification_head = ClassificationHead(h_size, dropout, n_labels, double)

    def forward(self, x):
        assert x.size(1) == self.input_size
        
        if self.double:
            x = x.type(torch.float64)
        else:
            x = x.type(torch.float32)
        
        att_out = self.attention_head(x)
        hidden_outputs = self.fc_linear(att_out)
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
        


class AttnFCRegression(Prediction):
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
        self.h_size = h_size
        self.dropout = dropout
        self.double = double

        if self.double:
            torch.set_default_dtype(torch.float64)
        else:
            torch.set_default_dtype(torch.float32)

        self.attention_head = AttentionHead(self.input_size, self.h_size)

        self.fc_linear = nn.Sequential(
            *[FCHiddenLinear(
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
        
        att_out = self.attention_head(x)

        hidden_outputs = self.fc_linear(att_out)
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


class AttnFCRegressionLayerNorm(Prediction):
    def __init__(self, input_size: int, n_layers: int, h_size: int, dropout: int, double: bool, layernorm_eps:float=1e-12):
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
        self.h_size = h_size
        self.dropout = dropout
        self.double = double

        if self.double:
            torch.set_default_dtype(torch.float64)
        else:
            torch.set_default_dtype(torch.float32)

        self.attention_head = AttentionHead(self.input_size, self.h_size)

        self.fc_linear = nn.Sequential(
            *[FCHiddenLinearLayerNorm(
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
        
        att_out = self.attention_head(x)

        hidden_outputs = self.fc_linear(att_out)
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

class AttnResFCRegression(Prediction):
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
        self.h_size = h_size
        self.dropout = dropout
        self.double = double

        if self.double:
            torch.set_default_dtype(torch.float64)
        else:
            torch.set_default_dtype(torch.float32)

        self.attention_head = AttentionHead(self.input_size, self.h_size)

        self.fc_linear = nn.Sequential(
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
        
        att_out = self.attention_head(x)

        hidden_outputs = self.fc_linear(att_out)
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










class AttentionHead(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super().__init__()
        self.in_features = in_features
        self.middle_features = hidden_dim

        self.W = nn.Linear(in_features, hidden_dim)
        self.V = nn.Linear(in_features, hidden_dim)

        self.dense = nn.Linear(hidden_dim, hidden_dim)
        self.actf = nn.ReLU()
        self.out_features = hidden_dim

    def forward(self, features):
        att_score = torch.tanh(self.W(features))
        attention_weights = torch.softmax(att_score, dim = 1)
        features = self.V(features)
        context_vector = attention_weights * features
        context_vector = self.dense(context_vector)
        context_vector = self.actf(context_vector)

        return context_vector

class FCHiddenLinear(nn.Module):
    def __init__(self, hidden_dim: int, layernorm_eps:float=1e-12, dropout: float=0.1, double: bool=False):
        super(FCHiddenLinear, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.double = double

        if self.double:
            torch.set_default_dtype(torch.float64)
        else:
            torch.set_default_dtype(torch.float32)

        self.dense = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, hidden_states):
        hidden_states_out = self.dense(hidden_states)
        hidden_states_out = self.dropout(hidden_states_out)
        hidden_states_out = self.activation(hidden_states_out)
        return hidden_states_out
    

class FCHiddenLinearLayerNorm(nn.Module):
    def __init__(self, hidden_dim: int, layernorm_eps:float=1e-12, dropout: float=0.1, double: bool=False):
        super(FCHiddenLinearLayerNorm, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.double = double

        if self.double:
            torch.set_default_dtype(torch.float64)
        else:
            torch.set_default_dtype(torch.float32)

        self.dense = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(hidden_dim, eps=layernorm_eps)
        self.activation = nn.ReLU()
    
    def forward(self, hidden_states):
        hidden_states_out = self.dense(hidden_states)
        hidden_states_out = self.dropout(hidden_states_out)
        hidden_states_out = self.activation(hidden_states_out)
        hidden_states_out = self.activation(hidden_states_out)
        return hidden_states_out
        