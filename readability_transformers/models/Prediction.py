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

import os
import json
import inspect
import importlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
from loguru import logger
from typing import List, Tuple
import tqdm
from tqdm import tqdm
from tqdm.autonotebook import trange
from importlib import import_module


from readability_transformers.readers import PredictionDataReader

class Prediction(nn.Module):
    def __init__(self, input_size: int, double: bool):
        """Defines the Prediction model superclass. All Prediction models that will inherit this class.

        Args:
            input_size (int): length of one row of input.
        """
        super(Prediction, self).__init__()
        self.input_size = input_size
        self.double = double

        if self.double:
            torch.set_default_dtype(torch.float64)
        else:
            torch.set_default_dtype(torch.float32)

    def fit(
        self,
        train_reader: PredictionDataReader,
        valid_reader: PredictionDataReader,
        train_metric: torch.nn.Module,
        evaluation_metrics: List[torch.nn.Module],
        batch_size: int,
        epochs: int,
        lr: float,
        weight_decay: float,
        output_path: str,
        evaluation_steps: int,
        save_best_model: bool,
        show_progress_bar: bool,
        gradient_accumulation: int,
        config: dict,
        device: str
    ):
        """Prediction model fitting code for isolated Prediction model training. This is distinct from
        full ReadabilityTransformer fitting training, which would also have gradients spreading to the
        sentence embedding model as well.
        """
    
        logger.add(os.path.join(output_path, "log.out"))
        self.device = device

        if evaluation_steps % gradient_accumulation != 0:
            logger.warning("Evaluation steps is not a multiple of gradient_accumulation. This may lead to perserve interpretation of evaluations.")
        # 1. Set up training
        train_loader = train_reader.get_dataloader(batch_size=batch_size)
        valid_loader = valid_reader.get_dataloader(batch_size=batch_size)

        self.to(self.device)
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.best_loss = 999999

        training_steps = 0
        self.train()
        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):

            epoch_train_loss = []
            for batch_idx, batch in enumerate(tqdm(train_loader, desc="Iteration", smoothing=0.05, disable=not show_progress_bar)):
                training_steps += 1
                inputs = batch["inputs"].to(self.device)
                targets = batch["target"].to(self.device)
                predicted_scores = self.forward(inputs)

                if "standard_err" in batch.keys():
                    standard_err = batch["standard_err"].to(self.device)
                    loss = train_metric(predicted_scores, targets, standard_err) / gradient_accumulation
                else:
                    loss = train_metric(predicted_scores, targets) / gradient_accumulation
                # loss = 4.0*loss
                loss.backward()
                epoch_train_loss.append(loss.item() * gradient_accumulation) 
            
            
                if (training_steps - 1) % gradient_accumulation == 0 or training_steps == len(train_loader):
                    torch.nn.utils.clip_grad_value_(self.parameters(), 1.)
                    
                    optimizer.step()
                    optimizer.zero_grad()
                
                if training_steps % evaluation_steps == 0:
                    logger.info(f"Evaluation Step: {training_steps} (Epoch {epoch}) epoch train loss avg={np.mean(epoch_train_loss)}")
                    self._eval_during_training(
                        valid_loader=valid_loader,
                        evaluation_metrics=evaluation_metrics,
                        output_path=output_path,
                        save_best_model=save_best_model,
                        current_step=training_steps,
                        config=config
                    )
       
            logger.info(f"Epoch {epoch} train loss avg={np.mean(epoch_train_loss)}")
            epoch_train_loss = []
            # One epoch done.
            self._eval_during_training(
                valid_loader=valid_loader,
                evaluation_metrics=evaluation_metrics,
                output_path=output_path,
                save_best_model=save_best_model,
                current_step=training_steps,
                config=config
            )

    def _eval_during_training(
        self, 
        valid_loader: torch.utils.data.DataLoader,
        evaluation_metrics: List[torch.nn.Module],
        output_path: str, 
        save_best_model: bool,
        current_step: int,
        config: dict
    ):
        with torch.no_grad():
            losses = dict()
            for eval_metric in evaluation_metrics:
                eval_metric_name = eval_metric.__class__.__name__
                losses[eval_metric_name] = []
            for batch_idx, batch in enumerate(valid_loader):
                inputs = batch["inputs"].to(self.device)
                targets = batch["target"].to(self.device)
                predicted_scores = self.forward(inputs)

                for eval_metric in evaluation_metrics:
                    eval_metric_name = eval_metric.__class__.__name__
                    loss = eval_metric(predicted_scores, targets)
                    losses[eval_metric_name].append(loss.item())

        sum_loss = 0
        for losskey in losses.keys():
            mean = np.mean(losses[losskey])
            losses[losskey] = mean
            sum_loss += mean
        losses["mean"] = sum_loss / len(losses.keys())
            
        df = pd.DataFrame([losses])
        df.index = [current_step]
        csv_path = os.path.join(output_path, "evaluation_results.csv" )
        df.to_csv(csv_path, mode="a", header=(not os.path.isfile(csv_path)), columns=losses.keys()) # append to current fike.
        
        if save_best_model:
            if sum_loss < self.best_loss:
                self.save(output_path, config)
                self.best_loss = sum_loss

    
    def save(self, path, config):
        if path is None:
            return

        os.makedirs(path, exist_ok=True)
        logger.info("Save model to {}".format(path))
        torch.save(self, os.path.join(path, "pytorch_model.bin"))

        module_parameters = self.config()
        json.dump(module_parameters, open(os.path.join(path, "config.json"), "w"))
        pickle.dump(config, open(os.path.join(path, "RTconfig.pkl"), "wb"))



class ResFCPrediction(Prediction):
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
            
        if self.double:
            torch.set_default_dtype(torch.float64)
        else:
            torch.set_default_dtype(torch.float32)

        self.input_size = input_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.norm_1 = nn.LayerNorm(input_size)
        self.fc_1 = nn.Linear(input_size, h_size, bias=True)
        self.af_1 = nn.ReLU()

        for n in range(2, self.n_layers):
            fc_layer_name = f"fc_{n}"
            af_name = f"af_{n}"
            norm= f"norm_{n}"
            drop = f"dropout_{n}"
            setattr(self,norm, nn.LayerNorm(h_size))
            setattr(self, fc_layer_name, nn.Linear(h_size, h_size, bias=True))
            setattr(self, af_name, nn.ReLU())
            setattr(self, drop, nn.Dropout(dropout))
        self.norm_last = nn.LayerNorm(h_size)
        self.fc_last = nn.Linear(h_size, 1, bias=True)
        self.af_last = nn.ReLU()


    def forward(self, x):
        assert x.size(1) == self.input_size
        
        if self.double:
            x = x.type(torch.float64)
        else:
            x = x.type(torch.float32)

        x = self.norm_1(x)
        x = self.fc_1(x)
        x = self.af_1(x)

        for n in list(range(2, self.n_layers)):
            norm = getattr(self, f"norm_{n}")
            layer = getattr(self, f"fc_{n}")
            af = getattr(self, f"af_{n}")
            drop = getattr(self, f"dropout_{n}")
            
            x = norm(x)
            x = x + af(drop(layer(x)))
           
        x = self.norm_last(x)
        x = self.fc_last(x)
        x = self.af_last(x)
        x = torch.squeeze(x, dim=-1)
        return x

    def config(self):
        return {
            "input_size": self.input_size,
            "n_layers": self.n_layers,
            "dropout": self.dropout,

        }



class FCPrediction(Prediction):
    def __init__(self, input_size: int, n_layers: int, h_size: int, double: bool):
        """PREDICTION_MODEL_1: Fully-Connected NN. 
        This is MODEL_1 of the list of possible tail-end models for the ReadabilityTransformer,
        where it predicts the Readability score given the features extracted and the embedding from the SentenceTransformer.

        Args:
            input_size (int): # of values per one row of data.
            n_layers (int): number of layers for the fully connected NN.
            h_size (int): Size of hidden layer
        """
        super().__init__(input_size, double)
            
        if self.double:
            torch.set_default_dtype(torch.float64)
        else:
            torch.set_default_dtype(torch.float32)

        self.input_size = input_size
        self.n_layers = n_layers

        self.norm_1 = nn.LayerNorm(input_size)
        self.fc_1 = nn.Linear(input_size, h_size, bias=True)
        self.af_1 = nn.ReLU()
        for n in range(2, self.n_layers):
            fc_layer_name = f"fc_{n}"
            af_name = f"af_{n}"
            norm= f"norm_{n}"
            setattr(self,norm, nn.LayerNorm(h_size))
            setattr(self, fc_layer_name, nn.Linear(h_size, h_size, bias=True))
            setattr(self, af_name, nn.ReLU())
        self.norm_last = nn.LayerNorm(h_size)
        self.fc_last = nn.Linear(h_size, 1, bias=True)
        self.af_last = nn.ReLU()

    

    def forward(self, x):
        assert x.size(1) == self.input_size
        
        if self.double:
            x = x.type(torch.float64)
        else:
            x = x.type(torch.float32)

        for n in range(1, self.n_layers):
            fc_layer_name = f"fc_{n}"
            af_name = f"af_{n}"
            norm_name = f"norm_{n}"
            norm = getattr(self, norm_name)
            layer = getattr(self, fc_layer_name)
            af = getattr(self, af_name)
            x = norm(x)
            x = layer(x)
            x = af(x)
        x = self.norm_last(x)
        x = self.fc_last(x)
        x = self.af_last(x)
        x = torch.squeeze(x, dim=-1)
        return x

    def config(self):
        return {
            "input_size": self.input_size,
            "n_layers": self.n_layers
        }
