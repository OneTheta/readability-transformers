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
import tqdm
import pickle
import inspect
import importlib
from importlib import import_module

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from loguru import logger
from typing import List, Tuple
from tqdm import tqdm
from tqdm.autonotebook import trange

from ..readers import PredictionDataReader

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

                if "standard_err" in batch.keys() and "standard_errors" in train_metric.forward.__code__.co_varnames:
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
                    self.train()
    
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
            self.train()

    def _eval_during_training(
        self, 
        valid_loader: torch.utils.data.DataLoader,
        evaluation_metrics: List[torch.nn.Module],
        output_path: str, 
        save_best_model: bool,
        current_step: int,
        config: dict
    ):
        self.eval()
        with torch.no_grad():
            losses = dict()
            for eval_metric in evaluation_metrics:
                eval_metric_name = eval_metric.__class__.__name__
                losses[eval_metric_name] = []

            targets_collect = []
            predictions_collect = []
            for batch_idx, batch in enumerate(valid_loader):
                inputs = batch["inputs"].to(self.device)
                targets = batch["target"].to(self.device)
                predicted_scores = self.forward(inputs)

                targets_collect.append(targets)
                predictions_collect.append(predicted_scores)
            
            if len(targets_collect) > 1:
                targets_full = torch.stack(targets_collect[:-1], dim=0).flatten(end_dim=1)
                targets_full = torch.cat((targets_full, targets_collect[-1]), dim=0)
                predictions_full = torch.stack(predictions_collect[:-1], dim=0).flatten(end_dim=1)
                predictions_full = torch.cat((predictions_full, predictions_collect[-1]), dim=0) # because last batch may not be full
            else:
                targets_full = targets_collect[0]
                predictions_full = predictions_collect[0]
                
            for eval_metric in evaluation_metrics:
                eval_metric_name = eval_metric.__class__.__name__
                loss = eval_metric(predictions_full, targets_full)
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

        self.train()

    
    def save(self, path, config):
        if path is None:
            return

        os.makedirs(path, exist_ok=True)
        logger.info("Save model to {}".format(path))
        torch.save(self, os.path.join(path, "pytorch_model.bin"))

        module_parameters = self.config()
        json.dump(module_parameters, open(os.path.join(path, "config.json"), "w"))
        pickle.dump(config, open(os.path.join(path, "RTconfig.pkl"), "wb"))




class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, hidden_size, hidden_dropout, num_labels, double: bool = True):
        super().__init__()

        if double:
            torch.set_default_dtype(torch.float64)
        else:
            torch.set_default_dtype(torch.float32)

        self.double = double
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features

        if self.double:
            x = x.type(torch.float64)
        else:
            x = x.type(torch.float32)

        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RegressionHead(nn.Module):
    """Head for sentence-level regression tasks."""

    def __init__(self, hidden_size, hidden_dropout, double: bool = True):
        super().__init__()

        if double:
            torch.set_default_dtype(torch.float64)
        else:
            torch.set_default_dtype(torch.float32)

        self.double = double
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout)
        self.out_proj = nn.Linear(hidden_size, 1)

    def forward(self, features, **kwargs):
        x = features

        if self.double:
            x = x.type(torch.float64)
        else:
            x = x.type(torch.float32)

        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x