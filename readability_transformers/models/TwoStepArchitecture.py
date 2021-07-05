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
import torch.optim as optim
import pickle
from loguru import logger
from typing import List, Tuple, Union
import tqdm
from tqdm import tqdm
from tqdm.autonotebook import trange
from importlib import import_module

from sentence_transformers import SentenceTransformer, InputExample

from readability_transformers.readers import PairwiseDataReader
from .TransformersEncoder import TransformersEncoder
from .Prediction import FCPrediction

class TwoStepArchitecture(nn.Module):
    def __init__(self, sentence_transformer: SentenceTransformer, device: str, double: bool):
        super(TwoStepArchitecture, self).__init__()
        self.sentence_transformer = sentence_transformer
        self.device = device
        self.double = double
        
        if self.double:
            torch.set_default_dtype(torch.float64)
        else:
            torch.set_default_dtype(torch.float32)

    def fit(
        self,
        train_reader: Union[PairwiseDataReader, list],
        valid_reader: PairwiseDataReader,
        train_metric: torch.nn.Module,
        evaluation_metrics: List[torch.nn.Module],
        batch_size: int,
        write_csv: str,
        epochs: int,
        gradient_accumulation: int,
        warmup_steps: int,
        lr: int,
        weight_decay: int,
        output_path: str,
        evaluation_steps: int,
        show_progress_bar: bool,
        save_best_model: bool,
        freeze_trf_steps: int
    ):
        train_loaders = []
        if type(train_reader) == list:
            for epoch in range(epochs):
                train_loader = train_reader[epoch].get_dataloader(batch_size=batch_size)
                train_loader.collate_fn = self.smart_batching_collate
                train_loaders.append(train_loader)
        assert len(train_loaders) == epochs

        valid_loader = valid_reader.get_dataloader(batch_size=batch_size)
        valid_loader.collate_fn = self.smart_batching_collate

        logger_handler = logger.add(os.path.join(output_path, "log.out"))
    
        if evaluation_steps % gradient_accumulation != 0:
            logger.warning("Evaluation steps is not a multiple of gradient_accumulation. This may lead to perserve interpretation of evaluations.")
        
        self.freeze_trf_steps = freeze_trf_steps
        self.model.to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.best_loss = 99999

        training_steps = 0
        self.model.train()

        if self.freeze_trf_steps is not None and self.freeze_trf_steps > 0:
            for param in self.model.sentence_transformer.parameters():
                param.requires_grad = False

        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            if type(train_reader) == list:
                train_loader = train_loaders[epoch]
            epoch_train_loss = []
            for batch_idx, batch in enumerate(tqdm(train_loader, desc="Iteration", smoothing=0.05, disable=not show_progress_bar)):
                training_steps += 1

                # batch = List[InputExample]
                sentence_features, labels = batch
                predicted_scores = self.model.forward(sentence_features)
                
                loss = train_metric(predicted_scores, labels) / gradient_accumulation
                
                loss.backward()
                epoch_train_loss.append(loss.item() * gradient_accumulation)

                if (training_steps - 1) % gradient_accumulation == 0 or training_steps == len(train_loader):
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), 1.)
                    
                    optimizer.step()
                    optimizer.zero_grad()

                if training_steps % evaluation_steps == 0:
                    logger.info(f"Evaluation Step: {training_steps} (Epoch {epoch}) epoch train loss avg={np.mean(epoch_train_loss)}")
                    self._eval_during_training(
                        valid_loader=valid_loader,
                        evaluation_metrics=evaluation_metrics,
                        output_path=output_path,
                        save_best_model=save_best_model,
                        current_step=training_steps
                    )

                
                if self.freeze_trf_steps is not None and self.freeze_trf_steps > 0:
                    if training_steps >= self.freeze_trf_steps:
                        print("Unfreezing trf model...")
                        for param in self.model.sentence_transformer.parameters():
                            assert param.requires_grad == False # shoudve been set off
                            param.requires_grad = True
                        self.freeze_trf_steps = None

            logger.info(f"Epoch {epoch} train loss avg={np.mean(epoch_train_loss)}")
            epoch_train_loss = []
            # One epoch done.
            self._eval_during_training(
                valid_loader=valid_loader,
                evaluation_metrics=evaluation_metrics,
                output_path=output_path,
                save_best_model=save_best_model,
                current_step=training_steps
            )
    
        logger.remove(logger_handler)

    def _eval_during_training(
        self, 
        valid_loader: torch.utils.data.DataLoader,
        evaluation_metrics: List[torch.nn.Module],
        output_path: str, 
        save_best_model: bool,
        current_step: int
    ):
        with torch.no_grad():
            losses = dict()
            for eval_metric in evaluation_metrics:
                eval_metric_name = eval_metric.__class__.__name__
                losses[eval_metric_name] = []
            for batch_idx, batch in enumerate(valid_loader):
                sentence_features, targets = batch
                predicted_scores = self.model.forward(sentence_features)

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
                self.sentence_transformer.save(output_path)
                self.best_loss = sum_loss

    def smart_batching_collate(self, batch):
        """
        FROM sentence_transformers.

        Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model
        Here, batch is a list of tuples: [(tokens, label), ...]
        :param batch:
            a batch from a SmartBatchingDataset
        :return:
            a batch of tensors for the model
        """
        num_texts = len(batch[0].texts)
        texts = [[] for _ in range(num_texts)]
        labels = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text)

            labels.append(example.label)

        labels = torch.tensor(labels).to(self.device)

        sentence_features = []
        for idx in range(num_texts):
            tokenized = self.sentence_transformer.tokenize(texts[idx])
            self.batch_to_device(tokenized)
            sentence_features.append(tokenized)

        if self.double:
            labels = labels.type(torch.float64)
        else:
            labels = labels.type(torch.float32)
        return sentence_features, labels

    def batch_to_device(self, batch):
        """
        send a pytorch batch to a device (CPU/GPU)
        """
        target_device = self.device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(target_device)
            
            if self.double:
                if batch[key].dtype == torch.float32:
                    batch[key] = batch[key].type(torch.float64)
            else:
                if batch[key].dtype == torch.float64:
                    batch[key] = batch[key].type(torch.float32)

        return batch


class TwoStepTRFPrediction(TwoStepArchitecture):
    def __init__(self, **kwargs):
        super().__init__(
            sentence_transformer=kwargs["sentence_transformer"], 
            device=kwargs["device"], 
            double=kwargs["double"]
        )
        self.model = TwoStepTRFPrediction_Wrapper(
            kwargs["embedding_size"],
            kwargs["sentence_transformer"],
            kwargs["n_layers"],
            kwargs["h_size"],
            kwargs["encoder_layers"],
            kwargs["trf_heads"],
            kwargs["trf_dropout"],
            kwargs["device"],
            kwargs["double"]
        )


class TwoStepTRFPrediction_Wrapper(nn.Module):
    def __init__(
        self, 
        embedding_size: int, 
        sentence_transformer: SentenceTransformer, 
        n_layers: int, 
        h_size: int, 
        encoder_layers: int,
        trf_heads: int,
        trf_dropout: int,
        device: str,
        double: bool,
    ):
        """Given a pair of texts, we apply sentence_transformers to obtain two embeddings.
        then concatenate to [embed_1 | embed_2], which is then => transformers_encoder => [cross_encoded]
        [cross_encoded] => FC => prediction of their difficulty similarity score.

        This is to test the approach of pre-training the sentence_transformers in a different, related prediction task like this.

        In the actual ReadabilityTransformers, we then only take out the sentence_transformers from this setup
        to use as the text encoding. The intuition is that if this TwoStepTRFPrediction is effective, so will the sentence_transformers.
        """
        # super().__init__(sentence_transformer, device, double)
        super(TwoStepTRFPrediction_Wrapper, self).__init__()

        self.device = device
        self.embedding_size = embedding_size
        self.n_layers = n_layers
        self.h_size = h_size
        self.encoder_layers = encoder_layers
        self.trf_heads = trf_heads
        self.trf_dropout = trf_dropout
        self.double = double

        self.sentence_transformer = sentence_transformer
        
        # self.transformers_encoder = TransformersEncoder(
        #     max_seq_len=self.embedding_size * 2,
        #     embedding_size=self.trf_heads,
        #     encoder_layers=self.encoder_layers,
        #     heads=self.trf_heads,
        #     dropout=self.trf_dropout,
        #     device=device
        # )

        self.fully_connected = FCPrediction(
            input_size=self.embedding_size * 2,
            n_layers=self.n_layers,
            h_size=self.h_size,
            double=self.double
        )
        print(self.fully_connected)


    def forward(self, sentence_features):
        embeddings_1 = self.sentence_transformer(sentence_features[0])["sentence_embedding"]
        embeddings_2 = self.sentence_transformer(sentence_features[1])["sentence_embedding"]
        
        combined_embeddings = torch.cat((embeddings_1, embeddings_2), dim=-1)

        # combined_embeddings = combined_embeddings.unsqueeze(dim=-1)
        # combined_embeddings = combined_embeddings.repeat_interleave(self.trf_heads, dim=-1)
        
        # trf_encoded = self.transformers_encoder(combined_embeddings)
        # trf_encoded = torch.mean(trf_encoded, dim=-1)
        trf_encoded = combined_embeddings

        predictions = self.fully_connected(trf_encoded)

        return predictions

            
class TwoStepFCPrediction(TwoStepArchitecture):
    def __init__(
        self, 
        embedding_size: int, 
        sentence_transformer: SentenceTransformer, 
        n_layers: int, 
        h_size: int,
        device: str,
        double: bool,
    ):
        """Given a pair of texts, we apply sentence_transformers to obtain two embeddings.
        then concatenate to [embed_1 | embed_2], which is then => FC => prediction of their difficulty similarity score.

        This is to test the approach of pre-training the sentence_transformers in a different, related prediction task like this.

        In the actual ReadabilityTransformers, we then only take out the sentence_transformers from this setup
        to use as the text encoding. The intuition is that if this TwoStepTRFPrediction is effective, so will the sentence_transformers.
        """
        super().__init__(sentence_transformer, device, double)
        

        self.device = device
        self.embedding_size = embedding_size
        self.n_layers = n_layers
        self.h_size = h_size

        self.sentence_transformer = sentence_transformer

        self.fully_connected = FCPrediction(
            input_size=self.embedding_size * 2,
            n_layers=self.n_layers,
            h_size=self.h_size,
            double=self.double
        )

    def forward(self, sentence_features):
        embeddings_1 = self.sentence_transformer(sentence_features[0])["sentence_embedding"]
        embeddings_2 = self.sentence_transformer(sentence_features[1])["sentence_embedding"]
        
        combined_embeddings = torch.cat((embeddings_1, embeddings_2), dim=-1)

        predictions = self.fully_connected(combined_embeddings)

        return predictions

            
   