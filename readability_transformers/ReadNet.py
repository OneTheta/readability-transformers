import os
import re
import time
import json
import shutil
import pickle
import requests
from typing import List, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loguru import logger
from tqdm import tqdm
from tqdm.autonotebook import trange
from sentence_transformers import SentenceTransformer, models

from .readers import DeepFeaturesDataReader

from .ReadabilityTransformer import ReadabilityTransformer
from .models import ReadNetModel
from .file_utils import load_from_cache_pickle, save_to_cache_pickle, path_to_rt_model_cache, download_rt_model


class ReadNet(ReadabilityTransformer):
    def __init__(
        self,
        model_name: str,
        device: str,
        double: bool,
        new_checkpoint_path: str = None,
        **kwargs
    ):
        super(ReadNet, self).__init__(
            model_name=model_name, 
            device=device, 
            double=double, 
            new_checkpoint_path=new_checkpoint_path
        )
        
        if hasattr(self, "model") and self.model is not None:
            logger.info("Loaded ReadNet model.")
            # alre4ady loaded
            # if kwargs is not None:
            #     raise Exception("Supplied model initialization configs when model is already loaded.")
        else:
            # if not already loaded, at least the st_model has already been loaded with param:model_name
            logger.info(f"Initializing readnet with args: {kwargs}")
            self.init_readnet(
                st_model=self.st_model,
                device=device,
                **kwargs
            )
        
        torch_dtype = torch.double if self.double else torch.float32
        self.model = self.model.to(self.device, torch_dtype)

    def get_st_model(self):
        if hasattr(self, "st_model") and self.st_model is not None:
            return self.st_model
        elif hasattr(self, "model") and self.model is not None:
            return self.model.sent_block.blocks

            
    def setup_load_checkpoint(self, model_path: str):
        torch_path = os.path.join(model_path, "pytorch_model.bin")

        if os.path.isfile(torch_path):
            logger.info(f"Readnet model found at {model_path}")
            model = torch.load(torch_path, map_location=self.device)

            rt_config_path = os.path.join(model_path, "RTconfig.pkl")
            rt_config = pickle.load(open(rt_config_path, "rb"))
            for key in rt_config.keys():
                attribute = rt_config[key]
                if torch.is_tensor(attribute):
                    attribute.to(self.device)
                
                setattr(model, key, attribute)
            self.model = model
            self.tokenizer = model.sent_block.blocks.tokenizer
            return model
        else:
            logger.info("RaedNet model not found. Creating new...")
            os.makedirs(model_path, exist_ok=True)
            return None

        
        

        # Readnet has a different model checkpoint structure.
    
    def init_readnet(self, st_model, d_model=768, n_heads=6, n_blocks=6, n_feats_sent=145, n_feats_doc=223, device="cuda"):
        model = ReadNetModel(
            sentence_transformers = st_model,
            d_model = d_model,
            n_heads = n_heads,
            n_blocks = n_blocks,
            n_feats_sent = n_feats_sent,
            n_feats_doc = n_feats_doc
        )
        model = model.to(device)
        self.model = model
        self.tokenizer = st_model.tokenizer
        return model

        

    def init_st_model(self, st_model = None, st_model_name: str = None, max_seq_length: int = None):
        if max_seq_length is None:
                raise Exception("Parameter max_seq_length must be supplied if initializing an empty SentenceTransformer.")
        
        if os.path.isdir(st_model_name):
            # it's a directory to an already trained SentenceTransformer model:
            # e.g. "checkpoints/stsb/0_SentenceTransformer/"
            st_model = SentenceTransformer(st_model_name, device=self.device)
        else:
            # it's a huggingface checkpoint
            # e.g. "bert-base-uncased"
            st_word_embedding_model = models.Transformer(st_model_name, max_seq_length=max_seq_length)
            st_pooling_model = models.Pooling(st_word_embedding_model.get_word_embedding_dimension())
            st_dense_model = models.Dense(
                in_features=st_pooling_model.get_sentence_embedding_dimension(), 
                out_features=max_seq_length,
                activation_function=nn.Tanh()
            )
            st_model = SentenceTransformer(
                modules=[st_word_embedding_model, st_pooling_model, st_dense_model],
                device=self.device
            )
        
        st_model.to(self.device)
        self.st_model = st_model
        self.tokenizer = st_model.tokenizer
        return st_model

    def fit(
        self,
        train_reader: DeepFeaturesDataReader,
        valid_reader: DeepFeaturesDataReader,
        train_metric: torch.nn.Module,
        evaluation_metrics: List[torch.nn.Module],
        batch_size: int,
        epochs: int,
        lr: float,
        weight_decay: float,
        evaluation_steps: int,
        save_best_model: bool,
        show_progress_bar: bool,
        gradient_accumulation: int,
        start_saving_from_step: int = 0,
        freeze_trf_steps: int = None,
        output_path: str = None,
        device: str = "cpu"
    ):
        """
        One way to train the ReadabilityTransformers is by doing st_fit() then rp_fit(), where these are trained
        separately. Another way is through this fit() method, which trains this entire architecture end-to-end,
        from the SentenceTransformer to the final regression prediction.
        """
        if output_path is None:
            output_path = self.model_path
        config = self.get_config()
        logger.add(os.path.join(output_path, "log.out"))
        if evaluation_steps % gradient_accumulation != 0:
            logger.warning("Evaluation steps is not a multiple of gradient_accumulation. This may lead to perserve interpretation of evaluations.")
        if output_path is None:
            output_path = self.model_path
        
        # 1. Set up training
        train_loader = train_reader.get_dataloader(batch_size=batch_size)
        valid_loader = valid_reader.get_dataloader(batch_size=batch_size)

        self.device = device
        self.model.to(self.device)

        self.freeze_trf_steps = freeze_trf_steps
        if self.freeze_trf_steps is not None and self.freeze_trf_steps > 0:
            for param in self.get_st_model().parameters():
                param.requires_grad = False
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        self.best_loss = 9999999
        training_steps = 0
        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            epoch_train_loss = []
            for batch_idx, batch in enumerate(tqdm(train_loader, desc="Iteration", smoothing=0.05, disable=not show_progress_bar)):
                training_steps += 1

                tokenized_inputs = batch["inputs"]
                sent_feats = batch["sentence_features"].to(self.device)
                doc_feats = batch["document_features"].to(self.device)
                targets = batch["target"].to(self.device)

                predicted_scores = self.model.forward(tokenized_inputs, sent_feats, doc_feats)

                if "standard_err" in batch.keys():
                    standard_err = batch["standard_err"].to(self.device)
                    loss = train_metric(predicted_scores, targets, standard_err) / gradient_accumulation
                else:
                    loss = train_metric(predicted_scores, targets) / gradient_accumulation
                # loss = 4.0*loss

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
                        current_step=training_steps,
                        config=config,
                        start_saving_from_step=start_saving_from_step
                    )
                if self.freeze_trf_steps is not None and self.freeze_trf_steps > 0:
                    if training_steps >= self.freeze_trf_steps:
                        print("Unfreezing trf model...")
                        for param in self.get_st_model().parameters():
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
                current_step=training_steps,
                config=config,
                start_saving_from_step=start_saving_from_step
            )

    def _eval_during_training(
        self, 
        valid_loader: torch.utils.data.DataLoader,
        evaluation_metrics: List[torch.nn.Module],
        output_path: str, 
        save_best_model: bool,
        current_step: int,
        config: dict,
        start_saving_from_step: int = 0
    ):
        self.model.eval()

        targets_collect = []
        predictions_collect = []
        with torch.no_grad():
            losses = dict()
            for eval_metric in evaluation_metrics:
                eval_metric_name = eval_metric.__class__.__name__
                losses[eval_metric_name] = []

            for batch_idx, batch in enumerate(valid_loader):
                tokenized_inputs = batch["inputs"]
                sent_feats = batch["sentence_features"].to(self.device)
                doc_feats = batch["document_features"].to(self.device)
                targets = batch["target"].to(self.device)

                predicted_scores = self.model.forward(tokenized_inputs, sent_feats, doc_feats)

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
        
        # if save_best_model:
        #     if current_step > start_saving_from_step:
        #         if sum_loss < self.best_loss:
        #             self.save(output_path, config)
        #             self.best_loss = sum_loss
        self.save(output_path, config)
        self.best_loss = sum_loss
            
        self.model.train()
      
    def save(self, path, config):
        if path is None:
            return

        logger.info("Saving model to {}".format(path))
        
        torch.save(self.model, os.path.join(path, "pytorch_model.bin"))
        config = self.get_config() # stuff to do with features n such
        pickle.dump(config, open(os.path.join(path, "RTconfig.pkl"), "wb"))

    def forward(self, tokenized_inputs, sent_feats, doc_feats):
        
        return self.model(tokenized_inputs, sent_feats, doc_feats)

    def get_config(self):
        return dict()

                