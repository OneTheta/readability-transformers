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


from sentence_transformers import SentenceTransformer, models, losses, evaluation

from .readers import PairwiseDataReader, PredictionDataReader, FeaturesDataReader
from .models import Prediction, FCPrediction, ResFCClassification, ResFCRegression, TwoStepArchitecture, TwoStepFCPrediction, TwoStepTRFPrediction
from .features import FeatureBase
from .file_utils import load_from_cache_pickle, save_to_cache_pickle, path_to_rt_model_cache, download_rt_model


CACHE_DIR = os.path.expanduser("~/.cache/readability-transformers/data")
RT_MODEL_LOOKUP = "http://readability.1theta.com/models/"

class ReadabilityTransformer(nn.Module):
    def __init__(
        self,
        model_name: str,
        device: str,
        double: bool,
        new_checkpoint_path: str = None,
        default_st_model: str = None
    ):
        '''Initialize the ReadabilityTransformer, which first requires an instantiation
        of the SentenceTransformer then an instantiation of the Prediction model.

        Args:
            model_name (str): The transformer model to initialize the SentenceTransformer.
                This is the same as the model name in the HuggingFace library list of transformer
                models. e.g. "bert-base-uncased".
            max_seq_length (int): Same usual idea as with HuggingFace transformers. Possible text 
                length the model can have as input.
            device (str): "cpu" or some version of "cuda" to run the torch modules in.
            new_checkpoint_path (str): if set, copies the model at model_path to the new_checkpoint_path
                directory, which will be our new working directory.
            double (bool): Whether this model should use weights that are float32 or float64. 
        '''
        super(ReadabilityTransformer, self).__init__()

        # if [None, None] == [model_name, new_checkpoint_path]:
        #     raise Exception("Must either provide a valid path to a checkpoint or a model supported by ReadabilityTransformers through model_name or new_checkpoint_path to create new.")

        self.model_name = model_name
        self.device = device
        self.double = double
        self.model_path = None
        self.st_model = None
        self.rp_model = None
        self.new_checkpoint_path = new_checkpoint_path

        if self.model_name:
            self.model_path = self.process_model_path(self.model_name)
            if self.model_path is not None:
                self.setup_load_checkpoint(self.model_path)


        if self.new_checkpoint_path is not None:
            if os.path.isdir(new_checkpoint_path) and self.model_name is None:
                logger.warning("You are initializing a ReadabilityTransformer model even though the folder exists already. This will reset the current checkpoint. Are you sure?")
                logger.warning("Waiting 3 seconds just in case...")
                time.sleep(3)
                shutil.rmtree(self.new_checkpoint_path)

            if self.model_path:
                shutil.copytree(self.model_path, new_checkpoint_path)
            else:
                os.makedirs(self.new_checkpoint_path, exist_ok=True)
                self.model_path = self.new_checkpoint_path
                self.setup_load_checkpoint(self.new_checkpoint_path)

            self.model_path = self.new_checkpoint_path
        elif self.new_checkpoint_path is None and self.model_path is None:
            print("Initializing readability transformers without checkpoint, must supply output_path to fit later on.")
    
        

    def process_model_path(self, model_path: str):
        '''
        model_path can be:
            1. A folder to a local checkpoint
            2. A model name that exists in the ReadabilityTransformers system
            3. A url to a zip file containing a valid ReadabilityTransformers checkpoint.
            4. a model name that exists in the huggingface system (e.g. "roberta-base")
        
        For #2 and #3, we try to save the model to ~/.cache/readability-transformers/models/
        '''
        if os.path.isdir(model_path):
            return model_path
        else:
            if "http" not in model_path:
                # Maybe it's case 2:
                local_path = path_to_rt_model_cache(model_path)

                if local_path is not None:
                    return local_path
                else:    
                    try:
                        url = requests.get(RT_MODEL_LOOKUP + model_path).json()["url"]
                        if url is not None:
                            local_path = download_rt_model(url)
                            return local_path
                    except:
                        pass
                    
                    new_st_model = self.init_st_model(st_model_name=model_path, max_seq_length=512)
                    return None
                            
                    # maybe it's a huggingface model:

            else:       
                # case 3
                local_path = download_rt_model(url)
                return local_path

        

    def setup_load_checkpoint(self, model_path):
        st_path = os.path.join(model_path, "0_SentenceTransformer")
        st_tf_path = os.path.join(st_path, "0_Transformer")

        rp_path = os.path.join(model_path, "1_Prediction")
        rp_torch_path = os.path.join(rp_path, "pytorch_model.bin")

        if os.path.isdir(model_path):
            logger.info(f"Model folder found at {model_path}.")
            if os.path.isdir(st_path) and os.path.isdir(st_tf_path):
                logger.info(f"SentenceTransformer model found at {st_path}. Loading...")
                self.st_model, self.st_loss = self._load_st_model()
                logger.info("Loaded ST model")
            else:
                logger.info(f"SentenceTransformer model not found at {st_path}.")
                os.makedirs(st_path, exist_ok=True)

            if os.path.isdir(rp_path) and os.path.isfile(rp_torch_path):
                logger.info(f"Prediction model found at {rp_path}. Loading...")
                self.rp_model = self._load_rp_model()
                logger.info("Loaded Prediction model")
            else:
                logger.info(f"Prediction model NOT found at {rp_path}. Creating new...")
                os.makedirs(rp_path, exist_ok=True)
        else:
            logger.info("Model folder not found. Creating entirely new checkpoint.")
            os.mkdir(model_path)
            os.mkdir(st_path)
            os.mkdir(rp_path)
        
        if self.st_model is None:
            logger.info("SentenceTransformer model is not loaded. Make sure you run init_st_model() to initialize a new instance.")
        if self.rp_model is None:
            logger.info("Prediction model is not loaded. Make sure you run init_rp_model() to initialize a new instance.")
        
        self.DEFAULT_SUB_GROUPS = ['CKKF_', 'POSF_',  # 'PhrF_', 'TrSF_', 
              'EnDF_', 'EnGF_', 'ShaF_', 'TraF_',
              'TTRF_', 'VarF_', 'PsyF_', 'WorF_']
        self.st_path = st_path
        self.rp_path = rp_path

    def _load_st_model(self):
        model_path = self.model_path
        st_model_path = os.path.join(model_path, "0_SentenceTransformer")
        st_model = SentenceTransformer(st_model_path, device=self.device)
        st_loss = losses.CosineSimilarityLoss(self.st_model)

        st_model.to(self.device)
        st_loss.to(self.device)

        return st_model, st_loss

    def _load_rp_model(self):
        model_path = self.model_path
        rp_model_path = os.path.join(model_path, "1_Prediction")
        rp_model_torch_path = os.path.join(rp_model_path, "pytorch_model.bin")

        rp_model = torch.load(rp_model_torch_path, map_location=self.device)
        assert self.double == rp_model.double

        rp_model.to(self.device)
        rp_model.device = self.device

        rp_config_path = os.path.join(rp_model_path, "config.json")
        rt_config_path = os.path.join(rp_model_path, "RTconfig.pkl")

        rp_config = json.load(open(rp_config_path, "r"))
        rt_config = pickle.load(open(rt_config_path, "rb"))

        for key in rp_config.keys():
            attribute = rp_config[key]
            if torch.is_tensor(attribute):
                attribute.to(self.device)
            setattr(rp_model, key, attribute)

        for key in rt_config.keys():
            attribute = rt_config[key]
            if torch.is_tensor(attribute):
                attribute.to(self.device)
            
            setattr(self, key, attribute)

        return rp_model

    def init_st_model(self, st_model: SentenceTransformer = None, st_model_name: str = None, max_seq_length: int = None):
        """Give premade_st_model to just inject a separately built SentenceTransformer
        Give st_model_name & max_seq_length to build a SentenceTransformer with default configs for given huggingface model.
        """
        device = self.device

        if st_model is None:
            if st_model_name is None:
                raise Exception("Must supply either a SentenceTransformer object or st_model_name")

            if self.st_model:
                # already had st_model loaded
                logger.warning("You are initializing a new SentenceTransformer model even though one exists already. This will reset the current checkpoint. Are you sure?")
                logger.warning("Waiting 5 seconds just in case...")
                time.sleep(5)
                shutil.rmtree(self.st_path)
                os.mkdir(self.st_path)

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
                    device=device
                )
                st_model.to(device)
                
        st_loss = losses.CosineSimilarityLoss(st_model)

        self.st_model = st_model
        self.st_loss = st_loss

        # have a saved initial version
        if hasattr(self, "st_path") and self.st_path is not None:
            self.st_model.save(self.st_path)

    def init_rp_model(
        self, 
        rp_model: Union[Prediction, str],
        features: List[str] = None,
        embedding_size: int = None,
        n_layers: int = None,
        h_size: int = 256,
        dropout: int = 0.1,
        n_labels=1 # only used if the request model is a classification model.
    ):

        if self.st_model is None:
            raise Exception("Cannot initialize ReadabilityPrediction model before loading/initializing a SentenceTransformer model.")

        if self.rp_model: # class already has a rp_model loaded
            logger.warning("You are initializing a new Reading Prediction model even though one exists already. This will reset the current checkpoint. Are you sure?")
            logger.warning("Waiting 5 seconds just in case...")
            time.sleep(5)
            shutil.rmtree(self.rp_path)
            os.mkdir(self.rp_path)
        

        if isinstance(rp_model, str):
            if None in [features, embedding_size, n_layers, h_size, dropout]:
                raise Exception("Missing parameters for prediction model instantiation")

            self.features = features
            self.embedding_size = embedding_size
            feature_count = len(self.features)
            input_size = feature_count + embedding_size
            
            if rp_model == "ResFCClassification":
                rp_model = ResFCClassification(input_size, n_layers, h_size, dropout=dropout, n_labels=n_labels, double=self.double)
            elif rp_model == "ResFCRegression":
                rp_model = ResFCRegression(input_size, n_layers, h_size, dropout=dropout, double=self.double)
            else:
                raise Exception(f"Could not find ReadabilityPrediction model {rp_model_name}")

        self.rp_model = rp_model.to(self.device)
        # self.rp_model.features_in_order = features


    def st_fit(
        self, 
        train_readers: List[PairwiseDataReader], 
        valid_reader: PairwiseDataReader, 
        batch_size: int,
        write_csv: str, 
        epochs: int, 
        warmup_steps: int, 
        lr: int, 
        output_path: str, 
        evaluation_steps: int, 
        show_progress_bar: bool
    ):
        output_path = os.path.join(output_path, "0_SentenceTransformer")
        '''The SentenceTransformer is a component to the full ReadabilityTransformer architecture. This function
        trains the SentenceTransformer before training the full ReadabilityTransformer.
        '''
        train_dataloaders = []
        for reader in train_readers:
            train_dataloader = reader.get_dataloader(batch_size=batch_size)
            train_dataloaders.append(train_dataloader)
        
        valid_dataset = valid_reader.get_dataset()

        sentences1 = [i.texts[0] for i in valid_dataset]
        sentences2 = [i.texts[1] for i in valid_dataset]
        scores = [i.label for i in valid_dataset]
        evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, scores, write_csv=write_csv)

        logger.info("Fitting SentenceTransformer model...")
        self.st_model.fit(
            train_objectives=[(loader, self.st_loss) for loader in train_dataloaders], 
            epochs=1, 
            warmup_steps=warmup_steps,
            optimizer_params={"lr": lr},
            output_path=output_path,
            evaluator=evaluator, 
            evaluation_steps=evaluation_steps,
            show_progress_bar=show_progress_bar
        )

    def st_forward(self, sentences):
        return self.st_model.encode(sentences)

    def rp_fit(
        self,
        train_reader: PredictionDataReader,
        valid_reader: PredictionDataReader,
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
        st_embed_batch_size: int = None,
    ):
        """
        @TODO
        st_embed_batch_size: need to implement automatic pre_apply_embeddings() here
        if not done yet.
        """
        device = self.device
        output_path = os.path.join(self.model_path, "1_Prediction")
        config = self.get_config()

        self.rp_model.fit(
            train_reader,
            valid_reader,
            train_metric,
            evaluation_metrics,
            batch_size,
            epochs,
            lr,
            weight_decay,
            output_path,
            evaluation_steps,
            save_best_model,
            show_progress_bar,
            gradient_accumulation,
            config=config,
            device=device
        )

    def get_twostep_trainer(
        self,
        model: str,
        **kwargs
    ) -> TwoStepArchitecture: 
        """Instantiate a TwoStepArchitecture used in pre-training the sentence_transformer bi-encoder.
        After instantiating it, one would use the .fit() function to train the sentence_transformer with it.
        Or just use the ReadabilityTransformers twostep_fit() API.

        Supported models: {"FC", "TRF"}
        """
        model = model.lower()
        
        if self.st_model is None:
            raise Exception("SentenceTransformer model not found.")

        sentence_transformer = self.st_model
        device = self.device
        double = self.double
        
        if model == "fc":
            return TwoStepFCPrediction(
                sentence_transformer=sentence_transformer,
                device=device,
                double=double,
                **kwargs
            )
        elif model == "trf":
            return TwoStepTRFPrediction(
                sentence_transformer=sentence_transformer,
                device=device,
                double=double,
                **kwargs
            )

    def get_config(self):
        """Get the parameters we want to re-load when reloading this RT model.
        """
        feature_extractors = []
        for feature_extractor in self.feature_extractors:
            if hasattr(feature_extractor, "trf_model"):
                del feature_extractor.trf_model
            if hasattr(feature_extractor, "tokenizer"):
                del feature_extractor.tokenizer
            feature_extractors.append(feature_extractor)

        config = {
            "normalize": self.normalize,
            "features": self.features,
            "feature_extractors": feature_extractors,
            "blacklist_features": self.blacklist_features
        }
        if self.normalize:
            config["feature_maxes"] = self.feature_maxes
            config["feature_mins"] = self.feature_mins
        if hasattr(self, "target_max"):
            config["target_max"] = self.target_max
            config["target_min"] = self.target_min


        return config
    
    def fit(
        self,
        train_reader: FeaturesDataReader,
        valid_reader: FeaturesDataReader,
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

        logger.add(os.path.join(output_path, "log.out"))
        self.device = device
        self.to(self.device)
        self.train()
        config = self.get_config()
        if evaluation_steps % gradient_accumulation != 0:
            logger.warning("Evaluation steps is not a multiple of gradient_accumulation. This may lead to perserve interpretation of evaluations.")
        
        # 1. Set up training
        train_loader = train_reader.get_dataloader(batch_size=batch_size)
        valid_loader = valid_reader.get_dataloader(batch_size=batch_size)


        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.best_loss = 9999999
        
        training_steps = 0

        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            epoch_train_loss = []
            for batch_idx, batch in enumerate(tqdm(train_loader, desc="Iteration", smoothing=0.05, disable=not show_progress_bar)):
                training_steps += 1

                inputs = batch["inputs"]
                passage_inputs = inputs["text"]
                feature_inputs = inputs["features"].to(self.device)

                targets = batch["target"].to(self.device)

                predicted_scores = self.forward(passage_inputs, feature_inputs)

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
        self.eval()
        self.st_model.eval()
        self.rp_model.eval()

        targets_collect = []
        predictions_collect = []
        with torch.no_grad():
            losses = dict()
            for eval_metric in evaluation_metrics:
                eval_metric_name = eval_metric.__class__.__name__
                losses[eval_metric_name] = []

            for batch_idx, batch in enumerate(valid_loader):
                inputs = batch["inputs"]
                passage_inputs = inputs["text"]
                feature_inputs = inputs["features"].to(self.device)

                targets = batch["target"].to(self.device)
                predicted_scores = self.forward(passage_inputs, feature_inputs)

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
        self.st_model.train()
        self.rp_model.train()

    
    def save(self, path, config):
        if path is None:
            return

        logger.info("Saving model to {}".format(path))
        logger.info("Saving to 0_SentenceTransformers...")
        self.st_model.save(os.path.join(path, "0_SentenceTransformer"))
        logger.info("Saving to 1_Prediction...")
        self.rp_model.save(os.path.join(path, "1_Prediction"), self.get_config())



    def forward(self, passages: List[str], features: List[List[Union[float, np.array, torch.Tensor]]]):
        """Forward pass of the model with gradients whose outputs you can train with a loss function backwards pass.
        No normalization/denormalization efforts are done. This is a pure full RT model pass.

        Args:
            passages: text passages to derive readability score predictions from.
            features: each element in the features parameters is an array of feature values for corresponding
                passage in passages.
        """
        
        assert len(features) == len(passages)
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.double if self.double else torch.float32).to(self.device)
        features.to(self.device)
        """
        1. Get forward pass from st_model. This code is based on UKPLab/sentence-transformers.
        """

        

        text_features = self.st_model.tokenize(passages)
        for key in text_features:
            text_features[key] = text_features[key].to(self.device)
        
        out_features = self.st_model(text_features)
        embeddings = out_features["sentence_embedding"].to(self.device)
        """
        2. model_inputs = concat(feature_array, embeddings)
        """
        N = len(passages)
        E = features.size(-1)
        F = embeddings.size(-1)

        rp_inputs = torch.cat((features, embeddings), axis=-1).to(self.device)
        assert rp_inputs.size() == (N, E+F)

        """
        4. Predict
        """
 

        predicted_scores = self.rp_model(rp_inputs)
        return predicted_scores
    

    def predict(self, passages: List[str], batch_size: int = 1):
        """This is an inference function without gradients. For a forward pass of the model with gradients
        that lets you train the model, refer to self.forward(). 

        Args:
            passages (List[str]): list of texts to predict readability scores on.
            batch_size (int): batch_size to pass the input data through the model.
        """
        passages = self.basic_preprocess_text(passages)
        self.to(self.device)
        self.st_model.to(self.device)
        self.rp_model.to(self.device)
        self.eval()
        self.st_model.eval()
        self.rp_model.eval()
        
        """
        1. Sentences -> Lingfeat features
        """
        if self.rp_model is None or self.st_model is None:
            raise Exception("Cannot make predictions without both the SentenceTransformer model and the Prediction model loaded.")
        
        # below values must exist if rp_model was loaded.
        features = self.features
        feature_extractors = self.feature_extractors
        blacklist_features = self.blacklist_features

        if self.normalize:
            feature_maxes = self.feature_maxes
            feature_mins = self.feature_mins
            normalize = lambda value, max_val, min_val: max(min((value - min_val) / (max_val - min_val), 0.999), 0.001)
            
            denormalize = lambda value, max_val, min_val: (value * (max_val - min_val)) + min_val

        feature_dict_array = []
        for batch_idx in range(len(passages) // batch_size + 1):
            if batch_idx * batch_size == len(passages):
                break
            one_batch = passages[batch_idx*batch_size : (batch_idx + 1)*batch_size]

            feature_dict_list_collect = []
            for one_extractor in feature_extractors:
                feature_dict_list = one_extractor.extract_in_batches(one_batch)
                assert len(feature_dict_list) == len(one_batch)
                feature_dict_list_collect.append(feature_dict_list)

            for passage_idx in range(len(one_batch)):
                # complete_faetures_per_passage = full feature dictionary corresponding for that passage 
                # from all the feature extractors
                complete_features_per_passage = dict()
                for feature_dict_list in feature_dict_list_collect:
                    passage_feature_dict = feature_dict_list[passage_idx]
                    for key in passage_feature_dict.keys():
                        complete_features_per_passage["feature_"+key]  = passage_feature_dict[key]
                feature_dict_array.append(complete_features_per_passage)
            

        feature_matrix = []
        for passage_idx, feat_dict in enumerate(feature_dict_array):
            passage_feature_num_array = []
            for feat_idx, one_feature in enumerate(features): # preserve order of original model input
                feat_value = feat_dict[one_feature]
                if self.normalize:
                    max_val = feature_maxes[one_feature]
                    min_val = feature_mins[one_feature]
                    feat_value = normalize(feat_value, max_val, min_val)
                passage_feature_num_array.append(feat_value)
            feature_matrix.append(passage_feature_num_array)

        assert len(feature_matrix) == len(passages)
        assert len(feature_matrix[0]) == len(self.features)

        predictions_collect = []
        
        
        with torch.no_grad():
            batch_size = batch_size
            n_batches = len(passages) // batch_size

            for full_batch_idx in range(len(passages) // batch_size + 1):
                if (full_batch_idx * batch_size == len(passages)):
                    break
                start_idx = full_batch_idx * batch_size
                end_idx = (full_batch_idx + 1) * batch_size

                passage_batch = passages[start_idx : end_idx]
                feature_batch = feature_matrix[start_idx : end_idx]
                
                one_prediction = self.forward(passage_batch, feature_batch)
                predictions_collect.append(one_prediction)


            if len(predictions_collect) > 1:
                predictions = torch.stack(predictions_collect[:-1], dim=0).flatten(end_dim=1)
                predictions = torch.cat((predictions, predictions_collect[-1]), dim=0) # because last batch may not be full
            else:
                predictions = predictions_collect[0]

           

        """
        5. Denormalize Prediction
        """
        if hasattr(self, "target_column") and self.target_column is not None and hasattr(self, "target_max") and self.target_max is not None:
            target_max = self.target_max
            target_min = self.target_min
            
            predictions = denormalize(predictions, target_max, target_min)

        return predictions

    def basic_preprocess_text(self, text_input: Union[str, List[str]]) -> str:
        text = text_input
        if isinstance(text_input, str):
            text = [text_input]

        collect = []
        for one_text in text:
            one_text = one_text.replace("\n", " ").replace("\t", " ").replace("  ", " ")
            one_text = one_text.replace("‘", "'").replace(" – ","—")
            # fix_spaces = re.compile(r'\s*([?!.,]+(?:\s+[?!.,]+)*)\s*')
            # one_text = fix_spaces.sub(lambda x: "{} ".format(x.group(1).replace(" ", "")), one_text)


            one_text = one_text.strip()
            collect.append(one_text)
        
        if isinstance(text_input, str):
            return collect[0]
        else:
            return collect

    def pre_apply_features(
        self, 
        df_list: List[pd.DataFrame],
        feature_extractors: List[FeatureBase],
        batch_size: int = None,
        text_column: str = "excerpt",
        target_column: str = None,
        cache: bool = False,
        cache_ids: List[str] = None,
        normalize=False,
        extra_normalize_columns: List[str] = None
    ) -> List[pd.DataFrame]: 
        """Applies the feature classes on the dataframes in df_list such that new columns are appended for each feature.
        ReadabilityTransformers is able to do such feature extraction on-the-fly as well. But this can be inefficient when
        training multiple epochs, since the same feature extraction will occur repeatedly. This function allows one feature
        extraction pass to re-use over and over again.

        Args:
            df_list (List[pd.DataFrame]): List of Pandas DataFrames that contain text to apply the features to, the column name of which
                is specified by the text_column argument.
            feature_classes (List[FeatureBase]): List of Feature classes to use to apply features to the text.
            text_column: Name of the column in the dataframe that contains the text in interest. Defaults to "excerpt".
            target_column: Name of the column in the dataframe that contains the target value of the model.
            cache (boolf): Whether to cache the dataframes generated after applying the features. Feature extraction can be a costly
                process that might serve well to do only once per dataset. Defaults to False.
            cache_ids (List[str]): Must be same length as df_list. The ID value for the cached feature dataset generated from the datasets
                in df_list. Defaults to None.
            normalize (bool): Normalize the feature values to 0 to 1. Defaults to True.
            extra_normalize_columns (List[str]): Name of existing columns to apply normalization to as well, since otherwise
                normalization will only be applied to the features extracted from this function. 
        
        Returns:
            feature_dfs (List[pd.DataFrame]): List of pd.DataFrames that are copies of df_list except with 
                feature columns.
        """
        features_applied = dict() 

        if cache_ids:
            if len(cache_ids) != len(df_list):
                raise Exception("The list of labels must match the df_list parameter.")
        if cache:
            if cache_ids is None:
                raise Exception("Cache set to true but no label for the cache.")
            else:
                # Load what we can from cache.
                for cache_id in cache_ids:
                    cache_loaded = load_from_cache_pickle("preapply", f"features_{cache_id}")
                    if cache_loaded is not None:
                        logger.info(f"Found & loading cache ID: {cache_id}...")
                        features_applied[cache_id] = cache_loaded

        for one_df in df_list:
            col_list = one_df.columns.values
            already_features = [col for col in col_list if col.startswith("feature_")]
            if len(already_features) > 0:
                raise Exception("DataFrame has columns starting with 'feature_'. This is not allowed since applied features will take this format.")
       
        if extra_normalize_columns is None:
            extra_normalize_columns = []
        else:
            if not normalize:
                raise Exception("normalize is set to False but extra_normalize_columns is supplied.")
        
        remaining_ids = list(range(len(df_list))) # default ids are just counters
        remaining = df_list
        if cache:
            current_loaded = features_applied.keys()
            assert set(current_loaded).issubset(set(cache_ids))
            remaining_ids = list(set(cache_ids).difference(set(current_loaded)))
            remaining = [df_list[cache_ids.index(remain_id)] for remain_id in remaining_ids]
        # remaining == dataframes that we have not yet loaded from cache.
        # obviously if cache == False then remaining == full requested df_list

        for remain_id, remain_data in zip(remaining_ids, remaining):
            logger.info(f"Extracting features for cache_id={remain_id}")

            if batch_size is not None:
                # extract in baches
                text_list = remain_data[text_column].values
                for batch_idx in trange(len(text_list) // batch_size + 1):
                    if batch_idx * batch_size == len(text_list):
                        break
                    one_batch = list(text_list[batch_idx*batch_size : (batch_idx + 1)*batch_size])
                

                    feature_dict_list_collect = []
                    for one_extractor in feature_extractors:
                        feature_dict_list = one_extractor.extract_in_batches(one_batch)
                        assert len(feature_dict_list) == len(one_batch)
                        feature_dict_list_collect.append(feature_dict_list)

                    # feature_dict_list_collect.shape == (#features, #passages)
                    passages_features = [] # trying to get (#passages, #features)
                    for passage_idx in range(len(one_batch)):
                        # complete_faetures_per_passage = full feature dictionary corresponding for that passage 
                        # from all the feature extractors
                        complete_features_per_passage = dict()
                        for feature_dict_list in feature_dict_list_collect:
                            passage_feature_dict = feature_dict_list[passage_idx]
                            for key in passage_feature_dict.keys():
                                complete_features_per_passage[key] = passage_feature_dict[key]
                        passages_features.append(complete_features_per_passage)
                        
                    '''
                    passages_features = [
                        {text_1_feature_1, text_1_feature_2, text_1_feature_3, ...},
                        {text_2_feature_1, text_2_feature_2, text_2_feature_3, ...},
                        ...
                    ]
                    '''
                    
                    df_idx_list = remain_data.index[batch_idx*batch_size : (batch_idx + 1) * batch_size]     
                    text_batch_features = np.array([list(i.values()) for i in passages_features])
                    feature_names = passages_features[0].keys()
                    feature_names = ['feature_' + k for k in feature_names]
                    assert text_batch_features.shape == (len(one_batch), len(feature_names))
                    assert text_batch_features.shape == (len(df_idx_list), len(feature_names))
                    remain_data.loc[df_idx_list, feature_names] = text_batch_features
            
            else:
                for idx, row in tqdm(remain_data.iterrows(), total=len(remain_data)):
                    text = row[text_column] # refer to docstring
                    text_features = dict()
                    for one_extractor in feature_extractors:
                        feature_dict = one_extractor.extract(text)
                        for feature_key in feature_dict.keys():
                            text_features[feature_key] = feature_dict[feature_key]
                    remain_data.loc[idx, ['feature_' + k for k in text_features.keys()]] = text_features.values()
            
            if cache:
                logger.info(f"Saving '{remain_id}' to cache...")
                save_to_cache_pickle("preapply", f"features_{remain_id}", "features_"+remain_id+".pkl", remain_data)
            
            features_applied[remain_id] = remain_data

        # features_applied == list of DFs where feature columns have been appended.
        final_list = None
        if cache:
            final_list = [features_applied[cache_id] for cache_id in cache_ids]
        else:
            final_list = [features_applied[idx] for idx in range(len(df_list))]

        feature_list = [col for col in final_list[0].columns.values if col.startswith("feature_")]
        feature_list = feature_list + extra_normalize_columns
        if target_column not in feature_list and target_column is not None:
            # if target_column not given in extra_normalize_columns
            # kind of just an error check since we do want the target_column in the feature_list.
            feature_list.append(target_column)
        
        pre_blacklist_features = []
        if normalize:
            self.normalize = True
            self.feature_maxes = dict()
            self.feature_mins = dict()
            for feature in feature_list:
                self.feature_maxes[feature] = []
                self.feature_mins[feature] = []
            
            for df in final_list:
                for feature in feature_list:
                    one_feat_list = df[feature].values
                    self.feature_maxes[feature].append(one_feat_list.max())
                    self.feature_mins[feature].append(one_feat_list.min())
            
            
            for feature in feature_list:
                fmax = max(self.feature_maxes[feature])
                fmin = min(self.feature_mins[feature])

                if fmax == fmin:
                    pre_blacklist_features.append(feature)
                else:
                    scalar = (fmax - fmin)

                    fmax = fmax + 0.1*scalar
                    fmin = fmin - 0.1*scalar
                    # to avoid 0.00 and 1.00 values for target & room for extra space at inference time.

                    self.feature_maxes[feature] = fmax
                    self.feature_mins[feature] = fmin

                    for df_idx, df in enumerate(final_list):
                        final_list[df_idx][feature] = df[feature].apply(lambda x: (x-fmin) / (fmax - fmin))
       
        # fix nan values.
        blacklist_features = []
        for feature in feature_list:
            if feature in pre_blacklist_features:
                blacklist_features.append(feature)
            else:
                for df in final_list:
                    nan_count = df[feature].isnull().sum()
                    na_count = df[feature].isna().sum()
                    
                    if nan_count > 0 or na_count > 0:
                        blacklist_features.append(feature)

        logger.info(f"Columns excluded for NaN values (count={len(blacklist_features)}): {blacklist_features}")
        for df_idx, df in enumerate(final_list):
            final_list[df_idx] = df.drop(columns=blacklist_features)

        # For code readability's sake, I'm listing all the instance var settings here:
        # when we load the model again, we'll find these values.
        self.target_column = target_column
        self.normalize = normalize

        if self.normalize:
            self.feature_maxes = self.feature_maxes
            self.feature_mins = self.feature_mins
            if self.target_column is not None:
                self.target_max = self.feature_maxes[target_column]
                self.target_min = self.feature_mins[target_column]
        self.blacklist_features = blacklist_features
        self.features = [col for col in final_list[0].columns.tolist() if col.startswith("feature_")]
        self.feature_extractors = feature_extractors
        return final_list

    def pre_apply_embeddings(
        self, 
        df_list: List[pd.DataFrame],
        text_column: str = "excerpt",
        batch_size: int = 1
    ) -> List[np.ndarray]:
        """Applies st_model upon the list of DataFrames given in df_list. Returns the corresponding
        list of ST embeddings.
        
        Args: 
            df_list (List[pd.DataFrame]) list of pd.DataFrames to calculate embeddings for.
            batch_size (int): Size of the batches to pass into the model for encoding.
        Returns:
            embed_list (List[np.ndarray]): list of corresponding np.ndarray st embeddings to the df_list.
        """
        model = self.st_model
    
        collect_embeddings = []
        for data in df_list:
            data_len = len(data)
            iter_count = data_len // batch_size
            leftovers_count = data_len - (iter_count * batch_size)
            
            one_df_embeddings = []
            for one_pass in trange(iter_count):
                one_batch = data.iloc[one_pass*batch_size : (one_pass + 1) * batch_size]
                one_batch_sentences = one_batch[text_column].to_list()

                assert type(one_batch_sentences) == list
                assert type(one_batch_sentences[0]) == str

                embeddings = model.encode(one_batch_sentences).tolist()
                one_df_embeddings = one_df_embeddings + embeddings
                    
            leftovers = data.iloc[(iter_count * batch_size):][text_column].to_list()
            leftover_embeddings = model.encode(leftovers).tolist()
            one_df_embeddings = one_df_embeddings + leftover_embeddings
            collect_embeddings.append(np.array(one_df_embeddings))

        self.embedding_size = collect_embeddings[0].shape[-1]
        return np.array(collect_embeddings)

    def get_feature_list(self, dataframe: pd.DataFrame = None) -> List[str]:
        if dataframe is not None:
            return [col for col in dataframe.columns.values if col.startswith("feature_")]
        return self.features

    def get_embedding_size(self, embedding: np.ndarray = None) -> int:
        if embedding is not None:
            return embedding.shape[-1]
        if hasattr(self, "embedding_size"):
            return self.embedding_size
        else:
            try:
                dims = self.st_model.get_sentence_embedding_dimension()
                if dims is not None:
                    return dims
            except:
                pass
        raise Exception("Can't get embedding size without having loaded an st_model corresponding to it.")
