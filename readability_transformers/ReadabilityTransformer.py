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
import time
import json
import shutil
import pickle
from typing import List, Union
import numpy as np
import pandas as pd
import torch
from torch import nn
from loguru import logger
from tqdm import tqdm
from tqdm.autonotebook import trange


from sentence_transformers import SentenceTransformer, models, losses, evaluation

from .readers import PairwiseDataReader, PredictionDataReader
from .models import Prediction, FCPrediction, TwoStepArchitecture, TwoStepFCPrediction, TwoStepTRFPrediction
from .features import FeatureBase
from .file_utils import load_from_cache_pickle, save_to_cache_pickle

CACHE_DIR = os.path.expanduser("~/.cache/readability-transformers/data")

class ReadabilityTransformer:
    def __init__(
        self,
        model_path: str,
        device: str,
        double: bool,
        new_checkpoint_path: str = None,
    ):
        '''Initialize the ReadabilityTransformer, which first requires an instantiation
        of the SentenceTransformer then an instantiation of the Prediction model.

        Args:
            st_model_name (str): The transformer model to initialize the SentenceTransformer.
                This is the same as the model name in the HuggingFace library list of transformer
                models. e.g. "bert-base-uncased".
            max_seq_length (int): Same usual idea as with HuggingFace transformers. Possible text 
                length the model can have as input.
            device (str): "cpu" or some version of "cuda" to run the torch modules in.
            new_checkpoint_path (str): if set, copies the model at model_path to the new_checkpoint_path
                directory, which will be our new working directory.
            double (bool): Whether this model should use weights that are float32 or float64. 
        '''
        self.model_path = model_path
        self.device = device
        self.double = double
        self.st_model = None
        self.rp_model = None
        self.new_checkpoint_path = new_checkpoint_path

        if self.new_checkpoint_path is not None:
            if os.path.isdir(new_checkpoint_path):
                logger.warning("You are initializing a ReadabilityTransformer model even though one exists already. This will reset the current checkpoint. Are you sure?")
                logger.warning("Waiting 5 seconds just in case...")
                time.sleep(5)
                shutil.rmtree(self.new_checkpoint_path)
            shutil.copytree(self.model_path, new_checkpoint_path)
            self.model_path = new_checkpoint_path

        if self.model_path:
            self.setup_load_checkpoint(self.model_path)

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
        if self.st_path:
            self.st_model.save(self.st_path)

    def init_rp_model(
        self, 
        features: List[str],
        rp_model: Prediction = None,
        rp_model_name: str = None, 
        embedding_size: int = None,
        n_layers: int = None,
    ):

        if self.st_model is None:
            raise Exception("Cannot initialize ReadabilityPrediction model before loading/initializing a SentenceTransformer model.")

        if self.rp_model:
            logger.warning("You are initializing a new Reading Prediction model even though one exists already. This will reset the current checkpoint. Are you sure?")
            logger.warning("Waiting 5 seconds just in case...")
            time.sleep(5)
            shutil.rmtree(self.rp_path)
            os.mkdir(self.rp_path)

        if rp_model is None:
            if None in [rp_model_name, embedding_size, n_layers]:
                raise Exception("Must supply either the RP_model itself or the paramters to create one.")


            self.features = features
            self.embedding_size = embedding_size
            feature_count = len(self.features)
            
            input_size = feature_count + embedding_size
            if rp_model_name == "fully-connected":
                rp_class = FCPrediction
            else:
                raise Exception(f"Could not find ReadabilityPrediction model {rp_model_name}")

            rp_model = rp_class(input_size, n_layers, h_size=256, double=self.double)

        self.rp_model = rp_model.to(self.device)
        self.rp_model.features_in_order = features


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
        config = {
            "normalize": self.normalize,
            "features": self.features,
            "feature_extractors": self.feature_extractors,
            "blacklist_features": self.blacklist_features
        }
        if self.normalize:
            config["feature_maxes"] = self.feature_maxes
            config["feature_mins"] = self.feature_mins
            config["target_max"] = self.target_max
            config["target_min"] = self.target_min


        return config

    def forward(self, passages: List[str], features: List[List[Union[float, np.array, torch.Tensor]]]):
        """Forward pass of the model with gradients whose outputs you can train with a loss function backwards pass.
        No normalization/denormalization efforts are done. This is a pure full RT model pass.

        Args:
            passages: text passages to derive readability score predictiosn from.
            features: each element in the features parameters is an array of feature values for corresponding
                passage in passages.
        """
        
        assert len(features) == len(passages)
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.double if self.double else torch.float32).to(self.device)
        """
        1. Get forward pass from st_model. This code is based on UKPLab/sentence-transformers.
        """

        self.st_model.to(self.device)

        text_features = self.st_model.tokenize(passages)
        for key in text_features:
            text_features[key] = text_features[key].to(self.device)
        
        out_features = self.st_model(text_features)
        embeddings = out_features["sentence_embedding"]
        
            
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
 
        self.rp_model.to(self.device)

        predicted_scores = self.rp_model(rp_inputs)
        return predicted_scores
    

    def predict(self, passages: List[str], batch_size: int = 1):
        """This is an inference function without gradients. For a forward pass of the model with gradients
        that lets you train the model, refer to self.forward(). 

        Args:
            passages (List[str]): list of texts to predict readability scores on.
            batch_size (int): batch_size to pass the input data through the model.
        """
        
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
            normalize = lambda value, max_val, min_val: (value - min_val) / (max_val - min_val)
            denormalize = lambda value, max_val, min_val: (value * (max_val - min_val)) + min_val

        feature_dict_array = []
        for one_passage in passages:
            passage_features = dict()
            for one_extractor in self.feature_extractors:
                one_extraction = one_extractor.extract(one_passage)
                for key in one_extraction.keys():
                    passage_features["feature_"+key] = one_extraction[key]
            feature_dict_array.append(passage_features)
            
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

            for full_batch_idx in range(n_batches):
                start_idx = full_batch_idx * batch_size
                end_idx = (full_batch_idx + 1) * batch_size

                passage_batch = passages[start_idx : end_idx]
                feature_batch = feature_matrix[start_idx : end_idx]

                assert len(passage_batch) == batch_size
                assert len(feature_batch) == batch_size
                
                one_prediction = self.forward(passage_batch, feature_batch)
                predictions_collect.append(one_prediction)

            predictions = torch.stack(predictions_collect, dim=0).flatten(end_dim=1)
            if (n_batches * batch_size) < len(passages):
                # there is extra
                start_idx = n_batches * batch_size 

                leftovers_passage = passages[start_idx:]
                leftovers_features = feature_matrix[start_idx:]

                assert len(leftovers_passage) < batch_size
                assert len(leftovers_features) < batch_size

                leftovers_predictions = self.forward(leftovers_passage, leftovers_features)
                predictions = torch.cat((predictions, leftovers_predictions))
            

        """
        5. Denormalize Prediction
        """
        target_max = self.target_max
        target_min = self.target_min
        
        predictions = denormalize(predictions, target_max, target_min)

        return predictions

    def pre_apply_features(
        self, 
        df_list: List[pd.DataFrame],
        feature_extractors: List[FeatureBase],
        text_column: str = "excerpt",
        target_column: str = "target",
        cache: bool = False,
        cache_ids: List[str] = None,
        normalize=True,
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
            cache (bool): Whether to cache the dataframes generated after applying the features. Feature extraction can be a costly
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
            for idx, row in tqdm(remain_data.iterrows(), total=len(remain_data)):
                text = row[text_column] # refer to docstring
                text_features = dict()
                for one_extractor in feature_extractors:
                    feature_dict = one_extractor.extract(text)
                    for feature_key in feature_dict.keys():
                        value = feature_dict[feature_key]
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
        if target_column not in feature_list:
            # if target_column not given in extra_normalize_columns
            # kind of just an error check since we do want the target_column in the feature_list.
            feature_list.append(target_column)
        
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
                if feature == target_column: # to avoid 0.00 and 1.00 values for target & room for extra space at inference time.
                    fmax = fmax + 0.02
                    fmin = fmin - 0.02
                self.feature_maxes[feature] = fmax
                self.feature_mins[feature] = fmin

                for df_idx, df in enumerate(final_list):
                    final_list[df_idx][feature] = df[feature].apply(lambda x: (x-fmin) / (fmax - fmin))
       
        # fix nan values.
        blacklist_features = []
        for feature in feature_list:
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
        self.normalize = normalize
        self.feature_maxes = self.feature_maxes
        self.feature_mins = self.feature_mins
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
        return self.embedding_size
