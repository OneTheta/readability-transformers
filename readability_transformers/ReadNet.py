import os
import pickle
from typing import List, Union

import nltk
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
nltk.download('punkt')
from tqdm import tqdm
from loguru import logger
from tqdm.autonotebook import trange
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer, models

from .readers import DeepFeaturesDataReader
from .features import FeatureBase
from .models import ReadNetModel
from .file_utils import (
    load_from_cache_pickle, 
    save_to_cache_pickle, 
    path_to_rt_model_cache, 
    download_rt_model
)
from .ReadabilityTransformer import ReadabilityTransformer
from .readers.DeepFeaturesDataReader import stack_tokenized



def _extract_sentence_features(remain_data: List[str], feature_extractors, sentence_limit, token_lower_limit):
    sentence_level_features = []
    for text in tqdm(remain_data, total=len(remain_data)):        
        features_per_sentence = []
        for one_sentence in sent_tokenize(text)[:sentence_limit]:
            tokens_with_words = [word for word in word_tokenize(one_sentence) if True in [k.isalpha() for k in word]]
            if len(tokens_with_words) >= token_lower_limit:
                try:
                    text_features = dict()
                    for one_extractor in feature_extractors:
                        feature_dict = one_extractor.extract(text)
                        for feature_key in feature_dict.keys():
                            text_features["sent_feature_" + feature_key] = feature_dict[feature_key]
                    
                    features_per_sentence.append(text_features)
                except:
                    print(f"Encountered unexpected error extracting sentence features for sentence: '{one_sentence}'. Empty features added.")
                    features_per_sentence.append(dict())
            else:
                features_per_sentence.append(dict())
        sentence_level_features.append(features_per_sentence)
    
    return sentence_level_features

def _extract_document_features(passages, feature_extractors, batch_size=1):
    feature_dict_array = []
    for batch_idx in range(len(passages) // batch_size + 1):
        if batch_idx * batch_size == len(passages):
            break
        one_batch = passages[batch_idx*batch_size : (batch_idx + 1)*batch_size]

        feature_dict_list_collect = []
        for one_extractor in feature_extractors:
            feature_dict_list = one_extractor.extract_in_batches(one_batch)
            feature_dict_list_collect.append(feature_dict_list)

            assert len(feature_dict_list) == len(one_batch)

        for passage_idx in range(len(one_batch)):
            complete_features_per_passage = dict()
            for feature_dict_list in feature_dict_list_collect:
                passage_feature_dict = feature_dict_list[passage_idx]
                for key in passage_feature_dict.keys():
                    complete_features_per_passage["feature_"+key]  = passage_feature_dict[key]
            feature_dict_array.append(complete_features_per_passage)
    return feature_dict_array
        
def _feature_dict_to_matrix_normalize(feature_dict_array, features, normalize=False, feature_maxes=None, feature_mins=None, normalize_func=None):
    pre_norm_feature_matrix = []
    feature_matrix = []
    for passage_idx, feat_dict in enumerate(feature_dict_array):
        pre_norm_passage_feature_num_array = []
        passage_feature_num_array = []
        for feat_idx, one_feature in enumerate(features): # preserve order of original model input
            feat_value = feat_dict[one_feature]
            pre_norm_passage_feature_num_array.append(feat_value)
            if normalize:
                max_val = feature_maxes[one_feature]
                min_val = feature_mins[one_feature]
                feat_value = normalize_func(feat_value, max_val, min_val)
            passage_feature_num_array.append(feat_value)

        pre_norm_feature_matrix.append(pre_norm_passage_feature_num_array)
        feature_matrix.append(passage_feature_num_array)
    return pre_norm_feature_matrix, feature_matrix


def _format_sentence_features(sentence_features, sentence_limit):
    for doc_idx, document_features in enumerate(sentence_features): # document_features == (# sents, sent_feat_dict)
        for sent_idx, sent_feat_dict in enumerate(document_features): 
            sentence_features[doc_idx][sent_idx] = list(sentence_features[doc_idx][sent_idx].values())
    n_docs = len(sentence_features)
    n_features = len(sentence_features[0][0])
    # we want to end up with [#docs, 50 sentences, # features]
    new_sent_feats = []
    collect_n_sents = []
    for one_doc in sentence_features:
        n_sents = len(one_doc)
        collect_n_sents.append(n_sents)
        remaining_sents = sentence_limit - n_sents
        one_doc = one_doc + [[0] * n_features]*remaining_sents

        new_one_doc = []
        for one_sent in one_doc:
            if len(one_sent) < n_features:
                one_sent = [0] * n_features
            new_one_doc.append(one_sent)

        new_sent_feats.append(new_one_doc)

    return new_sent_feats



def _sent_feature_dict_to_matrix_normalize(sent_feature_dict_array, features, sentence_limit, normalize=False, feature_maxes=None, feature_mins=None, normalize_func=None):
    n_features = len(features)
    prenormed_sent_feats = sent_feature_dict_array
    for doc_idx, document_features in enumerate(sent_feature_dict_array): # document_features == (# sents, sent_feat_dict)
        for sent_idx, sent_feat_dict in enumerate(document_features): 
            if len(sent_feat_dict.keys()) > 0:
                one_sent_collect = []
                prenorm_sent_collect = []
                for feature in features:
                    this_sent_this_feature = sent_feat_dict[feature]
                    prenorm_sent_collect.append(this_sent_this_feature)
                    if normalize:
                        normalized = normalize_func(this_sent_this_feature, feature_maxes[feature], feature_mins[feature])
                        one_sent_collect.append(normalized)
                    else:
                        one_sent_collect.append(this_sent_this_feature)
                sent_feature_dict_array[doc_idx][sent_idx] = one_sent_collect
            else:
                sent_feature_dict_array[doc_idx][sent_idx] = [0] * n_features

        remaining_sents = sentence_limit - len(document_features)
        document_features = document_features + [[0] * n_features]*remaining_sents
        sent_feature_dict_array[doc_idx] = document_features


    return prenormed_sent_feats, sent_feature_dict_array
            



    for feature in feature_list:
        global_feature_collect = [] # all instances of this feature across the full dataset
        for dataset_features in final_list: # dataset_features == (# docs, # sents, sent_feat_dict)
            for document_features in dataset_features: # document_features == (# sents, sent_feat_dict)
                for sent_feat_dict in document_features: 
                    if len(sent_feat_dict.keys()) > 0:
                        # if not empty dictionary of features, which happens when it's a super small or broken sentence
                        global_feature_collect.append(sent_feat_dict[feature])
        fmax = max(global_feature_collect)
        fmin = min(global_feature_collect)

        if fmax == fmin:
            pre_blacklist_features.append(feature)
        else:
            scalar = (fmax - fmin)

            fmax = fmax + 0.02*scalar
            fmin = fmin - 0.02*scalar
            # to avoid 0.00 and 1.00 values for target & room for extra space at inference time.

            self.sent_feature_maxes[feature] = fmax
            self.sent_feature_mins[feature] = fmin

def get_example_feat_list(final_list):
    for dataset in final_list:
        for document in dataset:
            for sentence in document:
                if len(sentence.keys()) > 0:
                    return sentence
    return None

def batch_to_device(batch, target_device):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(target_device)
    return batch


class ReadNet(ReadabilityTransformer):
    def __init__(
        self,
        model_name: str,
        device: str,
        double: bool,
        sentence_limit: int = None,
        token_lower_limit: int = None,
        token_upper_limit: int = None,
        new_checkpoint_path: str = None,
        **kwargs
    ):
        super(ReadNet, self).__init__(
            model_name=model_name, 
            device=device, 
            double=double, 
            new_checkpoint_path=new_checkpoint_path
        )
        required_rn_keys = ["d_model", "n_heads", "n_blocks", "n_feats_sent", "n_feats_doc"]
        
        if hasattr(self, "model") and self.model is not None:
            logger.info("Loaded ReadNet model.")
            # alre4ady loaded
            # if kwargs is not None:
            #     raise Exception("Supplied model initialization configs when model is already loaded.")
        else:
            if None in [sentence_limit, token_lower_limit, token_upper_limit]:
                raise Exception(f"Must supply all of [sentence_limit, token_lower_limit, token_upper_limit] when initializing new ReadNet.")
            
            self.sentence_limit = sentence_limit
            self.token_lower_limit = token_lower_limit
            self.token_upper_limit = token_upper_limit


            readnet_config_keys = kwargs.keys()
            if len(readnet_config_keys) > 0:
                if False in [key in required_rn_keys for key in readnet_config_keys]:
                    raise Exception(f"Initializing ReadNet model requires at least these keys: {required_rn_keys}")
                else:
                    # if not already loaded, at least the st_model has already been loaded with param:model_name
                    logger.info(f"Initializing readnet with args: {kwargs}")
                    self.init_readnet(
                        st_model=self.st_model,
                        device=device,
                        **kwargs
                    )
                    
            # or dont actually initialize readnet model at all
            print("Initialized ReadNet class but have not initialize the model. Make sure model.init_readnet() is called later.")
    

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
                
                setattr(self, key, attribute)
            self.model = model
            self.tokenizer = model.sent_block.blocks.tokenizer
            return model
        else:
            logger.info("Readnet model not found. Creating new...")
            os.makedirs(model_path, exist_ok=True)
            return None
        

        # Readnet has a different model checkpoint structure.
    
    def init_readnet(self, st_model=None, d_model=768, n_heads=6, n_blocks=6, n_feats_sent=145, n_feats_doc=223, device="cuda"):
        if st_model is None:
            if hasattr(self, "st_model") and self.st_model is not None:
                st_model = self.st_model
            else:
                raise Exception("Tried to initialize readnet model when either 1. model is already loaded 2. st_model is not yet loaded")

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
        torch_dtype = torch.double if self.double else torch.float32
        self.model = self.model.to(self.device, torch_dtype)
        return self.model

        

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

                tokenized_inputs = batch_to_device(batch["inputs"], self.device)
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
            
                break

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
                tokenized_inputs = batch_to_device(batch["inputs"], self.device)
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

    def predict(self, passages: List[str], batch_size: int=1):
        out_verbose = self.predict_verbose(passages, batch_size)
        return out_verbose[0]

    def predict_verbose(self, passages: List[str], batch_size: int = 1):
        """This is an inference function without gradients. For a forward pass of the model with gradients
        that lets you train the model, refer to self.forward(). 

        Args:
            passages (List[str]): list of texts to predict readability scores on.
            batch_size (int): batch_size to pass the input data through the model.
        """
        passages = self.basic_preprocess_text(passages)
        self.model.to(self.device)
        self.model.eval()
        device = self.device
        double = self.double

        """
        1. Passages -> {doc_features, sent_features}
        """
        doc_features = self.features
        doc_feature_extractors = self.feature_extractors
        doc_blacklist_features = self.blacklist_features

        sent_features = self.sent_features
        sent_feature_extractors = self.sent_feature_extractors
        sent_blacklist_features = self.sent_blacklist_features
        sentence_limit = self.sentence_limit
        token_lower_limit = self.token_lower_limit
        token_upper_limit = self.token_upper_limit

        if self.normalize:
            normalize = lambda value, max_val, min_val: max(min((value - min_val) / (max_val - min_val), 0.999), 0.001)
            denormalize = lambda value, max_val, min_val: (value * (max_val - min_val)) + min_val

        """
        1.a. Passages -> doc_features
        """

        doc_feature_dict_array = _extract_document_features(passages, doc_feature_extractors, batch_size)
        pre_norm_feature_matrix, doc_feature_matrix = _feature_dict_to_matrix_normalize(
            doc_feature_dict_array,
            doc_features,
            self.normalize,
            self.feature_maxes if self.normalize else None,
            self.feature_mins if self.normalize else None,
            normalize_func=normalize if self.normalize else None
        )
        assert len(doc_feature_matrix) == len(passages)
        assert len(doc_feature_matrix[0]) == len(doc_features)

        """
        1.b. Passages -> sent_features
        """
        sent_feature_dict_array = _extract_sentence_features(passages, sent_feature_extractors, sentence_limit, token_lower_limit)
        pre_norm_sent_feature_matrix, sent_feature_matrix = _sent_feature_dict_to_matrix_normalize(
            sent_feature_dict_array,
            sent_features,
            sentence_limit,
            self.normalize,
            self.sent_feature_maxes if self.normalize else None,
            self.sent_feature_mins if self.normalize else None,
            normalize_func=normalize if self.normalize else None
        )

        assert len(sent_feature_matrix) == len(passages)
        assert len(sent_feature_matrix[0][0]) == len(sent_features)

        """
        1.c. Passages -> Tokenized inputs
        """
        tokenizer = self.tokenizer
        tokenized_collect = []
        sentence_splitted = [sent_tokenize(text)[:sentence_limit] for text in passages]
        for document in sentence_splitted:
            n_sents = len(document)
            document = document + [""] * (sentence_limit - n_sents)
            tokenized = tokenizer(
                document,
                max_length=token_upper_limit,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            tokenized = batch_to_device(tokenized, device)
            tokenized_collect.append(tokenized)
        
        """
        1.d. These input features => ready for model pass
        """
        torch_dtype = torch.double if double else torch.float32

        sentence_features = torch.tensor(sent_feature_matrix, dtype=torch_dtype).to(device)
        document_features = torch.tensor(doc_feature_matrix, dtype=torch_dtype).to(device)       
        tokenized_inputs = tokenized_collect

        """
        2. Model pass
        """
        predictions_collect = []
        with torch.no_grad():
            n_batches = len(passages) // batch_size
            for full_batch_idx in range((len(passages) // batch_size) + 1):
                if (full_batch_idx * batch_size == len(passages)):
                    break
                start_idx = full_batch_idx * batch_size
                end_idx = (full_batch_idx + 1) * batch_size

                batch_inputs = stack_tokenized(tokenized_inputs[start_idx : end_idx])
                batch_sentence_features = sentence_features[start_idx : end_idx]
                batch_document_features = document_features[start_idx : end_idx]

                one_prediction = self.model.forward(batch_inputs, batch_sentence_features, batch_document_features)
                predictions_collect.append(one_prediction)
        
        if len(predictions_collect) > 1:
            predictions = torch.stack(predictions_collect[:-1], dim=0).flatten(end_dim=1)
            predictions = torch.cat((predictions, predictions_collect[-1]), dim=0) # because last batch may not be full
        else:
            predictions = predictions_collect[0]
    
        """
        3. Denormalize Prediction
        """
        if hasattr(self, "target_max") and self.target_max is not None:
            target_max = self.target_max
            target_min = self.target_min
            
            predictions = denormalize(predictions, target_max, target_min)

        return (
            predictions,
            doc_feature_matrix,
            sent_feature_matrix,
            pre_norm_feature_matrix,
            pre_norm_sent_feature_matrix
        )

    def pre_apply_sentence_features(
        self, 
        df_list: List[pd.DataFrame],
        feature_extractors: List[FeatureBase],
        batch_size: int = None,
        text_column: str = "excerpt",
        cache: bool = False,
        cache_ids: List[str] = None,
        normalize=False,
        extra_normalize_columns: List[str] = None
    ) -> np.ndarray: 
        """Applies the feature classes on the dataframes in df_list such that new columns are appended for each feature.
        ReadabilityTransformers is able to do such feature extraction on-the-fly as well. But this can be inefficient when
        training multiple epochs, since the same feature extraction will occur repeatedly. This function allows one feature
        extraction pass to re-use over and over again.

        Args:
            df_list (List[pd.DataFrame]): List of Pandas DataFrames that contain text to apply the features to, the column name of which
                is specified by the text_column argument.
            feature_classes (List[FeatureBase]): List of Feature classes to use to apply features to the text.
            text_column: Name of the column in the dataframe that contains the text in interest. Defaults to "excerpt".
            cache (boolf): Whether to cache the dataframes generated after applying the features. Feature extraction can be a costly
                process that might serve well to do only once per dataset. Defaults to False.
            cache_ids (List[str]): Must be same length as df_list. The ID value for the cached feature dataset generated from the datasets
                in df_list. Defaults to None.
            normalize (bool): Normalize the feature values to 0 to 1. Defaults to True.
            extra_normalize_columns (List[str]): Name of existing columns to apply normalization to as well, since otherwise
                normalization will only be applied to the features extracted from this function. 
        
        Returns:
            feature_dfs (np.ndarray): A numpy matrix of dimensions: (# of documents, # of sentences, # of features)
        """
        features_applied = dict() 

        token_lower_limit = self.token_lower_limit
        sentence_limit = self.sentence_limit
        self.sent_normalize = normalize

        if cache_ids:
            if len(cache_ids) != len(df_list):
                raise Exception("The list of labels must match the df_list parameter.")
        if cache:
            if cache_ids is None:
                raise Exception("Cache set to true but no label for the cache.")
            else:
                # Load *what we can* from cache.
                for cache_id in cache_ids:
                    cache_loaded = load_from_cache_pickle("preapply", f"sent_features_{cache_id}")
                    if cache_loaded is not None:
                        logger.info(f"Found & loading cache ID: {cache_id}...")
                        features_applied[cache_id] = cache_loaded

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

        # extract features
        for remain_id, remain_data in zip(remaining_ids, remaining):
            logger.info(f"Extracting features for cache_id={remain_id}")

            if batch_size is not None:
                raise Exception("Currently does not support batched sentence feature extraction. Fix this in future versions.")
                # remain_data = _extract_features_batches(remain_id, remain_data, feature_extractors, text_column)
            else:
                remain_data_text_list = list(remain_data[text_column].values())
                remain_data = _extract_sentence_features(
                    remain_data_text_list, 
                    feature_extractors, 
                    sentence_limit=sentence_limit,
                    token_lower_limit=token_lower_limit
                )
            features_applied[remain_id] = remain_data

            if cache:
                logger.info(f"Saving '{remain_id}' to cache...")
                save_to_cache_pickle("preapply", f"sent_features_{remain_id}", "sent_features_"+remain_id+".pkl", remain_data)

        # features_applied == list of: (# documents, # sentences, feature dictionary per sentence)
        if cache:
            final_list = [features_applied[cache_id] for cache_id in cache_ids]
        else:
            final_list = [features_applied[idx] for idx in range(len(df_list))]

        first_full_feat_list = None

        feature_list = [key for key in get_example_feat_list(final_list) if key.startswith("sent_feature")]
        feature_list = feature_list + extra_normalize_columns
        
        pre_blacklist_features = []
        if self.sent_normalize:
            self.sent_feature_maxes = dict()
            self.sent_feature_mins = dict()
            for feature in feature_list:
                self.feature_maxes[feature] = []
                self.feature_mins[feature] = []
            
            # final_list == (# datasets, # documents, # sentences, sent_feat_dict)
            
            for feature in feature_list:
                global_feature_collect = [] # all instances of this feature across the full dataset
                for dataset_features in final_list: # dataset_features == (# docs, # sents, sent_feat_dict)
                    for document_features in dataset_features: # document_features == (# sents, sent_feat_dict)
                        for sent_feat_dict in document_features: 
                            if len(sent_feat_dict.keys()) > 0:
                                # if not empty dictionary of features, which happens when it's a super small or broken sentence
                                global_feature_collect.append(sent_feat_dict[feature])
                fmax = max(global_feature_collect)
                fmin = min(global_feature_collect)

                if fmax == fmin:
                    pre_blacklist_features.append(feature)
                else:
                    scalar = (fmax - fmin)

                    fmax = fmax + 0.02*scalar
                    fmin = fmin - 0.02*scalar
                    # to avoid 0.00 and 1.00 values for target & room for extra space at inference time.
                    self.sent_feature_maxes[feature] = fmax
                    self.sent_feature_mins[feature] = fmin

            # Finished populating self.sent_feature_maxes, self.sent_feature_mins
            for dataset_idx, dataset_features in enumerate(final_list): # dataset_features == (# docs, # sents, sent_feat_dict)
                for doc_idx, document_features in enumerate(dataset_features): # document_features == (# sents, sent_feat_dict)
                    for sent_idx, sent_feat_dict in enumerate(document_features): 
                        if len(sent_feat_dict.keys()) > 0:
                            for feature in list(set(feature_list) - set(pre_blacklist_features)):
                                normalize = lambda x: (x - self.sent_feature_mins[feature]) / (self.sent_feature_maxes[feature] - self.sent_feature_mins[feature])
                                final_list[dataset_idx][doc_idx][sent_idx][feature] = normalize(sent_feat_dict[feature])
                        
       
        # fix nan values.
        blacklist_features = []
        # for feature in feature_list:
        #     if feature in pre_blacklist_features:
        #         blacklist_features.append(feature)
        #     else:
        #         for df in final_list:
        #             nan_count = df[feature].isnull().sum()
        #             na_count = df[feature].isna().sum()
                    
        #             if nan_count > 0 or na_count > 0:
        #                 blacklist_features.append(feature)
        blacklist_features = pre_blacklist_features
        logger.info(f"Columns excluded for NaN values (count={len(blacklist_features)}): {blacklist_features}")
        for dataset_idx, dataset_features in enumerate(final_list): # dataset_features == (# docs, # sents, sent_feat_dict)
            for doc_idx, document_features in enumerate(dataset_features): # document_features == (# sents, sent_feat_dict)
                for sent_idx, sent_feat_dict in enumerate(document_features): 
                    for feature in blacklist_features:
                        final_list[dataset_idx][doc_idx][sent_idx].pop(feature, None)
                    

        # For code readability's sake, I'm listing all the instance var settings here:
        # when we load the model again, we'll find these values.
        self.sent_normalize = self.sent_normalize
        if self.sent_normalize:
            self.sent_feature_maxes = self.sent_feature_maxes
            self.sent_feature_mins = self.sent_feature_mins
  
        self.sent_blacklist_features = blacklist_features
        self.sent_features = [key for key in get_example_feat_list(final_list) if key.startswith("sent_feature")]
        self.sent_feature_extractors = feature_extractors

        if cache:
            if self.sent_normalize:
                norm_metadata = {
                    "sent_feature_maxes": self.sent_feature_maxes,
                    "sent_feature_mins": self.sent_feature_mins
                }
                df = pd.DataFrame(norm_metadata)
                metadata_cache_name = "_".join(cache_ids)
                save_to_cache_pickle("preapply", f"norm_sent_metadata_{metadata_cache_name}", f"norm_sent_metadata_{metadata_cache_name}.pkl", df)

        return final_list

    def get_sentence_feature_list(self):
        return self.sent_features

    def get_config(self):
        feature_extractors = []
        for feature_extractor in self.feature_extractors:
            if hasattr(feature_extractor, "trf_model"):
                del feature_extractor.trf_model
            if hasattr(feature_extractor, "tokenizer"):
                del feature_extractor.tokenizer
            feature_extractors.append(feature_extractor)
        
        sent_feature_extractors = []
        for feature_extractor in self.sent_feature_extractors:
            if hasattr(feature_extractor, "trf_model"):
                del feature_extractor.trf_model
            if hasattr(feature_extractor, "tokenizer"):
                del feature_extractor.tokenizer
            feature_extractors.append(feature_extractor)
        
        config = {
            "normalize": self.normalize,
            "features": self.features,
            "feature_extractors": feature_extractors,
            "blacklist_features": self.blacklist_features,
            "sent_normalize": self.sent_normalize,
            "sent_features": self.sent_features,
            "sent_feature_extractors": sent_feature_extractors,
            "sent_blacklist_features": self.sent_blacklist_features,
            "sentence_limit": self.sentence_limit,
            "token_lower_limit": self.token_lower_limit,
            "token_upper_limit": self.token_upper_limit
        }
        if self.normalize:
            config["feature_maxes"] = self.feature_maxes
            config["feature_mins"] = self.feature_mins
        if hasattr(self, "target_max"):
            config["target_max"] = self.target_max
            config["target_min"] = self.target_min
        if self.sent_normalize:
            config["sent_feature_maxes"] = self.sent_feature_maxes
            config["sent_feature_mins"] = self.sent_feature_mins
        
        
        return config

                