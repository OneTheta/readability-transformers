import pickle
from typing import List

import nltk
import torch
import pandas as pd
nltk.download("punkt")
from nltk import sent_tokenize
from torch.utils.data import Dataset, DataLoader

from .DataReader import DataReader

def stack_tokenized(batch):
    one_sample = batch[0]
    keys = one_sample.keys()
    full_batch = dict()
    for key in keys:
        full_batch[key] = []
    
    for one_doc in batch:
        for key in one_doc:
            if isinstance(one_doc[key], torch.Tensor):
                full_batch[key].append(one_doc[key])

    for key in full_batch.keys():
        full_batch[key] = torch.stack(full_batch[key], dim=0)
    
    return full_batch


def format_sentence_features(sentence_features, sentence_limit):
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



class DeepFeaturesDataset(Dataset):
    def __init__(self, texts, sentence_features, document_features, tokenizer, sentence_limit, token_upper_limit, targets=None, standard_err=None, double=True):

        torch_dtype = torch.double if double else torch.float32

        self.targets = targets
        self.standard_err = standard_err
        if self.targets is not None:
            self.targets = torch.tensor(self.targets, dtype=torch_dtype)
        if self.standard_err is not None:
            self.standard_err = torch.tensor(self.standard_err, dtype=torch_dtype)
        
        sentence_features = format_sentence_features(sentence_features, sentence_limit)
        # document_features = format_document_features(document_features)
        self.sentence_features = torch.tensor(sentence_features, dtype=torch_dtype)
        self.document_features = torch.tensor(document_features, dtype=torch_dtype)
        self.data = []

        sentence_splitted = [sent_tokenize(text)[:sentence_limit] for text in texts]
        for document in sentence_splitted:
            n_sents = len(document)
            document = document + [""] * (sentence_limit - n_sents)
            assert len(document) == sentence_limit

            tokenized = tokenizer(
                document,
                max_length=token_upper_limit,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            # tokenized = batch_to_device(tokenized, device)
            self.data.append(tokenized)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return_dict = {
            "inputs": self.data[idx],
            "sentence_features": self.sentence_features[idx],
            "document_features": self.document_features[idx]
        }
        if self.targets is not None:
            return_dict["target"] = self.targets[idx]
        if self.standard_err is not None:
            return_dict["standard_err"] = self.standard_err[idx]

        return return_dict


def dataloader_collate_fn(batch):
    tokenized_collect = []
    sent_feat_collect = []
    doc_feat_collect = []
    targets_collect = []
    for one in batch:
        tokenized_collect.append(one["inputs"])
        sent_feat_collect.append(one["sentence_features"])
        doc_feat_collect.append(one["document_features"])
        if "target" in one.keys():
            targets_collect.append(one["target"])
    
    new_batch = dict()
    new_batch["inputs"] = stack_tokenized(tokenized_collect)
    new_batch["sentence_features"] = torch.stack(sent_feat_collect, dim=0)
    new_batch["document_features"] = torch.stack(doc_feat_collect, dim=0)
    if "target" in batch[0].keys():
        new_batch["target"] = torch.stack(targets_collect, dim=0)
    return new_batch



class DeepFeaturesDataReader(DataReader):
    # @TODO this is a rushed implementation just to get this going. Change this code later.
    # it doesnt fit the current DataReader framework because
    # 1. it requires a tokenizer input as parameter, 2. doesnt use a pandas df for loading features
    # this will require a complete change of pre_apply_features in ReadNet

    def __init__(
        self,
        dataframe: pd.DataFrame,
        features: List[str],
        sentence_features,
        tokenizer, 
        double,
        sentence_limit: int,
        token_upper_limit: int,
        text_column: str = "excerpt",
        target_column: str = "target",
    ):
        super(DeepFeaturesDataReader, self).__init__()
        
        self.dataframe = dataframe
        self.features = features
        self.sentence_features = sentence_features
        self.document_features = dataframe[features].values
        self.tokenizer = tokenizer
        self.double = double
        self.text_column = text_column
        self.target_column = target_column
    
        if "standard_error" in dataframe.columns.values:
            self.standard_err = list(dataframe["standard_error"].values)
        self.texts = list(dataframe[text_column].values)
        self.targets = list(dataframe[target_column].values)
        self.dataset = DeepFeaturesDataset(
            self.texts, 
            self.sentence_features, 
            self.document_features, 
            self.tokenizer, 
            targets=self.targets, 
            standard_err=None, 
            sentence_limit=sentence_limit,
            token_upper_limit=token_upper_limit,
            double=double
        )

    def get_dataset(self):
        return self.dataset

    def get_dataloader(self, batch_size: int, shuffle:bool = True):
        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=dataloader_collate_fn
        )
        return dataloader

    def get_standard_err_stats(self):
        standard_error = self.standard_err
        return {
            "min": standard_error.min(),
            "max": standard_error.max()
        }

    def __len__(self):
        return len(self.texts)
        

