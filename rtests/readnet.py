import os
import pickle
import argparse
import shutil
import argparse
import pandas as pd
import torch
from torch import nn
from torch.nn import MSELoss
from easydict import EasyDict as edict
from typing import List, Union

from readability_transformers import ReadNet, models, readers, dataset, losses
from readability_transformers.features import SentenceLingFeatExtractor, DocumentLingFeatExtractor

def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))

def readnet():
    device = "cuda"
    double = True
    count = 2
    batch_size = 64
    lr=1e-5
    gradient_accumulation=2

    # 0. Initialize ReadabilityTransformers
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("checkpoints/readnet", exist_ok=True)
    output_path = f'checkpoints/readnet/prediction_{4}'

    model = ReadNet(
        model_name="roberta-base",
        new_checkpoint_path=output_path,
        device=device,
        double=double
    )

    train_df = pd.read_csv("readnet_temp/train.csv")
    valid_df = pd.read_csv("readnet_temp/valid.csv")

    train_df = train_df.iloc[:30]
    valid_df = valid_df.iloc[:30]

    sentence_lf = SentenceLingFeatExtractor()
    document_lf = DocumentLingFeatExtractor()

    train_df, valid_df = model.pre_apply_features(
        df_list=[train_df, valid_df], 
        feature_extractors=[document_lf],
        text_column="excerpt",
        target_column="target",
        cache=True,
        cache_ids=["train_doc_lf", "valid_doc_lf"],
        normalize=True,
        extra_normalize_columns=["target"]
    )

    train_sent_feats, valid_sent_feats = model.pre_apply_sentence_features(
        df_list=[train_df, valid_df],
        feature_extractors=[sentence_lf],
        text_column="excerpt",
        cache=True,
        cache_ids=["train_sent_lf", "valid_sent_lf"],
        normalize=True
    )

    doc_features = model.get_feature_list()
    sent_features = model.get_sentence_feature_list()
    tokenizer = model.tokenizer

    train_reader = readers.DeepFeaturesDataReader(
        train_df,
        doc_features,
        train_sent_feats,
        tokenizer,
        double,
        text_column="excerpt",
        target_column="target"
    )

    train_loader = train_reader.get_dataloader(batch_size=1)

    for batch in train_loader:
        tokenized_inputs = batch["inputs"]

        input_ids = tokenized_inputs["input_ids"]
        attention_mask = tokenized_inputs["attention_mask"]
        sent_feats = batch["sentence_features"]
        doc_feats = batch["document_features"]

        # same batch size, same # sents per document
        assert input_ids.size()[:2] == sent_feats.size()[:2]
        assert attention_mask.size()[:2] == sent_feats.size()[:2]

        for doc_idx in range(input_ids.size(0)):
            one_doc_input_ids = input_ids[doc_idx]
            one_doc_sent_feats = sent_feats[doc_idx]

            for sent_idx in range(one_doc_input_ids.size(0)):
                one_sent_input_ids = one_doc_input_ids[sent_idx]
                one_sent_feats = one_doc_sent_feats[sent_idx]

                decoded = tokenizer.decode(one_sent_input_ids)
                replaced = decoded.replace("<s></s>", "").replace("<pad>", "")

                """
                Test the following 1-1 correspondence:
                
                Sentence features of sentence A is all zeros 
                <=>
                sentence A is "broken", which is defined as having too little actual words

                which would also show that the multidimensional sentence features array does
                correspond to the correct tokenized input text (as in we don't have sentence features
                mapping to incorrect sentences)
                """
                if one_sent_feats.sum() == 0:
                    assert len(replaced.split(" ")) < 3
                if len(replaced.split(" ")) < 3:
                    assert one_sent_feats.sum() == 0
            
                
                
                  
                    

if __name__ == "__main__":
    readnet()