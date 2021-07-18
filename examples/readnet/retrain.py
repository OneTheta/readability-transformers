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
    model = ReadNet(
        model_name="checkpoints/readnet/prediction_1",
        new_checkpoint_path='checkpoints/readnet/prediction_2',
        device=device,
        double=double
    )

    train_df = pd.read_csv("readnet_temp/train.csv")
    valid_df = pd.read_csv("readnet_temp/valid.csv")

    sentence_lf = SentenceLingFeatExtractor()
    document_lf = DocumentLingFeatExtractor()

    train_df, valid_df = model.pre_apply_features(
        df_list=[train_df, valid_df], 
        feature_extractors=[document_lf],
        text_column="excerpt",
        target_column="target",
        cache=True,
        cache_ids=["train_doc_lf_true", "valid_doc_lf_true"],
        normalize=True,
        extra_normalize_columns=["target"]
    )

    train_sent_feats, valid_sent_feats = model.pre_apply_sentence_features(
        df_list=[train_df, valid_df],
        feature_extractors=[sentence_lf],
        text_column="excerpt",
        cache=True,
        cache_ids=["train_sent_lf_true", "valid_sent_lf_true"],
        normalize=True
    )

    doc_features = model.get_feature_list()
    sent_features = model.get_sentence_feature_list()
    sentence_limit = model.sentence_limit
    token_upper_limit = model.token_upper_limit
    
    tokenizer = model.tokenizer

    train_reader = readers.DeepFeaturesDataReader(
        train_df,
        doc_features,
        train_sent_feats,
        tokenizer,
        double,
        sentence_limit=sentence_limit,
        token_upper_limit=token_upper_limit,
        text_column="excerpt",
        target_column="target"
    )
    valid_reader = readers.DeepFeaturesDataReader(
        valid_df,
        doc_features,
        valid_sent_feats,
        tokenizer,
        double,
        sentence_limit=sentence_limit,
        token_upper_limit=token_upper_limit,
        text_column="excerpt",
        target_column="target"
    )

    rp_train_metric = MSELoss()
    rp_evaluation_metrics = [RMSELoss, MSELoss()]

    model.fit(
        train_reader=train_reader,
        valid_reader=valid_reader,
        train_metric=rp_train_metric,
        evaluation_metrics=rp_evaluation_metrics,
        batch_size=batch_size,
        epochs=10,
        lr=lr,
        weight_decay=1e-8,
        evaluation_steps=2048,
        save_best_model=True,
        show_progress_bar=True,
        gradient_accumulation=gradient_accumulation,
        device=device,
        freeze_trf_steps=2048,
        start_saving_from_step=10000
    )

if __name__ == "__main__":
    readnet()