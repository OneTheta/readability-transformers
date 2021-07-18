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

    sentence_limit = 28
    token_lower_limit = 3
    token_upper_limit = 50

    # 0. Initialize ReadabilityTransformers
    model = ReadNet(
        model_name="distilroberta-base",
        new_checkpoint_path='checkpoints/readnet/prediction_1',
        device=device,
        double=double,
        sentence_limit=sentence_limit,
        token_lower_limit=token_lower_limit,
        token_upper_limit=token_upper_limit
    )
    # model = ReadNet(
    #     model_name="checkpoints/readnet/prediction_3",
    #     new_checkpoint_path=output_path,
    #     device=device,
    #     double=double
    # )

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

    # train_sent_feats = pickle.load(open("readnet_temp/train_sentence_level_features.pkl", "rb"))
    # valid_sent_feats = pickle.load(open("readnet_temp/valid_sentence_level_features.pkl", "rb"))

    # train_doc_feats = pickle.load(open("readnet_temp/train_document_level_features.pkl", "rb"))
    # valid_doc_feats = pickle.load(open("readnet_temp/valid_document_level_features.pkl", "rb"))

    # stderr_stats = train_reader.get_standard_err_stats()
    # rp_train_metric = losses.WeightedRankingMSELoss(
    #     alpha=0.5,
    #     min_err=stderr_stats["min"],
    #     max_err=stderr_stats["max"],
    #     min_weight=weighted_mse_min_weight
    # )
    rp_train_metric = MSELoss()
    rp_evaluation_metrics = [RMSELoss, MSELoss()]

    readnet_config = {
        "d_model": 512,
        "n_heads": 8,
        "n_blocks": 6,
        "n_feats_sent": len(sent_features),
        "n_feats_doc": len(doc_features)
    }
    model.init_readnet(**readnet_config)

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