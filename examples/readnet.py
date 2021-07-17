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
from readability_transformers.features import LingFeatExtractor

def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))

def readnet():
    device = "cuda"
    double = True
    count = 1
    batch_size = 64
    lr=1e-5
    gradient_accumulation=2

    # 0. Initialize ReadabilityTransformers
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("checkpoints/readnet", exist_ok=True)
    output_path = f'checkpoints/readnet/prediction_{count}'
    model = ReadNet(
        model_name="roberta-base",
        new_checkpoint_path=output_path,
        device=device,
        double=double,
        d_model = 512,
        n_heads = 8,
        n_blocks = 6,
        n_feats_sent = 145,
        n_feats_doc = 169
    )

    train_df = pd.read_csv("readnet_temp/train.csv")
    valid_df = pd.read_csv("readnet_temp/valid.csv")

    train_sent_feats = pickle.load(open("readnet_temp/train_sentence_level_features.pkl", "rb"))
    valid_sent_feats = pickle.load(open("readnet_temp/valid_sentence_level_features.pkl", "rb"))

    train_doc_feats = pickle.load(open("readnet_temp/train_document_level_features.pkl", "rb"))
    valid_doc_feats = pickle.load(open("readnet_temp/valid_document_level_features.pkl", "rb"))

    tokenizer = model.tokenizer

    train_reader = readers.DeepFeaturesDataReader(
        train_df,
        train_sent_feats,
        train_doc_feats,
        tokenizer,
        batch_size,
        device,
        double
    )
    valid_reader = readers.DeepFeaturesDataReader(
        valid_df,
        valid_sent_feats,
        valid_doc_feats,
        tokenizer,
        batch_size,
        device,
        double
    )

    # stderr_stats = train_reader.get_standard_err_stats()
    # rp_train_metric = losses.WeightedRankingMSELoss(
    #     alpha=0.5,
    #     min_err=stderr_stats["min"],
    #     max_err=stderr_stats["max"],
    #     min_weight=weighted_mse_min_weight
    # )
    rp_train_metric = MSELoss()
    
    rp_evaluation_metrics = [RMSELoss, MSELoss()]

    print(model)
    model.fit(
        train_reader=train_reader,
        valid_reader=valid_reader,
        train_metric=rp_train_metric,
        evaluation_metrics=rp_evaluation_metrics,
        batch_size=batch_size,
        epochs=5,
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