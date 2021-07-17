import os
import argparse
import shutil
import argparse
import pandas as pd
import torch
from torch import nn
from torch.nn import MSELoss
from easydict import EasyDict as edict
from typing import List, Union

from readability_transformers import ReadabilityTransformer, models, readers, dataset, losses
from readability_transformers.features import LingFeatExtractor, TransformersLogitsExtractor

def attn_end_to_end(args):
    count = args.count
    device = args.device
    n_layers = args.n_layers
    weighted_mse_min_weight = args.weighted_mse_min_weight
    lr = args.lr
    weight_decay = args.weight_decay
    gradient_accumulation = args.gradient_accumulation
    feature_trf = args.feature_trf
    st_model_checkpoint = args.st_model_checkpoint
    h_size = args.h_size
    dropout = args.dropout
    double = True

    # 0. Initialize ReadabilityTransformers
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("checkpoints/v18_eval_ete", exist_ok=True)
    output_path = f'checkpoints/v18_eval_ete/prediction_{count}'
    model = ReadabilityTransformer(
        st_model_checkpoint,
        new_checkpoint_path=output_path,
        device=device,
        double=double
    )

    # 1. Load dataset & Split
    commonlit_data = dataset.CommonLitDataset("train")
    train_df, valid_df = commonlit_data.stratified_split_data(
        ratios=(0.90, 0.10),
        ratio_cache_labels=("train_strat", "valid_strat")
    )

    # 2. Pre-Apply Features
    # 2.1 Define extractors 
    lf = LingFeatExtractor()
    # trf = TransformersLogitsExtractor(device)
    if feature_trf:
        feature_extractors = [lf, trf]
        cache_ids = ["train_lf_trf", "valid_lf_trf"]
    else:
        feature_extractors = [lf]
        cache_ids = ["train_lf_real", "valid_lf_real"]

    train_df = train_df[:50]
    valid_df = valid_df[:50]
    train_df, valid_df = model.pre_apply_features(
        df_list=[train_df, valid_df], 
        feature_extractors=feature_extractors,
        text_column="excerpt",
        target_column="target",
        cache=True,
        cache_ids=cache_ids,
        normalize=True,
        extra_normalize_columns=["target"]
    )

    features = model.get_feature_list() 
    embedding_size = model.get_embedding_size()

    train_reader = readers.FeaturesDataReader(train_df, features, text_column="excerpt", target_column="target")
    valid_reader = readers.FeaturesDataReader(valid_df, features, text_column="excerpt", target_column="target")

    # Initialize Prediction Model:
    prediction_model = models.AttnFCRegression(
        input_size = len(features) + embedding_size,
        n_layers = n_layers,
        h_size = h_size,
        dropout = dropout,
        double = double
    )
    model.init_rp_model(prediction_model)

    # Very Common-Lit specific:\
    stderr_stats = train_reader.get_standard_err_stats()
    rp_train_metric = losses.WeightedRankingMSELoss(
        alpha=0.5,
        min_err=stderr_stats["min"],
        max_err=stderr_stats["max"],
        min_weight=weighted_mse_min_weight
    )
    denormMSE = losses.DenormRMSELoss(model)
    rp_evaluation_metrics = [MSELoss(reduction="mean"), denormMSE]

    print(model)
    model.fit(
        train_reader=train_reader,
        valid_reader=valid_reader,
        train_metric=rp_train_metric,
        evaluation_metrics=rp_evaluation_metrics,
        batch_size=32,
        epochs=1300,
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
    parser = argparse.ArgumentParser()

    parser.add_argument('--count', default=1, help='count', type=int)
    parser.add_argument('--device', default="cuda:0")
    parser.add_argument('--n_layers', default=5, type=int)
    parser.add_argument('--weighted_mse_min_weight', default=0.7, type=float)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--weight_decay', default=1e-8, type=float)
    parser.add_argument('--gradient_accumulation', default=2, type=int)
    parser.add_argument('--dropout', default=0.1, type=int)
    parser.add_argument('--feature_trf', default=0, type=int)
    parser.add_argument('--h_size', default=512, type=int)
    parser.add_argument('--st_model_checkpoint', default="checkpoints/ablations/eval_twostep/pred_twostep/pred_twostep_2", type=str)

    args = parser.parse_args()

    attn_end_to_end(args)