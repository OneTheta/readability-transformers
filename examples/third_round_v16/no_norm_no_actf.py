import os
import argparse
import shutil
import argparse
import pandas as pd
import torch
from torch import nn
from torch.nn import MSELoss
from easydict import EasyDict as edict

from readability_transformers import ReadabilityTransformer
from readability_transformers.models import TwoStepTRFPrediction
from readability_transformers.readers import PairwiseDataReader, PredictionDataReader
from readability_transformers.dataset import CommonLitDataset
from readability_transformers.features import LingFeatExtractor, TransformersLogitsExtractor

from readability_transformers.losses import DenormRMSELoss, RankingMSELoss, WeightedRankingMSELoss

from readability_transformers import reset_cache

def rp_train(model, output_path, device, n_layers=7, weighted_mse_min_weight=0.7, lr=5e-6, weight_decay=0.01, gradient_accumulation=2):
    # model must already have st_model loaded.
    # by taking out rp_train like this, we make sure the rp training process is completely
    # constant over these tests.
    normalize = True
    
    # reset_cache()

    assert model.st_model is not None
    rp_options = {
        "model_path": output_path,
        "n_layers": n_layers,
        "ratios": (0.92, 0.08),
        "ratio_cache_labels": ("train_new_k", "valid_new_k"),
        "device": device,
        "st_embed_batch_size": 32,
        "weighted_mse_min_weight": weighted_mse_min_weight,
        "batch_size": 64,
        "epochs": 400,
        "lr": lr,
        "weight_decay": weight_decay,
        "evaluation_steps": 5000,
        "gradient_accumulation": gradient_accumulation,
    }
    rp_options = edict(rp_options)
    filename = os.path.realpath(__file__)
    shutil.copyfile(filename, os.path.join(rp_options.model_path, "rp_train.py"))


    lf = LingFeatExtractor()
    # trf = TransformersLogitsExtractor(device=device)

    commonlit_data = CommonLitDataset("train")
    train_df, valid_df = commonlit_data.split_train_valid_test(ratios=rp_options.ratios, ratio_cache_labels=rp_options.ratio_cache_labels)
    print(train_df.iloc[1].values)
    print(train_df.iloc[2].values)

    if normalize:
        cache_ids = ["train_v16_norm_lf", "valid_v16_norm_lf"]
    else:
        cache_ids = ["train_v16_no_norm_lf", "valid_v16_no_norm_lf"]
    train_df, valid_df = model.pre_apply_features(
        df_list=[train_df, valid_df], 
        feature_extractors=[lf],
        batch_size=20,
        text_column="excerpt",
        target_column="target",
        cache=True,
        cache_ids=cache_ids,
        normalize=normalize,
    )
    
    
    train_embed, valid_embed = model.pre_apply_embeddings(
        df_list=[train_df, valid_df],
        text_column="excerpt",
        batch_size=rp_options.st_embed_batch_size
    )

    # All are fine
    features = model.get_feature_list() # since it saves the info doing pre_apply_features()
    embedding_size = model.get_embedding_size()

    train_reader = PredictionDataReader(train_df, train_embed, features, text_column="excerpt", target_column="target")
    valid_reader = PredictionDataReader(valid_df, valid_embed, features, "excerpt", "target")
    
    model.init_rp_model(
        "ResFCRegression",
        features=features,
        embedding_size=embedding_size,
        n_layers=7,
        h_size=512,
        dropout=0.2,
    )

    # Very Common-Lit specific:\
    stderr_stats = train_reader.get_standard_err_stats()
    rp_train_metric = WeightedRankingMSELoss(
        alpha=0.6, 
        min_err=stderr_stats["min"],
        max_err=stderr_stats["max"],
        min_weight=rp_options.weighted_mse_min_weight
    )
    # rp_train_metric = MSELoss(reduction="mean")

    denormMSE = DenormRMSELoss(model)
    denormMSE = DenormRMSELoss(target_max=model.feature_maxes["target"], target_min=model.feature_mins["target"])

    rp_evaluation_metrics = [denormMSE]

    model.rp_fit(
        train_reader=train_reader,
        valid_reader=valid_reader,
        train_metric=rp_train_metric,
        evaluation_metrics=rp_evaluation_metrics,
        st_embed_batch_size=rp_options.st_embed_batch_size,
        batch_size=rp_options.batch_size,
        epochs=rp_options.epochs,
        lr=rp_options.lr,
        weight_decay=rp_options.weight_decay,
        evaluation_steps=rp_options.evaluation_steps,
        save_best_model=True,
        show_progress_bar=False,
        gradient_accumulation=rp_options.gradient_accumulation
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--count', default=2, help='count', type=int)
    parser.add_argument('--device', default="cuda:0")
    parser.add_argument('--n_layers', default=3, type=int)
    parser.add_argument('--weighted_mse_min_weight', default=0.7, type=float)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--weight_decay', default=1e-8, type=float)
    parser.add_argument('--gradient_accumulation', default=2, type=int)

    args = parser.parse_args()

    ablation_path = "checkpoints/dump"

    count = args.count
    device = args.device
    double = True

    output_path = os.path.join(ablation_path, f"v16_regression_{6}")
   
    model = ReadabilityTransformer(
        "checkpoints/ablations/eval_twostep/pred_twostep/pred_twostep_2",
        new_checkpoint_path=output_path,
        device=device,
        double=double
    )

    rp_train(model, output_path, args.device, int(args.n_layers), float(args.weighted_mse_min_weight), float(args.lr), float(args.weight_decay), int(args.gradient_accumulation))
    with open(os.path.join(output_path, "hyp.txt"), "w") as f:
        f.write(str(args))



