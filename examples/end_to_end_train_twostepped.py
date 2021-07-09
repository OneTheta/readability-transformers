import os
import argparse
import shutil
import argparse
import pandas as pd
import torch
from torch import nn
from torch.nn import MSELoss
from easydict import EasyDict as edict

from sentence_transformers import SentenceTransformer, models, evaluation
from readability_transformers import ReadabilityTransformer
from readability_transformers.models import TwoStepTRFPrediction
from readability_transformers.readers import PairwiseDataReader, FeaturesDataReader
from readability_transformers.dataset import CommonLitDataset
from readability_transformers.features import LingFeatExtractor

from readability_transformers.losses import DenormRMSELoss, RankingMSELoss, WeightedRankingMSELoss


def end_to_end_train(model, output_path, device, n_layers=7, weighted_mse_min_weight=0.7, lr=5e-6, weight_decay=0.01, gradient_accumulation=2):
    # model must already have st_model loaded.
    # this is an example of end-to-end training after having two-step trained the st_model.
    assert model.st_model is not None
    rp_options = {
        "model_path": output_path,
        "n_layers": n_layers,
        "ratios": (0.95, 0.05, 0.0),
        "ratio_cache_labels": ("train", "valid", "test"),
        "device": device,
        "st_embed_batch_size": 32,
        "weighted_mse_min_weight": weighted_mse_min_weight,
        "batch_size": 64,
        "epochs": 1600,
        "lr": lr,
        "weight_decay": weight_decay,
        "evaluation_steps": 5000,
        "gradient_accumulation": gradient_accumulation,
    }
    rp_options = edict(rp_options)
    filename = os.path.realpath(__file__)
    shutil.copyfile(filename, os.path.join(rp_options.model_path, "rp_train.py"))
    lf = LingFeatExtractor()

    commonlit_data = CommonLitDataset("train")
    train_df, valid_df, _ = commonlit_data.split_train_valid_test(ratios=rp_options.ratios, ratio_cache_labels=rp_options.ratio_cache_labels)

    train_df, valid_df = model.pre_apply_features(
        df_list=[train_df, valid_df], 
        feature_extractors=[lf],
        text_column="excerpt",
        cache=True,
        cache_ids=["train_lf", "valid_lf"],
        normalize=True,
        extra_normalize_columns=["target"]
    )

    # All are fine
    features = model.get_feature_list() # since it saves the info doing pre_apply_features()
    embedding_size = model.get_embedding_size()

    train_reader = FeaturesDataReader(train_df, features, text_column="excerpt", target_column="target")
    valid_reader = FeaturesDataReader(valid_df, features, "excerpt", "target")
    
    model.init_rp_model(
        rp_model_name="res-drop-fully-connected",
        features=features,
        embedding_size=embedding_size,
        n_layers=rp_options.n_layers
    )

    # Very Common-Lit specific:\
    stderr_stats = train_reader.get_standard_err_stats()
    rp_train_metric = WeightedRankingMSELoss(
        alpha=0.6, 
        min_err=stderr_stats["min"],
        max_err=stderr_stats["max"],
        min_weight=rp_options.weighted_mse_min_weight
    )

    denormMSE = DenormRMSELoss(model)
    denormMSE = DenormRMSELoss(target_max=model.feature_maxes["target"], target_min=model.feature_mins["target"])

    rp_evaluation_metrics = [MSELoss(reduction="mean"), denormMSE]

    print(model)
    model.fit(
        output_path=rp_options.model_path,
        train_reader=train_reader,
        valid_reader=valid_reader,
        train_metric=rp_train_metric,
        evaluation_metrics=rp_evaluation_metrics,
        batch_size=rp_options.batch_size,
        epochs=rp_options.epochs,
        lr=rp_options.lr,
        weight_decay=rp_options.weight_decay,
        evaluation_steps=rp_options.evaluation_steps,
        save_best_model=True,
        show_progress_bar=False,
        gradient_accumulation=rp_options.gradient_accumulation,
        device=device
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--count', default=2, help='count', type=int)
    parser.add_argument('--device', default="cuda:0")
    parser.add_argument('--n_layers', default=7, type=int)
    parser.add_argument('--weighted_mse_min_weight', default=0.7, type=float)
    parser.add_argument('--lr', default=5e-6, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--gradient_accumulation', default=2, type=int)

    args = parser.parse_args()

    ablation_path = "checkpoints/dump/end_to_end"

    count = args.count
    device = args.device
    double = True

    output_path = os.path.join(ablation_path, f"prediction_{count}")
    os.mkdir(output_path)

   
    model = ReadabilityTransformer(
        "checkpoints/ablations/eval_twostep/pred_twostep/pred_twostep_2",
        new_checkpoint_path=output_path,
        device=device,
        double=double
    )

    end_to_end_train(model, output_path, args.device, int(args.n_layers), float(args.weighted_mse_min_weight), float(args.lr), float(args.weight_decay), int(args.gradient_accumulation))
    with open(os.path.join(output_path, "hyp.txt"), "w") as f:
        f.write(str(args))



