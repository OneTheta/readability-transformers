import os
import shutil
import pandas as pd
from torch.nn import MSELoss


from easydict import EasyDict as edict
from readability_transformers import ReadabilityTransformer
from readability_transformers.readers import PairwiseDataReader, PredictionDataReader
from readability_transformers.dataset import CommonLitDataset
from readability_transformers.losses import WeightedRankingMSELoss, DenormRMSELoss
from readability_transformers.features import TRUNAJODExtractor

def rp_train():
    rp_options = {
        "model_path": "checkpoints/twostep_trf_3",
        "n_layers": 7,
        "ratios": (0.92, 0.08, 0.0),
        "device": "cuda:0",
        "st_embed_batch_size": 32,
        "weighted_mse_min_weight": 0.7,
        "batch_size": 64,
        "epochs": 1000,
        "lr": 5e-6,
        "weight_decay": 0.01,
        "evaluation_steps": 5000,
        "gradient_accumulation": 2,
    }
    rp_options = edict(rp_options)

    shutil.copyfile("examples/2_rp_train.py", os.path.join(rp_options.model_path, "rp_train.py"))

    model = ReadabilityTransformer(
        model_path=rp_options.model_path,
        device=rp_options.device,
        double=True
    )

    trunajod = TRUNAJODExtractor()

    commonlit_data = CommonLitDataset("train", cache=True)
    train_df, valid_df, _ = commonlit_data.split_train_valid_test(ratios=rp_options.ratios)

    train_df, valid_df = model.pre_apply_features(
        df_list=[train_df, valid_df], 
        feature_extractors=[trunajod],
        text_column="excerpt",
        cache=True,
        cache_ids=["train_trunajod", "valid_trunajod"],
        normalize=True,
        extra_normalize_columns=["target"]
    )
    train_embed, valid_embed = model.pre_apply_embeddings(
        df_list=[train_df, valid_df],
        text_column="excerpt",
        batch_size=rp_options.st_embed_batch_size
    )

    # All are fine
    features = model.get_feature_list(train_df)
    features = model.get_feature_list() # since it saves the info doing pre_apply_features()
    features = model.features
    
    embedding_size = model.get_embedding_size(train_embed)
    embedding_size = model.get_embedding_size()
    embedding_size = model.embedding_size

    train_reader = PredictionDataReader(train_df, train_embed, features, text_column="excerpt", target_column="target")
    valid_reader = PredictionDataReader(valid_df, valid_embed, features, "excerpt", "target")
    
    model.init_rp_model(
        rp_model_name="fully-connected",
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
    rp_train()