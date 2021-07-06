import os
import shutil
import pandas as pd
import torch
from torch import nn
from torch.nn import MSELoss
from easydict import EasyDict as edict

from sentence_transformers import SentenceTransformer, models, evaluation
from readability_transformers import ReadabilityTransformer
from readability_transformers.models import TwoStepTRFPrediction
from readability_transformers.readers import PairwiseDataReader, PredictionDataReader
from readability_transformers.dataset import CommonLitDataset
from readability_transformers.features import LingFeatExtractor

from readability_transformers.losses import DenormRMSELoss, RankingMSELoss, WeightedRankingMSELoss

base_model = "bert-base-uncased"
sample_k = 50
evaluation_steps = 1000
epochs = 4
max_seq_length = 256
out_features = 256
device = "cuda:0"
double = True
lr=1e-5 # @!IMPORTANT lr=4e-5 led to the model outputting only 0s.
weight_decay=1e-8
width_scale=1.1 # widen PairwiseDatareader
freeze_trf_steps=1000



def no_twostep_train(output_path):
    st_model = get_untrained_st_model(base_model)

    model = ReadabilityTransformer(
        model_path=output_path,
        device=device,
        double=double
    )
    model.init_st_model(st_model)

    rp_train(model, output_path)




def get_untrained_st_model(model_name="bert-base-uncased"):
    st_word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
    st_pooling_model = models.Pooling(st_word_embedding_model.get_word_embedding_dimension())
    st_dense_model = models.Dense(
        in_features=st_pooling_model.get_sentence_embedding_dimension(), 
        out_features=out_features,
        activation_function=nn.ReLU()
    )
    st_model = SentenceTransformer(
        modules=[st_word_embedding_model, st_pooling_model, st_dense_model],
        device=device
    )
    return st_model


def rp_train(model, output_path):
    # model must already have st_model loaded.
    # by taking out rp_train like this, we make sure the rp training process is completely
    # constant over these tests.
    assert model.st_model is not None
    rp_options = {
        "model_path": output_path,
        "n_layers": 7,
        "ratios": (0.92, 0.08, 0.0),
        "ratio_cache_labels": ("train", "valid", "test"),
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
    filename = os.path.realpath(__file__)
    shutil.copyfile(filename, os.path.join(rp_options.model_path, "train.py"))
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


no_twostep_path = "checkpoints/ablations/eval_twostep/no_twostep"
output_path = os.path.join(no_twostep_path, f"no_twostep_{1}")
os.mkdir(output_path)

no_twostep_train(output_path)