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
from readability_transformers.dataset import CEFRDataset
from readability_transformers.features import LingFeatExtractor, TransformersLogitsExtractor
from readability_transformers.losses import DenormRMSELoss, RankingMSELoss, WeightedRankingMSELoss

from readability_transformers import models

LABELS = ["A1", "A2", "B1", "B2", "C1", "C2"]

def train():
    device = "cuda:0"

  
    model = ReadabilityTransformer(
        "roberta-base",
        new_checkpoint_path="checkpoints/dump/classification_2",
        device=device,
        double=True
    )

    cefr_data = CEFRDataset("train")
    train_df, valid_df = cefr_data.split_train_valid_test(ratios=(0.95, 0.05), ratio_cache_labels=("train_2", "valid_2"))

    lf = LingFeatExtractor()
    trf = TransformersLogitsExtractor(device=device)

    train_df, valid_df = model.pre_apply_features(
        df_list=[train_df, valid_df], 
        feature_extractors=[lf, trf],
        text_column="text",
        cache=True,
        cache_ids=["train_lf_trf_cefr_full_2", "valid_lf_trf_cefr_full_2"],
        normalize=True
    )

    print(train_df.iloc[1].values)
    print(train_df.iloc[2].values)
    features = model.get_feature_list() # since it saves the info doing pre_apply_features()
    embedding_size = model.get_embedding_size()

    train_reader = FeaturesDataReader(train_df, features, text_column="text", target_column="label", classification=True, labels=LABELS)
    valid_reader = FeaturesDataReader(valid_df, features, text_column="text", target_column="label", classification=True, labels=LABELS)

    # readability_prediction = models.ResFCClassification(
    #     input_size=len(features) + embedding_size,
    #     n_layers=3,
    #     h_size=512,
    #     dropout=0.2,
    #     n_labels=len(LABELS),
    #     double=True
    # )

    # model.init_rp_model(
    #     rp_model_name="ResFCClassification",
    #     features=features,
    #     embedding_size=embedding_size,
    #     n_layers=3,
    #     h_size=512,
    #     dropout=0.2,
    #     n_labels=len(LABELS)
    # )

    input_size = len(features) + embedding_size
    n_labels = len(LABELS)
    n_layers = 6
    h_size = 512
    dropout = 0.2
    double = True
    
    rp_model = models.ResFCClassification(input_size, n_layers, h_size, dropout=dropout, n_labels=n_labels, double=double)
    model.init_rp_model(rp_model)
    
    # @TODO implement max margin loss
    rp_train_metric = nn.CrossEntropyLoss()
    rp_evaluation_metrics = [nn.CrossEntropyLoss()]

    model.fit(
        train_reader=train_reader,
        valid_reader=valid_reader,
        train_metric=rp_train_metric,
        evaluation_metrics=rp_evaluation_metrics,
        batch_size=16,
        epochs=500,
        lr=1e-5,
        weight_decay=1e-8,
        evaluation_steps=1000,
        save_best_model=True,
        show_progress_bar=True,
        gradient_accumulation=4,
        device="cuda:0"
    )


def test():
   
    passages = ["I've already been to Japan. It's a really beautiful country and people are very nice. I visited Tokyo, Kyoto, Hiroshima, Osaka and their fantastic temples and I climbed on the Mount Fujiyama. Also, I went to Saint-Pierre et Miquelon in order to work during two years, three years ago. I've ever lived in a such wide and authentic region. Of course, during my journey there, I also visited eastern Canada. Before visiting this place, I've never seen virginia deers, black bears and snow harfangs. I loved it. Finally, I've already lived in Guadeloupe, several islands in Caribean sea. I worked there for six month."]

    predictions = model.predict(passages, batch_size=1)

    print(predictions)

if __name__ == "__main__":
    train()

