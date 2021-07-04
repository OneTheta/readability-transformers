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
from readability_transformers.losses import RankingMSELoss
epochs = 5
max_seq_length = 256
out_features = 256
device = "cuda:0"
double = False

st_word_embedding_model = models.Transformer("bert-base-uncased", max_seq_length=max_seq_length)
st_pooling_model = models.Pooling(st_word_embedding_model.get_word_embedding_dimension())
st_dense_model = models.Dense(
    in_features=st_pooling_model.get_sentence_embedding_dimension(), 
    out_features=out_features,
    activation_function=nn.Tanh()
)
st_model = SentenceTransformer(
    modules=[st_word_embedding_model, st_pooling_model, st_dense_model],
    device=device
)

# st_model = SentenceTransformer("checkpoints/twostep_trf_3", device=device)

twostep_trf = TwoStepTRFPrediction(
    embedding_size=out_features,
    sentence_transformer=st_model,
    n_layers=5,
    h_size=256,
    encoder_layers=4,
    trf_heads=8,
    trf_dropout=0.1,
    double=double,
    device=device
)

commonlit_data = CommonLitDataset("train", cache=True)
train_df, valid_df, _ = commonlit_data.split_train_valid_test(ratios=(0.95, 0.05, 0.0))


train_readers = []
for epoch in range(epochs):
    train_reader = PairwiseDataReader(df=train_df, cache_name=f"train_grid_{epoch}", sample_k=100)
    train_readers.append(train_reader)
valid_reader = PairwiseDataReader(df=valid_df, cache_name="valid_grid", sample_k=100)


twostep_trf.fit(
    train_reader=train_readers,
    valid_reader=valid_reader,
    train_metric=RankingMSELoss(alpha=0.6),
    evaluation_metrics=[nn.MSELoss(), RankingMSELoss(alpha=0.6)],
    batch_size=25,
    write_csv=True,
    epochs=5,
    gradient_accumulation=1,
    warmup_steps=100,
    lr=2e-5,
    weight_decay=1e-8,
    output_path="checkpoints/twostep_trf_3",
    evaluation_steps=1000,
    show_progress_bar=False,
    save_best_model=True,
    freeze_trf_steps=1000
)
