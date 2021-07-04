import os
import shutil
import pandas as pd
import torch
from torch import nn
from torch.nn import MSELoss
from easydict import EasyDict as edict

from sentence_transformers import SentenceTransformer, models, losses, evaluation
from readability_transformers import ReadabilityTransformer
from readability_transformers.models import TwoStepFCPrediction
from readability_transformers.readers import PairwiseDataReader, PredictionDataReader
from readability_transformers.dataset import CommonLitDataset

epochs = 5
max_seq_length = 256
out_features = 128
device = "cuda:0"
double = False

if double:
    torch.set_default_dtype(torch.float64)
else:
    torch.set_default_dtype(torch.float32)

st_word_embedding_model = models.Transformer("bert-base-cased", max_seq_length=max_seq_length)
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


twostep_predict = TwoStepFCPrediction(
    embedding_size=out_features,
    sentence_transformer=st_model,
    n_layers=3,
    h_size=516,
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

for epoch in range(epochs):
    train_reader = train_readers[epoch]
    twostep_predict.fit(
        train_reader=train_reader,
        valid_reader=valid_reader,
        train_metric=nn.MSELoss(),
        evaluation_metrics=[nn.MSELoss()],
        batch_size=32,
        write_csv=True,
        epochs=epochs,
        gradient_accumulation=2,
        warmup_steps=1000,
        lr=5e-5,
        weight_decay=0.001,
        output_path="checkpoints/twostep_fc_1",
        evaluation_steps=2000,
        show_progress_bar=False,
        save_best_model=True
    )
