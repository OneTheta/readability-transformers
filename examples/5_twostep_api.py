import os
import shutil
import pandas as pd
import torch
from torch import nn
from torch.nn import MSELoss
from easydict import EasyDict as edict

from sentence_transformers import SentenceTransformer, models, losses, evaluation
from readability_transformers import ReadabilityTransformer
from readability_transformers.readers import PairwiseDataReader, PredictionDataReader
from readability_transformers.dataset import CommonLitDataset

epochs = 5
max_seq_length = 256
out_features = 128
device = "cuda:0"
double = False
output_path = "checkpoints/twostep_fc_1"

model = ReadabilityTransformer(
    model_path="checkpoints/twostep_fc_1",
    device=device,
    double=double
) 


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

model.init_st_model(st_model)

# GET DATASET
commonlit_data = CommonLitDataset("train", cache=True)
train_df, valid_df, _ = commonlit_data.split_train_valid_test(ratios=(0.95, 0.05, 0.0))

train_readers = []
for epoch in range(epochs):
    train_reader = PairwiseDataReader(df=train_df, cache_name=f"train_grid_{epoch}", sample_k=100)
    train_readers.append(train_reader)
valid_reader = PairwiseDataReader(df=valid_df, cache_name="valid_grid", sample_k=100)

# GET TWOSTEP TRAINER
trf_twostep_options = {
    "embedding_size": out_features,
    "n_layers": 2,
    "h_size": 256,
    "encoder_layers": 3,
    "trf_heads": 12,
    "trf_dropout": 0.1
}
twostep_predict = model.get_twostep_trainer("trf", **trf_twostep_options)

fc_twostep_options = {
    "embedding_size": out_features,
    "n_layers": 4,
    "h_size": 256
}
twostep_predict = model.get_twostep_trainer("fc", **fc_twostep_options)


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
        output_path=output_path,
        evaluation_steps=2000,
        show_progress_bar=False,
        save_best_model=True
    )
