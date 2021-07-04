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

rp_options = {
    "n_layers": 6,
    "weighted_mse_min_weight": 0.4,
    "batch_size": 64,
    "epochs": 1000,
    "lr": 1e-4,
    "weight_decay": 0.01,
    "evaluation_steps": 5000,
    "gradient_accumulation": 1,
}
rp_options = edict(rp_options)

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
twostep_trainer = model.get_twostep_trainer("trf", **trf_twostep_options)

fc_twostep_options = {
    "embedding_size": out_features,
    "n_layers": 4,
    "h_size": 256
}
twostep_trainer = model.get_twostep_trainer("fc", **fc_twostep_options)


for epoch in range(epochs):
    train_reader = train_readers[epoch]
    twostep_trainer.fit(
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


# now that we're done training st_model,
trunajod = TRUNAJODExtractor()

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
    batch_size=32
)

extracted_features = model.get_feature_list(train_df)
extracted_features = model.get_feature_list() # since it saves the info doing pre_apply_features()
extracted_features = model.features

embedding_size = model.get_embedding_size(train_embed)
embedding_size = model.get_embedding_size()
embedding_size = model.embedding_size

train_reader = PredictionDataReader(train_df, train_embed, features, text_column="excerpt", target_column="target")
valid_reader = PredictionDataReader(valid_df, valid_embed, features, "excerpt", "target")
    
model.init_rp_model(
    rp_model_name="fully-connected",
    features=extracted_features,
    embedding_size=embedding_size,
    n_layers=rp_options.n_layers
)
 # Very Common-Lit specific:\
stderr_stats = train_reader.get_standard_err_stats()
rp_train_metric = WeightedRankingMSELoss(
    alpha=2.0, 
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
    st_embed_batch_size=32,
    batch_size=rp_options.batch_size,
    epochs=rp_options.epochs,
    lr=rp_options.lr,
    weight_decay=rp_options.weight_decay,
    evaluation_steps=rp_options.evaluation_steps,
    save_best_model=True,
    show_progress_bar=False,
    gradient_accumulation=rp_options.gradient_accumulation
)