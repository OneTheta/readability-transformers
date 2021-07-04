import os
import shutil
import pandas as pd
from torch.nn import MSELoss
from easydict import EasyDict as edict

from sentence_transformers import SentenceTransformer, models, losses, evaluation
from readability_transformers import ReadabilityTransformer
from readability_transformers.readers import PairwiseDataReader, PredictionDataReader
from readability_transformers.dataset import CommonLitDataset
from readability_transformers.losses import WeightedRankingMSELoss



def st_train_manual_st():
    sbert_model = "roberta-base"

    name = sbert_model.replace("checkpoints/", "").replace("/0_SentenceTransformer", "").replace("/", "_")
    output_path = "checkpoints/save_"+name
    if "save_" in name:
        output_path = sbert_model.replace("/0_SentenceTransformer", "")
        # see that this saves the model in place.

    st_options = {
        "st_model_name": sbert_model,
        "ratios": (0.95, 0.05, 0.0),
        "ratio_cache_labels": ("train", "valid", "test"),
        "sample_k": 5,
        "batch_size": 18,
        "max_seq_length": 256,
        "warmup_steps": 100,
        "device": "cuda:0",
        "write_csv": True,
        "epochs": 10,
        "warmup_steps": 100,
        "lr": 1e-5,
        "output_path": output_path,
        "evaluation_steps": 3000,
        "show_progress_bar": False
    }
    st_options = edict(st_options)

    commonlit_data = CommonLitDataset("train")
    train_df, valid_df, _ = commonlit_data.split_train_valid_test(ratios=st_options.ratios, ratio_cache_labels=st_options.ratio_cache_labels)

    train_df_indices = train_df.index
    valid_df_indices = valid_df.index
    
    inter = set(train_df_indices).intersection(set(valid_df_indices))
    assert len(inter) == 0

    train_readers = []
    for epoch in range(st_options.epochs):
        train_reader = PairwiseDataReader(df=train_df, cache_name=f"train_grid_{epoch}", sample_k=st_options.sample_k)
        train_readers.append(train_reader)
    valid_reader = PairwiseDataReader(df=valid_df, cache_name="valid_grid", sample_k=st_options.sample_k)


    model = ReadabilityTransformer(
        model_path=st_options.output_path,
        device=st_options.device,
        double=True
    )

    if model.st_model is None:
        print("Initalizing new st_model since the provided checkpoint did not have one.")
        model.init_st_model(
            st_model_name=st_options.st_model_name,
            max_seq_length=st_options.max_seq_length
        )

    # shutil.copyfile("examples/3_st_trf_search.py", os.path.join(st_options.output_path, "st_trf_search.py"))

    model.st_fit(
        train_readers=train_readers,
        valid_reader=valid_reader,
        batch_size=st_options.batch_size,
        write_csv=st_options.write_csv,
        epochs=1,
        warmup_steps=st_options.warmup_steps,
        lr=st_options.lr,
        output_path=st_options.output_path,
        evaluation_steps=st_options.evaluation_steps,
        show_progress_bar=st_options.show_progress_bar
    )

if __name__ == "__main__":
    st_train_manual_st()