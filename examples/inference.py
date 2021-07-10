import os
import numpy as np
import pandas as pd
from torch.nn import MSELoss
from easydict import EasyDict as edict
from readability_transformers import ReadabilityTransformer
from readability_transformers.readers import PairwiseDataReader, PredictionDataReader
from readability_transformers.dataset import CommonLitDataset
from readability_transformers.losses import WeightedRankingMSELoss
from readability_transformers.file_utils import load_from_cache_pickle


TEST_CSV_PATH = 'readability_transformers/dataset/data/test.csv'
OUTPUT_CSV_PATH = './'

def get_test_df():
    commonlit_data = CommonLitDataset("test")
    return commonlit_data.data

def inference_on_dataset():
    model = ReadabilityTransformer(
        "checkpoints/dump/prediction_2",
        device="cuda:0",
        double=True
    )

    test_df = get_test_df()

    ids = test_df["id"].values
    passages = test_df["excerpt"].values

    predictions = model.predict(passages, batch_size=3)


    # making commmonlit submission
    submission = []
    for one in zip(ids, predictions.tolist()):
        one_id, one_prediction = one
        one_submit = {
            "id": one_id,
            "target": one_prediction
        }
        submission.append(one_submit)
    
    submission = pd.DataFrame(submission)

    submission_path = os.path.join(OUTPUT_CSV_PATH, "submission.csv")
    submission.to_csv(submission_path, index=False)





def inference_on_valid_split():
    model = ReadabilityTransformer(
        "checkpoints/dump/prediction_2",
        device="cuda:0",
        double=True
    )

    valid_df = load_from_cache_pickle("preapply", "features_valid_v1.5_lf_trf")
    train_df = load_from_cache_pickle("preapply", "features_train_v1.5_lf_trf")

    valid_idx = set(list(valid_df.index))
    train_idx = set(list(train_df.index))
    intersect = train_idx.intersection(valid_idx)
    print("intersect", intersect)

    print(valid_df.sample(n=3).values)
    print(train_df.sample(n=3).values)
    
    # valid_easiest_idx = valid_df["target"].idxmax()
    # print("valid_easiest_idx", valid_easiest_idx)
    # print(valid_df.loc[valid_easiest_idx].values)
    # train_easiest_idx = train_df["target"].idxmax()
    # print(train_df.loc[train_easiest_idx].values)

    result_df =  pd.read_csv("checkpoints/dump/prediction_2/1_Prediction/evaluation_results.csv")
    idx = result_df["mean"].idxmin()
    recorded_rmse = result_df.loc[idx]["DenormRMSELoss"]


    ids = valid_df["id"].values
    passages = valid_df["excerpt"].values
    targets = valid_df["target"].values

    predictions = model.predict(passages, batch_size=3)

    actual_rmse_1 = np.sqrt(np.mean((predictions.cpu().tolist()[:64]-targets[:64])**2))
    actual_rmse_2 = np.sqrt(np.mean((predictions.cpu().tolist()[64:]-targets[64:])**2))
    actual_rmse = np.mean([actual_rmse_1, actual_rmse_2])
    print("actual", actual_rmse)
    print("recorded_rmse", recorded_rmse)

# 0.3783097477302331

    # making commmonlit submission
    submission = []
    for one in zip(ids, predictions.tolist(), targets):
        one_id, one_prediction, true_target = one
        one_submit = {
            "id": one_id,
            "target": one_prediction,
            "true_target": true_target
        }
        submission.append(one_submit)
    
    submission = pd.DataFrame(submission)

    submission_path = os.path.join(OUTPUT_CSV_PATH, "submission.csv")
    submission.to_csv(submission_path, index=False)




if __name__ == "__main__":
    inference_on_valid_split()