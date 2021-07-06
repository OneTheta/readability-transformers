import os
import pandas as pd
from torch.nn import MSELoss
from easydict import EasyDict as edict
from readability_transformers import ReadabilityTransformer
from readability_transformers.readers import PairwiseDataReader, PredictionDataReader
from readability_transformers.dataset import CommonLitDataset
from readability_transformers.losses import WeightedRankingMSELoss


TEST_CSV_PATH = 'readability_transformers/dataset/data/test.csv'
OUTPUT_CSV_PATH = './'

def get_test_df():
    commonlit_data = CommonLitDataset("test")
    return commonlit_data.data

def inference_on_dataset():
    model = ReadabilityTransformer(
        "commonlit-lf-twostep-1",
        device="cpu",
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



if __name__ == "__main__":
    inference_on_dataset()