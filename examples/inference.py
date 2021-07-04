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

def inference_on_dataset():
    model = ReadabilityTransformer(
        model_path="export_checkpoints/pred_twostep_1",
        device="cpu",
        double=True
    )

    test_df = pd.read_csv(TEST_CSV_PATH)

    ids = test_df["id"].values
    passages = test_df["excerpt"].values

    predictions = model.predict(passages, batch_size=2)


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

def inference():
    options = {
        "model_checkpoint": "checkpoints/save",
        "device": "cuda:1"
    }
    options = edict(options)

    model = ReadabilityTransformer(
        model_path=options.model_path,
        device=options.device
    )

    texts = [
        "BERT out-of-the-box is not the best option for this task, as the run-time in your setup scales with the number of sentences in your corpus. I.e., if you have 10,000 sentences/articles in your corpus, you need to classify 10k pairs with BERT, which is rather slow."
    ]
    predictions = model(texts, batch_size=1)
    print(predictions)


if __name__ == "__main__":
    inference_on_dataset()