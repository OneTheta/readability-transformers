# Copyright 2021 One Theta. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
from typing import List
from loguru import logger

from readability_transformers.file_utils import CachedDataset
from readability_transformers.dataset import Dataset

DATASET_ID = "commonlit"
DATASET_ZIP_URL = 'https://1theta-readability-transformers.s3.us-east-2.amazonaws.com/commonlit.tar.gz'
DATAFILES_META = {
    "train": "data/train.csv",
    "test": "data/test.csv",
    "sample": "data/sample_submission.csv"
}

class CommonLitDataset(Dataset):
    def __init__(self, label: str):
        """Loads the CommonLit dataset.

        Args:
            label (str): CommonLit dataset consists of the "train" dataset and the 
                         "test" dataset used for Kaggle evaluation. 
            cache (bool): if set to True, caches the train-valid-test split when called. Usually we train the
                SentenceTransformer first then train the ReadabilityPrediction model. We usually want to use
                the same splitted train-valid-test throughout the whole process (unless doing some sort of ablation component study).
        Returns:
            data (pd.DataFrame): .csv -> pd.DataFrame instance of the dataset.
        """
        super().__init__(DATASET_ID, DATASET_ZIP_URL, DATAFILES_META)
        
        self.cached_dataset = CachedDataset(DATASET_ID, DATASET_ZIP_URL, DATAFILES_META)

        data_url = self.cached_dataset.get_datafile_from_id(label)
        print("data_url", data_url)
        self.data = pd.read_csv(data_url)

        for idx, row in self.data.iterrows():
            self.data.loc[idx, "excerpt"] = row["excerpt"].replace("\n", " ").replace("\t", " ").replace("  ", " ")
    