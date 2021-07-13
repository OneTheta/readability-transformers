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
import re
import pandas as pd
from typing import List, Union
from loguru import logger

from readability_transformers.file_utils import CachedDataset
from readability_transformers.dataset import Dataset

DATASET_ID = "cefr_basic"
DATASET_ZIP_URL = 'https://1theta-readability-transformers.s3.us-east-2.amazonaws.com/basic_proprocessed_cefr.zip'
DATAFILES_META = {
    "train": "basic_proprocessed_cefr.csv"
}

class CEFRDataset(Dataset):
    def __init__(self, label: str):
        """Loads the CEFR-Basic dataset.

        Args:
            label (str): CEFR Dataset only has "train" label.
            cache (bool): if set to True, caches the train-valid-test split when called. Usually we train the
                SentenceTransformer first then train the ReadabilityPrediction model. We usually want to use
                the same splitted train-valid-test throughout the whole process (unless doing some sort of ablation component study).
        Returns:
            data (pd.DataFrame): .csv -> pd.DataFrame instance of the dataset.
        """
        super().__init__(DATASET_ID, DATASET_ZIP_URL, DATAFILES_META)
        
        self.cached_dataset = CachedDataset(DATASET_ID, DATASET_ZIP_URL, DATAFILES_META)

        data_url = self.cached_dataset.get_datafile_from_id(label)

        self.data = pd.read_csv(data_url)

        for idx, row in self.data.iterrows():
            self.data.loc[idx, "text"] = self.basic_preprocess_text(row["text"])
    
        # drop one sentence data.. pretty weak solution
        self.data.drop(self.data[self.data.text.str.count(". ") <= 2].index, inplace=True)

    def basic_preprocess_text(self, text_input: Union[str, List[str]]) -> str:
        text = text_input
        if isinstance(text_input, str):
            text = [text_input]

        collect = []
        for one_text in text:
            one_text = one_text.replace("\n", " ").replace("\t", " ").replace("  ", " ")
            one_text = one_text.replace("‘", "'").replace(" – ","—")
            fix_spaces = re.compile(r'\s*([?!.,]+(?:\s+[?!.,]+)*)\s*')
            one_text = fix_spaces.sub(lambda x: "{} ".format(x.group(1).replace(" ", "")), one_text)


            one_text = one_text.strip()
            collect.append(one_text)
        
        if isinstance(text_input, str):
            return collect[0]
        else:
            return collect
