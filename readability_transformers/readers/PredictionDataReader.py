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

import torch
import numpy as np
import pandas as pd
from typing import List
from readability_transformers.readers import DataReader
from torch.utils.data import Dataset, DataLoader

class PredictionDataset(Dataset):
    def __init__(self, inputs, targets, standard_err=None, data_ids=None):
        self.inputs = inputs
        self.targets = targets
        self.standard_err = standard_err if standard_err is not None else None
        self.data_ids = data_ids if data_ids is not None else None

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return_dict = {
            "inputs": self.inputs[idx],
        }
        if self.targets is not None:
            return_dict["target"] = self.targets[idx]
        if self.standard_err is not None:
            return_dict["standard_err"] = self.standard_err[idx]
        if self.data_ids is not None:
            return_dict["data_ids"] = self.data_ids[idx]
        return return_dict

class PredictionDataReader(DataReader):
    def __init__(
        self, 
        feature_df: pd.DataFrame, 
        embedding_matrix: np.ndarray, 
        features: List[str], 
        text_column: str = "excerpt",
        target_column: str = "target",
        no_targets: bool = False, 
        double: bool = True
    ):
        """Takes a pandas dataframe of features and text and an embedding matrix made from a SentenceTransformer
        to create a general purpose datareader that the model code can  utilize.

        Args:
            feature_df (pd.DataFrame): DF object with columns [excerpt, target, standard_error, feature_*]
            embedding_matrix (np.ndarray): ST embedding matrix with same size.
            features (List[str]): list of features in order to pull from df
            no_targets (bool): for inference, we don't have a targets column. defaults to True.
        """
        super(PredictionDataReader, self).__init__()
        self.standard_err = None
        self.targets = None

        if len(feature_df) != embedding_matrix.shape[0]:
            raise Exception("The feature dataframe and the embedding matrix are not of the same shape.")
        
        if "standard_error" in feature_df.columns.values:
            self.standard_err = feature_df["standard_error"].values


        if not no_targets:
            if "target" not in feature_df.columns.values:
                raise Exception("Target column not found. If this is for inference, use no_targets=True.")
            self.targets = feature_df["target"].values
     
        self.features = features
        self.data_ids = feature_df["id"].values
        self.inputs_features = feature_df[self.features].values
        self.inputs_embeddings = embedding_matrix

        if not double: # since by default the above values are float64/double.
            print("NOT DOUBLE")
            self.inputs_features = self.inputs_features.astype(np.float32)
            self.inputs_embeddings = self.inputs_embeddings.astype(np.float32)
            self.standard_err = self.standard_err.astype(np.float32)
            self.targets = self.targets.astype(np.float32)

        """
        let N = # of rows, E = embedding size., F = # of features
        inputs_embeddiing.size() == (N, E)
        inputs_features.size() == (N, F)
            => inputs.size() == (N, E+F)
        """
        N = len(feature_df)
        E = len(self.inputs_embeddings[0])
        F = len(self.inputs_features[0])
        
        torch_dtype = torch.double if double else torch.float32
        self.inputs_features = torch.tensor(self.inputs_features, dtype=torch_dtype)
        self.inputs_embeddings = torch.tensor(self.inputs_embeddings, dtype=torch_dtype)

        self.inputs_features = (self.inputs_features - self.inputs_features.min()) / (self.inputs_features.max() - self.inputs_features.min()) # normalize data
        self.inputs = torch.cat((self.inputs_features, self.inputs_embeddings), axis=-1)
        assert self.inputs.size() == (N, E+F)

        self.dataset = PredictionDataset(self.inputs, self.targets, self.standard_err, self.data_ids)
    
    def get_input_features_in_order(self):
        return self.input_feature_names
        
    def get_standard_err_stats(self):
        standard_error = self.standard_err
        return {
            "min": standard_error.min(),
            "max": standard_error.max()
        }
    def get_dataset(self):
        return self.dataset
    
    def get_dataloader(self, batch_size: int, shuffle: bool = True):
        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )
        return dataloader

    def __len__(self):
        return len(self.inputs.size(0))
        
