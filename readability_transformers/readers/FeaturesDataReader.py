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

"""
FeaturesDataReader is basically PredictionDataReader without the sentence embeddings.
The idea is to only relay these raw features and text without any neural features,
to be used for the full pass through ReadabilityTransformers.fit().

In contrast, PredictionDataReader was for rp_model.fit() and it served that
purpose by having the sentence embeddings be part of the dataset, since rp_model.fit()
only trains the prediction layer, separate from the transformer neural features.
"""

import torch
import numpy as np
import pandas as pd
from typing import List
from readability_transformers.readers import DataReader
from torch.utils.data import Dataset, DataLoader

class FeaturesDataset(Dataset):
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

class FeaturesDataReader(DataReader):
    def __init__(
        self, 
        feature_df: pd.DataFrame, 
        features: List[str], 
        text_column: str = "excerpt",
        target_column: str = "target",
        id_column: str = None,
        classification: bool = False,
        labels: List[str] = None,
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
        super(FeaturesDataReader, self).__init__()
        self.standard_err = None
        self.targets = None
        self.classification = classification
        self.labels = labels
        
        if "standard_error" in feature_df.columns.values:
            self.standard_err = feature_df["standard_error"].values

        if not no_targets:
            if target_column not in feature_df.columns.values:
                raise Exception("Target column not found. If this is for inference, use no_targets=True.")
            self.targets = feature_df[target_column].tolist()

            if self.classification:
                # this is a datareader for a classifcation task.
                if self.labels is None:
                    raise Exception("Target labels not given for a classification task.")
                else:
                    target_index_list = []
                    for target in self.targets:
                        try:
                            index = self.labels.index(target)
                            target_index_list.append(index)
                        except:
                            raise Exception(f"target column has value {target} not found in labels={self.labels}")
                    self.targets = target_index_list

                
     
        
        self.features = features
        if id_column is not None:
            self.data_ids = feature_df[id_column].values
        else:
            self.data_ids = None
    
        self.inputs_features = feature_df[self.features].values

        if not double: # since by default the above values are float64/double.
            self.inputs_features = self.inputs_features.astype(np.float32)
            self.standard_err = self.standard_err.astype(np.float32)

            if not self.classification:
                self.targets = self.targets.astype(np.float32)

        N = len(feature_df)
        F = len(self.inputs_features[0])
        
        torch_dtype = torch.double if double else torch.float32
        self.inputs_features = torch.tensor(self.inputs_features, dtype=torch_dtype)
        self.texts = feature_df[text_column].values
        # What was i thinking here?? inputs_features is supposed to come in ALREADY normalized with special configurations.
        # self.inputs_features = (self.inputs_features - self.inputs_features.min()) / (self.inputs_features.max() - self.inputs_features.min()) # normalize data
        self.inputs = []
        for (passage, extracted_features) in zip(self.texts, self.inputs_features):
            one_input = {
                "text": passage,
                "features": extracted_features
            }
            self.inputs.append(one_input)
    

        self.dataset = FeaturesDataset(self.inputs, self.targets, self.standard_err, self.data_ids)
    
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
        
