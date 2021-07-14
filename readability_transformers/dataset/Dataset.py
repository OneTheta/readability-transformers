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

import os
import pickle
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List
import tqdm
from tqdm import tqdm
from tqdm.autonotebook import trange
from sentence_transformers import SentenceTransformer
from loguru import logger

from readability_transformers.file_utils import load_from_cache_pickle, save_to_cache_pickle
from readability_transformers.features import FeatureBase

CACHE_DIR = os.path.expanduser("~/.cache/readability-transformers/data")

def _holdout_splitter(df, ratios, random_state = None):
    split_collector = []
    for rat_idx, one_rat in enumerate(ratios[:-1]):
        one_frac = one_rat / (1 - sum(ratios[:rat_idx]))
        one_slice = df.sample(frac=one_frac, random_state=random_state)
        split_collector.append(one_slice)
        df = df.drop(one_slice.index)
    split_collector.append(df) # the remaining, i.e. ratios[:-1]
    return split_collector

def _stratified_splitter(df, ratios, target_column: str="target"):
    num_bins = int(np.floor(1 + np.log2(len(df)))) + 2
    cut = pd.cut(df['target'],bins=num_bins,labels=False)
    df["bin"] = cut

    collect_per_bin = []
    for one_bin in range(num_bins):
        one_bin_df = df[df.bin == one_bin]

        split_collector = _holdout_splitter(one_bin_df, ratios)
        collect_per_bin.append(split_collector)
    
    n_splits = len(ratios)
    collect_per_ratio = []
    for split_idx in range(n_splits):
        one_ratio_collect = []
        for bin_idx in range(num_bins):
            one_ratio_collect.append(collect_per_bin[bin_idx][split_idx])
        collect_per_ratio.append(one_ratio_collect)
    
    return [pd.concat(one_collect) for one_collect in collect_per_ratio]

class Dataset(ABC):
    @abstractmethod
    def __init__(self, dataset_id, dataset_zip_url, dataset_meta):
        self.dataset_id = dataset_id
        self.dataset_zip_url = dataset_zip_url
        self.dataset_meta = dataset_meta
        self.dataset_cache_url = os.path.join(CACHE_DIR, self.dataset_id)
        
     
    def _load_cache_or_split(self, ratios: tuple, splitter, ratio_cache_labels: tuple = None):
        """Returns train - valid - test split. Ratios are supplied in that order.

        Args:
            ratios       (tuple): ratio of train - valid - test, respectively.
            random_state (int): random seed for reproducibility
        Returns:
            train (pd.DataFrame): DataFrame object of the train data
            valid (pd.DataFrame)
            test  (pd.DataFrame)

        """
        ratio_labels = ratio_cache_labels if ratio_cache_labels is not None else [str(i) for i in range(len(ratios))]
        if len(ratio_labels) != len(ratios):
            raise Exception("ratios and ratio_cache_labels must be 1-1 corresponding")
        summed = sum(ratios)
        if summed != 1.0:
            raise Exception("Ratios must add up to 1.0. EX: (0.6, 0.2, 0.2)")
        else:
            cache_exists = False
            if ratio_cache_labels: # you want there to be a cache
                logger.info("Considering cache...")
               
                cache_loads = [load_from_cache_pickle(self.dataset_id, f"split_{fn}") for fn in ratio_labels]
                cache_exists = not (True in [isinstance(one, type(None)) for one in cache_loads])
                if cache_exists: # and you had the cache!
                    logger.info(f"Found and cached splits: {ratio_labels}")
                    return cache_loads

            if not cache_exists: # there is no cache 
                if ratio_cache_labels:
                    logger.info(f"Missing at least one cache from: {ratio_cache_labels}")
                logger.info("Creating new split...")

                df = self.data

                split_collector = splitter(df, ratios)

                if ratio_cache_labels: # but you wanted there to be cache. so make cache.
                    filenames = [f"{self.dataset_id}_split_{i}.pkl" for i in ratio_labels]
                
                    for (data_split, split_label, filename) in zip(split_collector, ratio_labels, filenames):
                        save_to_cache_pickle(self.dataset_id, f"split_{split_label}", filename, data_split)

                return split_collector


    
    def holdout_split_data(self, ratios: tuple, ratio_cache_labels: tuple = None):
        splitter = _holdout_splitter
        
        return self._load_cache_or_split(ratios, splitter, ratio_cache_labels)

    def stratified_split_data(self, ratios: tuple, ratio_cache_labels: tuple = None):
        splitter = _stratified_splitter
        return self._load_cache_or_split(ratios, splitter, ratio_cache_labels)