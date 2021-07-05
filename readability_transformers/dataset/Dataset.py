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

class Dataset(ABC):
    @abstractmethod
    def __init__(self, dataset_id, dataset_zip_url, dataset_meta):
        self.dataset_id = dataset_id
        self.dataset_zip_url = dataset_zip_url
        self.dataset_meta = dataset_meta
        self.dataset_cache_url = os.path.join(CACHE_DIR, self.dataset_id)
        
     
    def split_train_valid_test(self, ratios: tuple, ratio_cache_labels: tuple = None, random_state: int = 100):
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
                split_collector = []
                for rat_idx, one_rat in enumerate(ratios[:-1]):
                    one_frac = one_rat / (1 - sum(ratios[:rat_idx]))
                    one_slice = df.sample(frac=one_frac, random_state=random_state)
                    split_collector.append(one_slice)
                    df = df.drop(one_slice.index)
                split_collector.append(df) # the remaining, i.e. ratios[:-1]

                if ratio_cache_labels: # but you wanted there to be cache. so make cache.
                    filenames = [f"{self.dataset_id}_split_{i}.pkl" for i in ratio_labels]
                
                    for (data_split, split_label, filename) in zip(split_collector, ratio_labels, filenames):
                        save_to_cache_pickle(self.dataset_id, f"split_{split_label}", filename, data_split)

                return split_collector

    # def apply_features(
    #     self, 
    #     df_list: List[pd.DataFrame],
    #     feature_classes: List[FeatureBase],
    #     text_column: str = "excerpt",
    #     cache: bool = False,
    #     cache_ids: List[str] = None,
    #     normalize=True,
    #     extra_normalize_columns: List[str] = None
    # ) -> List[pd.DataFrame]: 
    #     """Applies the feature classes on the dataframes in df_list such that new columns are appended for each feature.
    #     ReadabilityTransformers is able to do such feature extraction on-the-fly as well. But this can be inefficient when
    #     training multiple epochs, since the same feature extraction will occur repeatedly. This function allows one feature
    #     extraction pass to re-use over and over again.

    #     Args:
    #         df_list (List[pd.DataFrame]): List of Pandas DataFrames that contain text to apply the features to, the column name of which
    #             is specified by the text_column argument.
    #         feature_classes (List[FeatureBase]): List of Feature classes to use to apply features to the text.
    #         text_column: Name of the column in the dataframe that contains the text in interest. Defaults to "excerpt".
    #         cache (bool): Whether to cache the dataframes generated after applying the features. Feature extraction can be a costly
    #             process that might serve well to do only once per dataset. Defaults to False.
    #         cache_ids (List[str]): Must be same length as df_list. The ID value for the cached feature dataset generated from the datasets
    #             in df_list. Defaults to None.
    #         normalize (bool): Normalize the feature values to 0 to 1. Defaults to True.
    #         extra_normalize_columns (List[str]): Name of existing columns to apply normalization to as well, since otherwise
    #             normalization will only be applied to the features extracted from this function. 
        
    #     Returns:
    #         feature_dfs (List[pd.DataFrame]): List of pd.DataFrames that are copies of df_list except with 
    #             feature columns.
    #     """
    #     features_applied = dict() 

    #     if cache_ids:
    #         if len(cache_ids) != len(df_list):
    #             raise Exception("The list of labels must match the df_list parameter.")
    #     if cache:
    #         if cache_ids is None:
    #             raise Exception("Cache set to true but no label for the cache.")
    #         else:
    #             os.makedirs(self.dataset_cache_url, exist_ok=True)
    #             for cache_id in cache_ids:
    #                 filepath = os.path.join(self.dataset_cache_url, "feature_"+cache_id+".pkl")
    #                 if os.path.isfile(filepath):
    #                     logger.info(f"Found & loading cache ID: {cache_id}...")
    #                     loaded = pickle.load(open(filepath, "rb"))
    #                     features_applied[cache_id] = loaded
    #     for one_df in df_list:
    #         col_list = one_df.columns.values
    #         already_features = [col for col in col_list if col.startswith("feature_")]
    #         if len(already_features) > 0:
    #             raise Exception("DataFrame has columns starting with 'feature_'. This is not allowed since applied features will take this format.")
    #     if extra_normalize_columns is None:
    #         extra_normalize_columns = []
    #     else:
    #         if not normalize:
    #             raise Exception("normalize is set to False but extra_normalize_columns is supplied.")
        
    #     remaining_ids = list(range(len(df_list))) # default ids are just counters
    #     remaining = df_list
    #     if cache:
    #         current_loaded = features_applied.keys()
    #         assert set(current_loaded).issubset(set(cache_ids))
    #         remaining_ids = list(set(cache_ids).difference(set(current_loaded)))
    #         remaining = [df_list[cache_ids.index(remain_id)] for remain_id in remaining_ids]

    #     for remain_id, remain_data in zip(remaining_ids, remaining):
    #         data = remain_data
    #         for idx, row in tqdm(remain_data.iterrows(), total=len(remain_data)):
    #             text = row[text_column] # refer to docstring
    #             text_features = dict()
    #             for one_extractor in feature_instances:
    #                 feature_dict = one_extractor.extract(one_text)
    #                 for feature_key in feature_dict.keys():
    #                     text_features[feature_key] = feature_dict[feature_key]
    #             remain_data.loc[idx, ['feature_' + k for k in text_features.keys()]] = text_features.values()
            
    #         features_applied[remain_id] = remain_data

    #         if cache:
    #             logger.info(f"Saving lingfeat {remain_id} to cache...")
    #             filepath = os.path.join(CACHE_DIR, "feature_"+remain_id+".pkl")
    #             pickle.dump(data, open(filepath, "wb"))
    #             features_applied[remain_id] = remain_data
         
    #     final_list = None
    #     if cache:
    #         final_list = [features_applied[cache_id] for cache_id in cache_ids]
    #     else:
    #         final_list = [features_applied[idx] for idx in range(len(df_list))]

    #     feature_list = [col for col in features_applied[0].columns.values if col.startswith("feature_")]
    #     feature_list = feature_list + extra_normalize_columns
        
    #     if normalize:
    #         self.feature_maxes = dict()
    #         self.feature_mins = dict()
    #         for feature in feature_list:
    #             self.feature_maxes[feature] = []
    #             self.feature_mins[feature] = []
            
    #         for df in final_list:
    #             for feature in features:
    #                 one_feat_list = df[feature].values
    #                 self.feature_maxes[feature].append(one_feat_list.max())
    #                 self.feature_mins[feature].append(one_feat_list.min())
            
    #         for feature in feature_list:
    #             fmax = max(self.feature_maxes[feature])
    #             fmin = min(self.feature_mins[feature])
    #             if feature == "target": # to avoid 0.00 and 1.00 values for target & room for extra space at inference time.
    #                 fmax = fmax + 0.02
    #                 fmin = fmin - 0.02
    #             self.feature_maxes[feature] = fmax
    #             self.feature_mins[feature] = fmin

    #             for df_idx, df in enumerate(final_list):
    #                 final_list[df_idx][feature] = df[feature].apply(lambda x: (x-fmin) / (fmax - fmin))
       
    #     # fix nan values.
    #     blacklist_features = []
    #     for feature in feature_list:
    #         for df in final_list:
    #             nan_count = df[feature].isnull().sum()
    #             na_count = df[feature].isna().sum()
                
    #             if nan_count > 0 or na_count > 0:
    #                 blacklist_features.append(feature)

    #     logger.info(f"Columns excluded for NaN values (count={len(blacklist_features)}): {blacklist_features}")
    #     for df_idx, df in enumerate(final_list):
    #         final_list[df_idx] = df.drop(columns=blacklist_features)
      
    #     self.blacklist_features = blacklist_features
    #     self.feature_list = [col for col in final_list[0].columns.tolist() if col.startswith("feature_")]
    #     return final_list



    # def apply_lingfeat_features(
    #     self, 
    #     df_list: List[pd.DataFrame], 
    #     cache_ids: List[str] = None, 
    #     cache: bool = False, 
    #     subgroups: List[str] = None,
    #     normalize=True,
    #     features: List[str] = None,
    #     feature_maxes: dict = None,
    #     feature_mins: dict = None
    # ) -> List[pd.DataFrame]:
    #     """Takes the df_list as input and outputs the same shape dataframe with the lingfeat features added

    #     Args:
    #         subgroups (List[str]): The list of "subgroup" labels for the lingfeat features. Lingfeat comes with
    #             255 feature extraction processes, which are grouped into 10 "subgroups". If not supplied, we simply
    #             apply the default subgroups: {'CKKF_', 'POSF_', 'EnDF_', 'EnGF_', 'ShaF_', 'TraF_', 'TTRF_', 'VarF_', 'PsyF_', 'WorF_'}
    #     Returns:
    #         features_applied (List[pd.DataFrame]): Same list of dataframes, except with the features added to them.
    #     """

    #     if not (None in [features, feature_maxes, feature_mins, subgroups]):
    #         if None in [features, feature_maxes, feature_mins, subgroups]:
    #             raise Exception("If supplying one of features, feature_maxes, feature_mins, must provide the rest too.")
    #         else:
    #             return self.apply_lingfeat_features_existing(df_list, subgroups, features, feature_maxes, feature_mins)
    #     subgroups = subgroups if subgroups else self.DEFAULT_SUB_GROUPS
    #     self.lingfeat_subgroups = subgroups


    #     features_applied = dict() 

    #     if cache_ids:
    #         if len(cache_ids) != len(df_list):
    #             raise Exception("The list of labels must match the df_list parameter.")
    #     if cache:
    #         if cache_ids is None:
    #             raise Exception("Cache set to true but no label for the cache.")
    #         else:
    #             for cache_id in cache_ids:
    #                 filepath = os.path.join(CACHE_DIR, "raw_lingfeat_"+cache_id+".pkl")
    #                 if os.path.isfile(filepath):
    #                     logger.info(f"Found LINGFEAT cache for {cache_id}")
    #                     loaded = pickle.load(open(filepath, "rb"))
    #                     features_applied[cache_id] = loaded

        
    #     remaining_ids = list(range(len(df_list))) # default ids are just counters
    #     remaining = df_list
    #     if cache:
    #         current_loaded = features_applied.keys()
    #         assert set(current_loaded).issubset(set(cache_ids))
    #         remaining_ids = list(set(cache_ids).difference(set(current_loaded)))
    #         remaining = [df_list[cache_ids.index(remain_id)] for remain_id in remaining_ids]

    #     logger.info(f"Extrating lingfeat features for {remaining_ids}")
    #     for remain_id, remain_data in zip(remaining_ids, remaining):
    #         data = remain_data
    #         for idx, row in tqdm(remain_data.iterrows(), total=len(remain_data)):
    #             text = row["excerpt"]
    #             LingFeat = extractor.pass_text(text)
    #             LingFeat.preprocess()
    #             features = {}
    #             for one_group in subgroups:
    #                 one_group_features = getattr(LingFeat, one_group)()
    #                 features = {**features, **one_group_features}
    #             remain_data.loc[idx, ['feature_' + k for k in features.keys()]] = features.values()
            
    #         features_applied[remain_id] = remain_data

    #         if cache:
    #             logger.info(f"Saving lingfeat {remain_id} to cache...")
    #             filepath = os.path.join(CACHE_DIR, "raw_lingfeat_"+remain_id+".pkl")
    #             pickle.dump(data, open(filepath, "wb"))
    #             features_applied[remain_id] = remain_data
         
    #     final_list = None
    #     if cache:
    #         final_list = [features_applied[cache_id] for cache_id in cache_ids]
    #     else:
    #         final_list = [features_applied[idx] for idx in range(len(df_list))]

    #     features = self.get_lingfeat_features(final_list[0])
    #     features.append("target")
    #     if normalize:
    #         self.lingfeat_maxes = dict()
    #         self.lingfeat_mins = dict()
    #         for feature in features:
    #             self.lingfeat_maxes[feature] = []
    #             self.lingfeat_mins[feature] = []
            
    #         for df in final_list:
    #             for feature in features:
    #                 one_feat_list = df[feature].values
    #                 self.lingfeat_maxes[feature].append(one_feat_list.max())
    #                 self.lingfeat_mins[feature].append(one_feat_list.min())
            
    #         for feature in features:
    #             fmax = max(self.lingfeat_maxes[feature])
    #             fmin = min(self.lingfeat_mins[feature])
    #             if feature == "target": # to avoid 0.00 and 1.00 values for target.
    #                 fmax = fmax + 0.01
    #                 fmin = fmin - 0.01
    #             self.lingfeat_maxes[feature] = fmax
    #             self.lingfeat_mins[feature] = fmin

    #             for df_idx, df in enumerate(final_list):
    #                 final_list[df_idx][feature] = df[feature].apply(lambda x: (x-fmin) / (fmax - fmin))
       
    #     # fix nan values.
    #     blacklist_features = []
    #     for feature in features:
    #         for df in final_list:
    #             nan_count = df[feature].isnull().sum()
    #             na_count = df[feature].isna().sum()
                
    #             if nan_count > 0 or na_count > 0:
    #                 blacklist_features.append(feature)

    #     logger.info(f"Columns excluded for NaN values (count={len(blacklist_features)}): {blacklist_features}")
    #     for df_idx, df in enumerate(final_list):
    #         final_list[df_idx] = df.drop(columns=blacklist_features)
      
    #     self.blacklist_features = blacklist_features
    #     self.lingfeat_features = [col for col in final_list[0].columns.tolist() if col.startswith("feature_")]
    #     return final_list

    # def apply_lingfeat_features_existing(self, df_list, subgroups, features, feature_maxes, feature_mins):
    #     normalize_feat = lambda value, feature_name: (value - feature_mins[feature_name]) / (feature_maxes[feature_name] - feature_mins[feature_name])

    #     for data_idx, data in enumerate(df_list):
    #         for idx, row in data.iterrows():
    #             text = row["excerpt"]
    #             LingFeat = extractor.pass_text(text)
    #             LingFeat.preprocess()

    #             out_features = {}
    #             for one_group in subgroups:
    #                 one_group_features = getattr(LingFeat, one_group)()
    #                 out_features = {**out_features, **one_group_features}
                
    #             out_feature_names = ['feature_'+k for k in out_features.keys()]
    
    #             for feature in features:
    #                 one_feat_value = out_features[feature.replace("feature_", "")]
    #                 normalized_value = normalize_feat(one_feat_value, feature)
                    
    #                 data.loc[idx, feature] = normalized_value
    #         df_list[data_idx] = data
        
    #     return df_list
            


    # def get_lingfeat_features(self, data: pd.DataFrame):
    #     columns = data.columns.tolist()
    #     columns = [i for i in columns if "feature_" in i]
    #     return columns

    # def get_lingfeat_feature_count(self, data: pd.DataFrame):
    #     """Looks at a pandas dataframe and returns the count of lingfeats that were applied to it.
    #     """
    #     columns = self.get_lingfeat_features(data)

    #     return len(columns)

   
    # def get_st_embedding_size(self, embed: np.ndarray):
    #     """Given a list of st_embeddings or one st_embedding, returns its size.
    #     """
    #     shape = embed.shape
    #     return shape[-1]            
        
    # def save_parameters(self, path_to_folder):
    #     output_path = os.path.join(path_to_folder, "dataset.pkl")
    #     members = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__") and not attr.startswith("_")]
    #     print("members", members)
    #     param_dict = dict()
    #     for member in members:
    #         param_dict[member] = getattr(self, member)
    #     pickle.dump(param_dict, open(output_path, "wb"))
        
    # def load_parameters(self, path_to_folder):
    #     output_path = os.path.join(path_to_folder, "dataset.pkl")

    #     param_dict = pickle.load(open(output_path, "rb"))
    #     for param, value in param_dict.items():
    #         setattr(self, param, value)
    #     return param_dict
    # def __len__(self):
    #     return len(self.data)