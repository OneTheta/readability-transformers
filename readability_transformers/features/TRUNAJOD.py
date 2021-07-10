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

from typing import List

import spacy
import TRUNAJOD.givenness
import TRUNAJOD.ttr
from TRUNAJOD import surface_proxies
from TRUNAJOD.syllabizer import Syllabizer
from readability_transformers.features import FeatureBase

NLP = spacy.load('en_core_web_sm')

class TRUNAJODExtractor(FeatureBase):
    def __init__(self, features: List[str] = None):
        self.features = [
            "lexical_diversity_mtld",
            "lexical_density",
            "pos_dissimilarity",
            "connection_words_ratio",
        ]

        if features is not None:
            valid = [feature for feature in features if feature not in self.features]
            if len(valid) > 0:
                raise Exception(f"Requested unsupported TRUNAJOD feature. Supported features: {self.features}")
            else:
                self.features = features

    def extract(self, text: str) -> dict:
        doc = NLP(text)
        feature_dict = dict()
        for feature in self.features:
            if feature == "lexical_diversity_mtld":
                feature_dict[feature] = TRUNAJOD.ttr.lexical_diversity_mtld(doc)
            elif feature == "lexical_density":
                feature_dict[feature] = surface_proxies.lexical_density(doc)
            elif feature == "pos_dissimilarity":
                feature_dict[feature] = surface_proxies.pos_dissimilarity(doc)
            elif feature == "connection_words_ratio":
                feature_dict[feature] = surface_proxies.connection_words_ratio(doc)
        
        renamed = dict()
        for key in feature_dict.keys():
            renamed["trunajod_"+key] = feature_dict[key]
        return renamed
    
    def extract_in_batches(self, texts: List[str]) -> List[dict]:
        features_collect = []
        for text in texts:
            feature_dict = self.extract(text)
            features_collect.append(feature_dict)
        return features_collect
        
        