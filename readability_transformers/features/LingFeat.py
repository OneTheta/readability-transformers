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

from .lf import extractor
from . import FeatureBase

class LingFeatExtractor(FeatureBase):
    def __init__(self, subgroups: List[str] = None):
        self.subgroups = ['CKKF_', 'POSF_',  # 'PhrF_', 'TrSF_', 
              'EnDF_', 'EnGF_', 'ShaF_', 'TraF_',
              'TTRF_', 'VarF_', 'PsyF_', 'WorF_']
        
        if subgroups is not None:
            valid = [feature for feature in subgroups if feature not in self.subgroups]
            if len(valid) > 0:
                raise Exception(f"Requested unsupported LINGFEAT subgroup. Supported subgroups: {self.subgroups}")
            else:
                self.subgroups = subgroups
    
    def extract(self, text: str) -> dict:
        num_tokens = len(text.split(" "))
        LingFeat = extractor.pass_text(text)
        LingFeat.preprocess()
        features = {}
        for one_group in self.subgroups:
            one_group_features = getattr(LingFeat, one_group)()
            features = {**features, **one_group_features}
        
        renamed = dict()
        for key in features.keys():
            renamed["lf_"+key] = features[key] / num_tokens
        
        return renamed

    def extract_in_batches(self, texts: List[str]) -> List[dict]:
        features_collect = []
        for text in texts:
            feature_dict = self.extract(text)
            features_collect.append(feature_dict)
        return features_collect
        
        