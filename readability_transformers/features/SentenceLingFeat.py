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

"""
REFERENCES

1. EnGF_: 
https://aclanthology.org/D08-1020.pdf
We use the Brown Coherence Toolkit6 to compute entity grids (Barzilay and Lapata, 2008) for each article. In each sentence, 
an entity is identified as the subject (S), object (O), other (X) (for example, part of a prepositional phrase), 
or not present (N). The probability of each transition type is computed. For example, an S-O transition occurs when an entity
is the subject in one sentence then an object in the next; X-N transition occurs when an entity appears
in non-subject or object position in one sentence and not present in the next, etc.7 The entity coherence
features are the probability of each of these pairs of transitions.
"""
FULL_SUBGROUPS = ['CKKF_', 'POSF_', 'PhrF_', 'TrSF_', 'EnDF_', 
    'EnGF_', 'ShaF_', 'TraF_', 'TTRF_', 'VarF_', 'PsyF_', 'WorF_'
]

FULL_SUBGROUPS = ['CKKF_', 'POSF_', 'TrSF_', 
     'ShaF_', 'TraF_', 'TTRF_', 'VarF_', 'PsyF_', 'WorF_'
]


SENTENCE_LEVEL_SUBGROUPS = [
    "EnDF_", # Related to the count of entities found in the text
    "PhrF_",  # Related to the counts of nouns, verbs, adverb phrases
    
    # Lexico-Semantic features are all done at token-levels. 
    "PsyF_", 
    "TTRF_", 
    "VarF_",
    "WorF_ ",

    "ShaF_", # has to do with count of syllables and words
    "PhrF+",  # has to do with counts of nouns verbs etc
    "TrSF+",  # has to do with parse tree heights
    "POSF_"
]
DOCUMENT_LEVEL_SUBGROUPS = [
    "EnGF_" # See Reference #1 above
]

class SentenceLingFeatExtractor(FeatureBase):
    def __init__(self, subgroups: List[str] = None):
        """
        The subgroup list is a subset from the full LingFeat subgroup list 
        where the subgroups that only make sense at a document-level are removed.

        Removed subgroups:
            (AdvancedSemantics): CKKF, OSKF, WBKF, WoKF
                The above uses Latent Dirichlet Allocation to look at distributions
                of "topics" across a document.
            (Discourse): EnGF
                EnGF uses this idea of an "entity grid", which is about transitions of references
                to various entities across different sentences.

        Removed Features:
            Some background:
                Features that start with "as_{feature_x}" refer to the value:
                    {feature_x} // # of sentences
                Features that start with "at_{feature_x}" refer to the value:
                    {feature_x} // # of tokens
                
            So features that start with as_ are removed.
        """

        self.subgroups = ['POSF_',  'PhrF_', 'TrSF_', 'EnDF_', 'ShaF_', 'TTRF_', 'VarF_', 'PsyF_', 'WorF_']
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
            if not key.startswith("as_"):
                renamed["lf_"+key] = features[key] / num_tokens
        
        return renamed

    def extract_in_batches(self, texts: List[str]) -> List[dict]:
        features_collect = []
        for text in texts:
            feature_dict = self.extract(text)
            features_collect.append(feature_dict)
        return features_collect
        
    
        