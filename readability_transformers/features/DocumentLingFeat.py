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

class DocumentLingFeatExtractor(FeatureBase):
    def __init__(self, subgroups: List[str] = None):
        """
        The subgroup list is a subset from the full LingFeat subgroup list 
        where it achieves additional signal when analyzed from a document-level.
        This is to be used when one is using SentenceFeatureExtractor for sentence-level 
        extraction and wants to also use a document-level feature extractor as a holistic 
        analysis of the document on top of sentence-level

        Added subgroups:
            (AdvancedSemantics): CKKF, OSKF, WBKF, WoKF {currently only supports CKKF}
                The above uses Latent Dirichlet Allocation to look at distributions
                of "topics" across a document.
            (Discourse): EnGF
                EnGF uses this idea of an "entity grid", which is about transitions of references
                to various entities across different sentences.
            (LexicoSemantic): VarF
                "Variation Ratio Features", ratio measures of noun, ver, adj, adv variations. Doing this at a document-level
                allows us to view variations at inter-sentence levels, instead of just intra-sentence with sentence-level extraction.
            (Shallow Traditional Features): TraF
                These are classic traditional features such as:
                    def flesch_grade_level(self):
                        result = 0.39*(self.n_token/self.n_sent) + 11.8*(self.n_syll/self.n_token) - 15.59
                        return result
                which are clearly done at a document-level.
        
        Removed subgroups:
            (Syntactic Features): PhrF, TrSF, POSF
                No need to do these again at a document level when that would just be the average of the sentence-level measurements.
                PhrF = counts of certain clauses like NP, VP, and so forth and ratios of these counts. 
                TrSF = Height of sentence parse trees.
                POSF = part of speech counts
            (LexicoSemantic): TTRF, PsyF, WorF
                These are just counts of tokens that satisfy some specific conditions that don't have more meaning at a document-level
                

        Removed Features:
            Some background:
                Features that start with "as_{feature_x}" refer to the value:
                    {feature_x} // # of sentences
                Features that start with "at_{feature_x}" refer to the value:
                    {feature_x} // # of tokens
                
            So features that start with as_ are removed.
        """

        self.subgroups = ['CKKF_',  'EnGF_', 'VarF_', 'TraF_', 'PhrF_', 'TrSF_']
        if subgroups is not None:
            valid = [feature for feature in subgroups if feature not in self.subgroups]
            if len(valid) > 0:
                raise Exception(f"Requested unsupported LINGFEAT subgroup. Supported subgroups: {self.subgroups}")
            else:
                self.subgroups = subgroups
    
    def extract(self, text: str) -> dict:
        LingFeat = extractor.pass_text(text, optimize_subgroups=self.subgroups)
        LingFeat.preprocess()
        features = {}
        for one_group in self.subgroups:
            one_group_features = getattr(LingFeat, one_group)()
            features = {**features, **one_group_features}
        
        renamed = dict()
        for key in features.keys():
            renamed["lf_"+key] = features[key]
        
        return renamed

    def extract_in_batches(self, texts: List[str]) -> List[dict]:
        features_collect = []
        for text in texts:
            feature_dict = self.extract(text)
            features_collect.append(feature_dict)
        return features_collect
        
    
        