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
        
        