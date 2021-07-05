
from typing import List
from readability_transformers.features import FeatureBase

class FeatureExtraction:
    def __init__(
        self, 
        feature_extractors: List[FeatureBase],
        text_column: str = "excerpt",
        cache: bool = True,
        cache_ids: List[str] = None,
        normalize: bool = True,
        extra_normalize_columns: List[str] = ["target"]
    ):
        self.feature_extractors = feature_extractors
        self.text_column = text_column
        self.cache = cache
        self.cache_ids = cache_ids
        self.normalize = normalize
        self.extra_normalize_columns = extra_normalize_columns

    def forward(self):
        pass