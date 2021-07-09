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
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from typing import List

from . import FeatureBase


class TransformersLogitsExtractor(FeatureBase):
    def __init__(self, device: str, custom_seq_classification_trf = None):
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.device = device
        self.num_labels = 6

        local_seq_classification_path = os.path.expanduser("~/.cache/readability-transformers/models/roberta_sequence_classification")
        if custom_seq_classification_trf is not None:
            self.trf_model = custom_seq_classification_trf
        else:
            self.trf_model = RobertaForSequenceClassification.from_pretrained(local_seq_classification_path)
        self.trf_model.to(self.device)
        self.trf_model.eval()
                
    
    def extract(self, text: str) -> dict:
        inputs = self.tokenizer(
            [text],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        inputs = inputs.to(self.device)

        outputs = self.trf_model(**inputs)
        soft_labels = outputs.logits   
        soft_labels = soft_labels.cpu().tolist()[0]

        features = dict()
        for c in range(self.num_labels):
            feature_name = f'trf_logits_{c + 1}'
            features[feature_name] = soft_labels[c]

        return features

        
        