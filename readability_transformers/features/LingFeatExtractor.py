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

# from lingfeat import extractor

# from readability_transformers.features import Feature

# class LingFeat(Feature):
#     def __init__(self, subgroups):
#         self.subgroups = subgroups
    
#     def extract(self, text: str) -> dict:
#         LingFeat = extractor.pass_text(text)
#         LingFeat.preprocess()
#         features = {}
#         for one_group in self.subgroups:
#             one_group_features = getattr(LingFeat, one_group)()
#             features = {**features, **one_group_features}
#         return features

        
        