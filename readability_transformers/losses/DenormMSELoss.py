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

import torch
import torch.nn as nn

class DenormMSELoss(nn.Module):
    def __init__(self, model=None, target_max=None, target_min=None):
        super(DenormMSELoss, self).__init__()

        if model is None:
            if target_max is None or target_min is None:
                raise Exception("Must supply either a ReadabilityTransformers model or target_max, target_min.")
            else:        
                self.target_max = target_max
                self.target_min = target_min
        else:
            self.target_max = model.feature_maxes["target"]
            self.target_min = model.feature_mins["target"]
        self.mse_criterion = nn.MSELoss(reduction="mean")

    def _denorm(self, values):
        values = values * (self.target_max - self.target_min)
        values = values + self.target_min
        return values

    def forward(self, input_scores, target_scores):
        denormed_inputs = self._denorm(input_scores)
        denormed_targets = self._denorm(target_scores)
        return self.mse_criterion(denormed_inputs, denormed_targets)
