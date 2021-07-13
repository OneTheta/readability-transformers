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

class WeightedMSELoss(nn.Module):
    def __init__(self, min_err: int, max_err: int, min_weight: int):
        """Weigh MSE Loss such that the greater the standard_err in the data, we weight
        the MSE loss less. Linearly interpolate the weights such that the datapoint with 
        the max error term would be weighed min_weight*original_mse, and the datapoint
        with the minimal error term would be weighed: 1*original_mse.

        e.g. (min_err, max_err) = (0.1, 1.2), min_weight = 0.3.
        Then f:[0.1, 1.2] => [0.3, 1]

        so [0.1, 1.2] => [0, 1.1] # -min_err
        [0, 1.1] => [0, 1]        # /(max_err - min_err)
        [0, 1] => [0, 0.7]        # *(1-min_weight)
        [0, 0.7] => [0.3, 1]      # + min_weight


        Arguments:
            min_err {int} -- [description]
            max_err {int} -- [description]
            min_weight {int} -- [description]
        """
        self.min_err = min_err
        self.max_err = max_err
        self.min_weight = min_weight

        super(WeightedMSELoss, self).__init__()

    def forward(self, input_scores, target_scores, standard_errors):    
        weights = (standard_errors - self.min_err) / (self.max_err - self.min_err)
        weights = weights * (1.0 - self.min_weight)
        weights = weights + self.min_weight
        
        return (weights * (input_scores - target_scores) ** 2).mean()
