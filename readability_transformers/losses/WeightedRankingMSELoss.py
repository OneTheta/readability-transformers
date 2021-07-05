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

from readability_transformers.losses.WeightedMSELoss import WeightedMSELoss

class WeightedRankingMSELoss(nn.Module):
    def __init__(self, alpha, min_err, max_err, min_weight):
        super(WeightedRankingMSELoss, self).__init__()
        self.alpha = alpha
        self.mse_criterion = WeightedMSELoss(min_err, max_err, min_weight)
        
 
    def forward(self, input_scores, target_scores, standard_errors):   
        '''Calculates loss for the difference between predicted scores and
        target scores with MSE and also giving it a ranking loss, penalizing 
        not only for the score values but also for ranking things incorrectly,
        which is a subtle difference.

        Args:
            input_rr (Torch.Tensor): predicted scores
            target_rr (Torch.Tensor): target list of scores
        '''   
       
        assert input_scores.size() == target_scores.size()

        if len(input_scores.shape) == 1:
            input_scores = torch.unsqueeze(input_scores, dim = 0)
            target_scores = torch.unsqueeze(target_scores, dim = 0)

        alpha = self.alpha

        mse_loss = self.mse_criterion(input_scores, target_scores, standard_errors)
        batch_size, N = input_scores.size()  
        pred_rr_repeated = input_scores.repeat_interleave(repeats=N, dim=1)
        pred_rr_repeated = pred_rr_repeated.reshape((batch_size, N, N))
        pred_pw_diff = pred_rr_repeated - pred_rr_repeated.transpose(1,2)

        true_rr_repeated = target_scores.repeat_interleave(repeats=N, dim=1)
        true_rr_repeated = true_rr_repeated.reshape((batch_size, N, N))
        true_pw_diff =  true_rr_repeated.transpose(1,2) - true_rr_repeated

        ranking_loss_matrix = nn.functional.relu(pred_pw_diff.mul(true_pw_diff))
 
        ranking_loss = torch.mean(ranking_loss_matrix)
     
        loss = mse_loss + ranking_loss*alpha
        
        return loss
