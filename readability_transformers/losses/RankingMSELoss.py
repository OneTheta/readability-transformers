import torch
import torch.nn as nn

from readability_transformers.losses.WeightedMSELoss import WeightedMSELoss

class RankingMSELoss(nn.Module):
    def __init__(self, alpha):
        super(RankingMSELoss, self).__init__()
        self.alpha = alpha
        self.mse_criterion = nn.MSELoss()
 
    def forward(self, input_scores, target_scores):    
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

        mse_loss = self.mse_criterion(input_scores, target_scores)
        batch_size, N = input_scores.size()  
        pred_rr_repeated = input_scores.repeat_interleave(repeats=N, dim=1)
        pred_rr_repeated = pred_rr_repeated.reshape((batch_size, N, N))
        pred_pw_diff = pred_rr_repeated - pred_rr_repeated.transpose(1,2)

        true_rr_repeated = target_scores.repeat_interleave(repeats=N, dim=1)
        true_rr_repeated = true_rr_repeated.reshape((batch_size, N, N))
        true_pw_diff =  true_rr_repeated.transpose(1,2) - true_rr_repeated

        ranking_loss = torch.mean(nn.functional.relu(pred_pw_diff.mul(true_pw_diff)))
        
        loss = mse_loss + ranking_loss*alpha
        
        return loss