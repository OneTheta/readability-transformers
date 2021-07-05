import torch
import torch.nn as nn

class DenormRMSELoss(nn.Module):
    def __init__(self, model=None, target_max=None, target_min=None):
        super(DenormRMSELoss, self).__init__()

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
        return torch.sqrt(self.mse_criterion(denormed_inputs, denormed_targets))
