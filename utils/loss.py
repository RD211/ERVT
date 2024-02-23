import torch
import torch.nn as nn

class weighted_RMSE(nn.Module):
    def __init__(self, weights, reduction='mean'):
        super().__init__()
        self.weights = weights
        self.reduction = reduction
        self.mseloss = nn.MSELoss(reduction='none')

    def forward(self, inputs, targets):
        batch_loss = self.mseloss(inputs, targets) * self.weights
        if self.reduction == 'mean':
            return torch.sqrt(torch.mean(batch_loss))
        elif self.reduction == 'sum':
            return torch.sqrt*(torch.sum(batch_loss))
        else:
            return batch_loss

class BCE_MSE(nn.Module):
    """
    Binary Cross Entropy with Weighted MSE Loss.
    """

    def __init__(self, weights, alpha, beta):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.weights = weights

        self.bce = nn.BCELoss()
        self.weighted_mse = weighted_RMSE(self.weights)

    def forward(self, x, target):
        bce_input = x[:, :, 0]
        rmse_input = x[:, :, 1:]

        l = self.alpha * self.bce(bce_input, target[2:]) + self.beta * self.weighted_mse(rmse_input, target[:2])

        return l
