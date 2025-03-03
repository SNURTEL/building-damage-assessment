import torch
import torch.nn as nn

# https://arxiv.org/pdf/1909.13589
# https://github.com/ZJULearning/MaxSquareLoss


class MaxSquareloss(nn.Module):
    def __init__(self, ignore_index=-1, num_class=19):
        """
        Initializes the MSLLoss class.

        Args:
            ignore_index: The index to ignore in the loss calculation. Defaults to -1.
            num_class: The number of classes. Defaults to 19.
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.num_class = num_class

    def forward(self, pred, prob):
        """
        Calculates the maximum squares loss.

        Args:
            pred: Predictions tensor of shape (N, C, H, W).
            prob: Probability tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: Maximum squares loss.

        """
        # prob -= 0.5
        mask = prob != self.ignore_index
        loss = -torch.mean(torch.pow(prob, 2)[mask]) / 2
        return loss


class IW_MaxSquareloss(nn.Module):
    def __init__(self, ignore_index=-1, num_class=19, ratio=0.2):
        """
        Initializes the MSLLoss object.

        Args:
            ignore_index: The index to ignore in the loss calculation. Defaults to -1.
            num_class: The number of classes. Defaults to 19.
            ratio: The ratio used for weighting approximated class difficulty in loss calculation. Defaults to 0.2.
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.num_class = num_class
        self.ratio = ratio

    def forward(self, pred, prob, label=None):
        """
        Forward pass of the MSL loss function.

        Args:
            pred: The predicted probabilities of shape (N, C, H, W).
            prob: The probability tensor of shape (N, C, H, W).
            label: The label tensor of shape (N, H, W) or None. If None, argpred will be used as the label.

        Returns:
            The computed loss value.
        """
        # prob -= 0.5
        N, C, H, W = prob.size()
        mask = prob != self.ignore_index
        maxpred, argpred = torch.max(prob, 1)
        mask_arg = maxpred != self.ignore_index
        argpred = torch.where(mask_arg, argpred, torch.ones(1).to(prob.device, dtype=torch.long) * self.ignore_index)
        if label is None:
            label = argpred
        weights = []
        batch_size = prob.size(0)
        for i in range(batch_size):
            hist = torch.histc(
                label[i].cpu().data.float(), bins=self.num_class + 1, min=-1, max=self.num_class - 1
            ).float()
            hist = hist[1:]
            weight = (
                (1 / torch.max(torch.pow(hist, self.ratio) * torch.pow(hist.sum(), 1 - self.ratio), torch.ones(1)))
                .to(argpred.device)[argpred[i]]
                .detach()
            )
            weights.append(weight)
        weights = torch.stack(weights, dim=0)
        mask = mask_arg.unsqueeze(1).expand_as(prob)
        prior = torch.mean(prob, (2, 3), True).detach()
        # loss = -torch.sum((torch.pow(prob, 2)*weights)[mask]) / (batch_size*self.num_class)
        # fix for tensor shape issue
        loss = -torch.sum((torch.pow(prob, 2) * weights.unsqueeze(dim=1))[mask]) / (batch_size * self.num_class)
        return loss
