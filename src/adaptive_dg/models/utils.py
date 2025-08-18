import torch
from torch import nn
from torch.nn import functional as F

def _gaussian_kernel_matrix(x, y, sigmas):
    """
    Computes the Gaussian Kernel matrix
    """
    dist = ((x[:, None, :] - y[None, :, :]) ** 2).sum(axis=-1)[:, :, None]
    beta = 1.0 / (2.0 * torch.Tensor(sigmas)[None, :]).cuda()
    s = torch.matmul(dist, beta)
    k = torch.exp(-s).sum(axis=-1)
    return k


def _mmd(embedding, z):#,  factor_z=2):
    """
    Computes the Maximum-Mean-Discrepancy (MMD): MMD(embedding,z) where
    z follows a standard normal distribution
    Computation is performed for multiple scales
    """
    # z = torch.randn(embedding.shape)
    #z = torch.randn(embedding.shape[0] * factor_z, embedding.shape[1]).cuda()
    sigmas = [
        1e-6,
        1e-5,
        1e-4,
        1e-3,
        1e-2,
        1e-1,
        1,
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        100,
        1e3,
        1e4,
        1e5,
        1e6,
    ]

    loss = torch.mean(_gaussian_kernel_matrix(embedding, embedding, sigmas))
    loss += torch.mean(_gaussian_kernel_matrix(z, z, sigmas))
    loss -= 2 * torch.mean(_gaussian_kernel_matrix(embedding, z, sigmas))
    return loss

class ECELoss(nn.Module):
    """
    due to https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece