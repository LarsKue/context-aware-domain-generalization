from torch import nn
import torch

import torch.nn.functional as F


from adaptive_dg.models.utils import ECELoss

from sklearn.decomposition import PCA
import adaptive_dg.models.utils_networks as utils_networks

from adaptive_dg.models.domain_generalizer import DomainGeneralizer, DGHParams
from lightning_trainable.modules import FullyConnectedNetwork

def _gaussian_kernel_matrix(x, y, sigmas):
    """
    Computes the Gaussian Kernel matrix
    """
    dist = ((x[:, None, :] - y[None, :, :]) ** 2).sum(axis=-1)[:, :, None]
    beta = 1.0 / (2.0 * torch.Tensor(sigmas)[None, :]).cuda()
    s = torch.matmul(dist, beta)
    k = torch.exp(-s).sum(axis=-1)
    return k

def _mmd(embedding, factor_z=2):
    """
    Computes the Maximum-Mean-Discrepancy (MMD): MMD(embedding,z) where
    z follows a standard normal distribution
    Computation is performed for multiple scales
    """
    # z = torch.randn(embedding.shape)
    z = torch.randn(embedding.shape[0] * factor_z, embedding.shape[1]).cuda()
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

def mmd_compare_two(embedding1, embedding2):
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

    loss = torch.mean(_gaussian_kernel_matrix(embedding1, embedding1, sigmas))
    loss += torch.mean(_gaussian_kernel_matrix(embedding2, embedding2, sigmas))
    loss -= 2 * torch.mean(_gaussian_kernel_matrix(embedding1, embedding2, sigmas))
    return loss

    
class Predictor(nn.Module):
    """
    Classsifier that can use the environment information
    """
    #def __init__(self, input_dim, output_dim, cond_dim=0, layer_width=2048):
    def __init__(self, input_dim, output_dim, cond_dim=0, layer_width=512, dropout=0.0, depth=2):
        super().__init__()
        self.cond_dim = cond_dim
        '''
        self.l1 = nn.Linear(input_dim+cond_dim, layer_width)
        self.l1 = nn.Linear(input_dim+cond_dim, output_dim)
        self.l2 = nn.Linear(layer_width+cond_dim, layer_width)
        self.l3 = nn.Linear(layer_width+cond_dim, output_dim)
        self.r = nn.ReLU()
        '''

        self.depth = depth
        self.input = nn.Linear(input_dim+cond_dim, layer_width)
        self.dropout = nn.Dropout(dropout)
        if depth >1:
            self.hiddens = nn.ModuleList([
                nn.Linear(layer_width+cond_dim, layer_width)
                for _ in range(depth-1)])
        self.output = nn.Linear(layer_width, output_dim)
        self.n_outputs =output_dim 

    def forward(self, x, c=None):
        if self.cond_dim >0:
            if c is None:
                c = torch.zeros(x.shape[0], self.cond_dim).to(x.device)
            x = torch.cat((x, c), 1)
            x = self.input(x)
            if self.depth > 0:
                x = self.dropout(x)
                x = F.relu(x)
            if self.depth >1:
                for hidden in self.hiddens:
                    x = torch.cat((x, c), 1)
                    x = hidden(x)
                    x = self.dropout(x)
                    x = F.relu(x)
            x = self.output(x)

        else:
            x = self.input(x)
            if self.depth > 0:
                x = self.dropout(x)
                x = F.relu(x)
            if self.depth > 1:
                for hidden in self.hiddens:
                    x = hidden(x)
                    x = self.dropout(x)
                    x = F.relu(x)
            x = self.output(x)
        return x

def cross_entropy_custom_fct(input, target):
    return torch.mean(-torch.sum(target * torch.log(input), 1))



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