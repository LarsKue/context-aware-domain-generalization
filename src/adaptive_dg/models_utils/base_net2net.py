import torch
import torch.nn as nn
import functools

# from net2net.modules.distributions.distributions import DiagonalGaussianDistribution


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class ActNorm(nn.Module):
    def __init__(self, num_features, logdet=False, affine=True,
                 allow_reverse_init=False):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.allow_reverse_init = allow_reverse_init

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:,:,None,None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = input.shape

        if self.training and self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.scale * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height*width*torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0]).to(input)
            return h, logdet

        return h

    def reverse(self, output):
        if self.training and self.initialized.item() == 0:
            if not self.allow_reverse_init:
                raise RuntimeError(
                    "Initializing ActNorm in reverse direction is "
                    "disabled by default. Use allow_reverse_init=True to enable."
                )
            else:
                self.initialize(output)
                self.initialized.fill_(1)

        if len(output.shape) == 2:
            output = output[:,:,None,None]
            squeeze = True
        else:
            squeeze = False

        h = output / self.scale - self.loc

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h


class BasicFullyConnectedNet(nn.Module):
    def __init__(self, dim, depth, hidden_dim=256, use_tanh=False, use_bn=False, out_dim=None, use_an=False):
        super(BasicFullyConnectedNet, self).__init__()
        layers = []
        layers.append(nn.Linear(dim, hidden_dim))
        if use_bn:
            assert not use_an
            layers.append(nn.BatchNorm1d(hidden_dim))
        if use_an:
            assert not use_bn
            layers.append(ActNorm(hidden_dim))
        layers.append(nn.LeakyReLU())
        for d in range(depth):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(hidden_dim, dim if out_dim is None else out_dim))
        if use_tanh:
            layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


_norm_options = {
        "in": nn.InstanceNorm2d,
        "bn": nn.BatchNorm2d,
        "an": ActNorm}





class FeatureLayer(nn.Module):
    def __init__(self, scale, in_channels=None, norm='IN'):
        super().__init__()
        self.scale = scale
        self.norm = _norm_options[norm.lower()]
        if in_channels is None:
            self.in_channels = 64*min(2**(self.scale-1), 16)
        else:
            self.in_channels = in_channels
        self.build()

    def forward(self, input):
        x = input
        for layer in self.sub_layers:
            x = layer(x)
        return x

    def build(self):
        Norm = functools.partial(self.norm, affine=True)
        Activate = lambda: nn.LeakyReLU(0.2)
        self.sub_layers = nn.ModuleList([
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=64*min(2**self.scale, 16),
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False),
                Norm(num_features=64*min(2**self.scale, 16)),
                Activate()])


class LatentLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LatentLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.build()

    def forward(self, input):
        x = input
        for layer in self.sub_layers:
            x = layer(x)
        return x

    def build(self):
        self.sub_layers = nn.ModuleList([
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True)
                    ])


class DecoderLayer(nn.Module):
    def __init__(self, scale, in_channels=None, norm='IN'):
        super().__init__()
        self.scale = scale
        self.norm = _norm_options[norm.lower()]
        if in_channels is not None:
            self.in_channels = in_channels
        else:
            self.in_channels = 64*min(2**(self.scale+1), 16)
        self.build()

    def forward(self, input):
        d = input
        for layer in self.sub_layers:
            d = layer(d)
        return d

    def build(self):
        Norm = functools.partial(self.norm, affine=True)
        Activate = lambda: nn.LeakyReLU(0.2)
        self.sub_layers = nn.ModuleList([
                nn.ConvTranspose2d(
                    in_channels=self.in_channels,
                    out_channels=64*min(2**self.scale, 16),
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False),
                Norm(num_features=64*min(2**self.scale, 16)),
                Activate()])


class DenseEncoderLayer(nn.Module):
    def __init__(self, scale, spatial_size, out_size, in_channels=None):
        super().__init__()
        self.scale = scale
        self.in_channels = 64*min(2**(self.scale-1), 16)
        if in_channels is not None:
            self.in_channels = in_channels
        self.out_channels = out_size
        self.kernel_size = spatial_size
        self.build()

    def forward(self, input):
        x = input
        for layer in self.sub_layers:
            x = layer(x)
        return x

    def build(self):
        self.sub_layers = nn.ModuleList([
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=0,
                    bias=True)])


class DenseDecoderLayer(nn.Module):
    def __init__(self, scale, spatial_size, in_size):
        super().__init__()
        self.scale = scale
        self.in_channels = in_size
        self.out_channels = 64*min(2**self.scale, 16)
        self.kernel_size = spatial_size
        self.build()

    def forward(self, input):
        x = input
        for layer in self.sub_layers:
            x = layer(x)
        return x

    def build(self):
        self.sub_layers = nn.ModuleList([
                nn.ConvTranspose2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=0,
                    bias=True)])


class ImageLayer(nn.Module):
    def __init__(self, out_channels=3, in_channels=64):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.build()

    def forward(self, input):
        x = input
        for layer in self.sub_layers:
            x = layer(x)
        return x

    def build(self):
        FinalActivate = lambda: torch.nn.Tanh()
        self.sub_layers = nn.ModuleList([
                nn.ConvTranspose2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False),
                FinalActivate()
                ])


