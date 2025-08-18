from torch  import nn
import torch.nn.functional as F
import torchvision

import clip

import copy

from adaptive_dg.models.set_encoder import SetEncoder
from adaptive_dg.models.set_encoders import SimpleSetEncoder
#from point_clouds.models.encoder import InvariantEncoder, InvariantEncoderHParams

class MNIST_Simple(nn.Module):
    """
    simple linear model on mnist
    """

    def __init__(self, input_shape):
        super(MNIST_Simple, self).__init__()

        self.l1 = nn.Linear(28*28*3, 128)
        self.l2 = nn.Linear(128, 16)
        self.relu = nn.ReLU()


    def forward(self, x):
        out = self.relu(self.l1(x.view(len(x), -1)))
        out = self.l2(out)
        return out

class MNIST_CNN(nn.Module):
    """
    due to https://github.com/facebookresearch/DomainBed/blob/main/domainbed/networks.py
    # Batch_norm layer added before ouptut
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """

    def __init__(self, input_shape):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.n_outputs = 128
        self.batch_norm = nn.BatchNorm1d(self.n_outputs)

    def forward(self, x):
        x = self.conv1(x) 
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.avgpool(x)
        x = x.view(len(x), -1)
        #x  = self.batch_norm(x)
        return x

class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([
            nn.Linear(hparams['mlp_width'], hparams['mlp_width'])
            for _ in range(hparams['mlp_depth']-2)])
        self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        self.n_outputs = n_outputs
        self.depth = hparams['mlp_depth']

    def forward(self, x):
        x = self.input(x)
        if self.depth > 0:
            x = self.dropout(x)
            x = F.relu(x)
            for hidden in self.hiddens:
                x = hidden(x)
                x = self.dropout(x)
                x = F.relu(x)
        x = self.output(x)
        return x

class ResNet(nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen
    (due to https://github.com/facebookresearch/DomainBed/blob/main/domainbed/networks.py)
    But we do not freeze BatchNorm
    """
    def __init__(self, input_shape, hparams):
        super(ResNet, self).__init__()
        if hparams['ResNet18']:
            self.network = torchvision.models.resnet18(pretrained=True)
            self.n_outputs = 512
        else:
            self.network = torchvision.models.resnet50(pretrained=True)
            self.n_outputs = 2048

        # self.network = remove_batch_norm_from_resnet(self.network)

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        del self.network.fc
        self.network.fc = Identity()

        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['resnet_dropout'])

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)

class MNIST_Autoencoder(nn.Module):
    def __init__(self, inp_size, enc_size=64):
        super(MNIST_Autoencoder, self).__init__()

        self.dim_inp = inp_size
        self.dim_h1 = 512
        self.dim_h2 = 256
        self.dim_z = enc_size

        self.fc1 = nn.Linear(self.dim_inp, self.dim_h1)
        self.bn1 = nn.BatchNorm1d(self.dim_h1)
        self.fc2 = nn.Linear(self.dim_h1,self.dim_h2)
        self.bn2 = nn.BatchNorm1d(self.dim_h2)
        self.enc = nn.Linear(self.dim_h2, self.dim_z)

        self.fc3 = nn.Linear(self.dim_z, self.dim_h2)
        self.bn3 = nn.BatchNorm1d(self.dim_h2)
        self.fc4 = nn.Linear(self.dim_h2, self.dim_h1)
        self.bn4 = nn.BatchNorm1d(self.dim_h1)
        self.out = nn.Linear(self.dim_h1,self.dim_inp)

        self.relu = nn.ReLU()
        self.acti = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.2)
    
    def encode(self, X):
        X = self.fc1(X)
        X = self.bn1(X)
        X = self.relu(X)
        X = self.dropout(X)
        X = self.fc2(X)
        X = self.bn2(X)
        X = self.relu(X)
        X = self.dropout(X)
        X = self.enc(X)
        return X
    
    def decode(self, X):
        X = self.fc3(X)
        X = self.bn3(X)
        X = self.relu(X)
        X = self.dropout(X)
        X = self.fc4(X)
        X = self.bn4(X)
        X = self.relu(X)
        X = self.dropout(X)
        X = self.out(X)
        X = self.acti(X)
        return X

    def forward(self, X):
        enc = self.encode(X)
        X = self.decode(enc)
        return X, enc

class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Clip(nn.Module):
    def __init__(self, hparams):
        super(Clip, self).__init__()
        self.model, _ = clip.load("ViT-B/32", device='cpu')

    def forward(self, x):
        return self.model.encode_image(x)

def Featurizer(input_shape, hparams_dic):
    """
    similar as in  https://github.com/facebookresearch/DomainBed/blob/main/domainbed/networks.py
    Auto-select an appropriate featurizer for the given input shape.
    """
    hparams = copy.deepcopy(hparams_dic)
    name = hparams.pop('name')
    if len(input_shape) == 1:
        return MLP(input_shape[0], hparams["output_dim"], hparams)
    elif input_shape[1:3] == (28, 28):
        return MNIST_Simple(input_shape)
    elif input_shape[1:3] == (224, 224):
        print(input_shape)
        if name == 'ResNet18' or name == 'ResNet50':
            return ResNet(input_shape, hparams)
        elif name == 'Clip':
            return Clip(hparams)
        else:
            raise NotImplementedError
    elif input_shape[1:3] == (64, 64):
        return MNIST_CNN(input_shape)

    else:
        raise NotImplementedError

        
def EncoderConstructor(hparams_dic):
    """
    Constructor for different set encoders
    Args:
        - name: name  of set encoder
        - hparams: dictionary of hparams
    Returns:
        - encoder: set encoder
    """
    hparams = copy.deepcopy(hparams_dic)
    match hparams["name"]:
        case "Standard":
            hparams.pop("name")
            encoder = SetEncoder(**hparams)
        case "Simple":
            hparams.pop("name")
            hparams.pop("input_dim")
            encoder = SimpleSetEncoder(hparams)

    return encoder
