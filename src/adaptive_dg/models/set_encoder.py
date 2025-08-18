import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial



class MAB(nn.Module):
    """Multihead attention block."""
    
    def __init__(self, dim_Q=64, dim_K=64, dim_V=64, num_heads=2, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/np.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

    
class PMA(nn.Module):
    """Pooling by multihead attention."""
    
    def __init__(self, dim, num_heads=2, num_seeds=32, ln=True):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)
        self.out_dim = int(dim * num_seeds)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X).flatten(start_dim=1)

class SkipConnection(nn.Module):
    def __init__(self, dim_input, dim_output, layer_width=1024, p=0.25):
        super(SkipConnection, self).__init__()
        n = layer_width

        self.l1 = nn.Linear(dim_input, n)
        self.l2 = nn.Linear(n, n)
        self.l6 = nn.Linear(n, dim_output)


        self.dropout = nn.Dropout(p=p)

        self.r1 =  nn.PReLU()

    def forward(self, x):

        out =  self.r1(self.l1(x))
        out =  self.r1(self.l2(out)) +  out
        
        out = self.dropout(out)
        out =  self.l6(out) 
        return out


class SetEncoder(nn.Module):
    def __init__(self, input_dim=8, latent_dim=8, output_dim=128, layer_width=1024, p=0.25, pma_args={}):
        super(SetEncoder, self).__init__()
        
        # Pre-pooling net
        self.enc = nn.Sequential(
            nn.Linear(input_dim, layer_width),
            nn.PReLU(),
            nn.LayerNorm(layer_width),
            nn.Dropout(p=p),
            nn.Linear(layer_width, layer_width),
            nn.PReLU(),
            nn.LayerNorm(layer_width),
            nn.Dropout(p=p),
            nn.Linear(layer_width, latent_dim),
        ) 
        
        # Learnable or fixed pooling
        if pma_args is None:
            self.pooler = partial(torch.mean, axis=1)
            dim_in_decoder = latent_dim
        else:
            self.pooler = PMA(latent_dim, **pma_args)
            dim_in_decoder = self.pooler.out_dim
            
        
        # Post pooling net
        self.dec = nn.Sequential(
                nn.Linear(dim_in_decoder, output_dim)
        )
        
    def forward(self, X):
        X = self.enc(X)
        X = self.pooler(X)
        X = self.dec(X)
        return X

class SetEncoderConv(nn.Module):
    def __init__(self,input_dim=8, latent_dim=8, output_dim=128, layer_width=1024, p=0.25, pma_args={},maxpool=False):
        ''' Convolutional SetEncoder that works on image input directly.
        '''
        super(SetEncoderConv, self).__init__()
        
        factor = int((224 / 2**3)**2*layer_width)
        # kinda dirty to define it like that but can't think of an elegant solution


        # Pre-pooling net
        if maxpool:
            self.enc = nn.Sequential(
                nn.Conv2d(input_dim,layer_width,3,1,1),
                nn.GroupNorm(1, layer_width),
                nn.PReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(p=p),
                nn.Conv2d(layer_width,layer_width,3,1,1),
                nn.GroupNorm(1, layer_width),
                nn.PReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(p=p),
                nn.Conv2d(layer_width,layer_width,3,1,1),
                nn.GroupNorm(1, layer_width),
                nn.PReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(p=p),
                nn.Flatten(),
                nn.Linear(factor, layer_width),
                nn.LayerNorm(layer_width),
                nn.PReLU(),
                nn.Dropout(p=p),
                nn.Linear(layer_width, latent_dim)
            )
        else:
            self.enc = nn.Sequential(
                nn.Conv2d(input_dim,layer_width,4,2,1),
                nn.BatchNorm2d(layer_width),
                nn.PReLU(),
                nn.Dropout(p=p),
                nn.Conv2d(layer_width,layer_width,4,2,1),
                nn.BatchNorm2d(layer_width),
                nn.PReLU(),
                nn.Dropout(p=p),
                nn.Conv2d(layer_width,layer_width,4,2,1),
                nn.BatchNorm2d(layer_width),
                nn.PReLU(),
                nn.Dropout(p=p),
                nn.Flatten(),
                nn.Linear(factor, layer_width),
                nn.LayerNorm(layer_width),
                nn.PReLU(),
                nn.Dropout(p=p),
                nn.Linear(layer_width, latent_dim)
            ) 
        
        # Learnable or fixed pooling
        if pma_args is None:
            self.pooler = partial(torch.mean, axis=1)
            dim_in_decoder = latent_dim
        else:
            self.pooler = PMA(latent_dim, **pma_args)
            dim_in_decoder = self.pooler.out_dim
            
        
        # Post pooling net
        self.dec = nn.Sequential(
                nn.Linear(dim_in_decoder, output_dim)
        )

    def forward(self, X):
        X = self.enc(X)
        X = self.pooler(X.unsqueeze(0))
        X = self.dec(X)
        return X