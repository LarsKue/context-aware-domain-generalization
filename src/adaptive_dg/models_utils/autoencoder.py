from torch import nn
import torch
# from utils import get_norm_layer, init_net, choose_rand_patches, Patch2Image, RandomCrop
from adaptive_dg.models_utils.blocks import UpBlockBig, UpBlockSmall, UpBlockBig, SEBlock, conv2d, InitLayer


class Encoder(nn.Module):
    
    def __init__(self, encoded_space_dim):
        super().__init__()
        
        ### Convolutional section
        '''
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        '''
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, 128, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
### Linear section
        self.encoder_lin = nn.Sequential(
            # nn.Linear(3 * 3 * 32, 128),
            #nn.Linear(30752, 128),
            #nn.ReLU(True),
            ###
            nn.Linear(30752, 2*1024),
            nn.ReLU(True),
            nn.Linear(2*1024, encoded_space_dim)
            ###
            #nn.Linear(128, encoded_space_dim)
        )
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x
    
class Decoder(nn.Module):
    
    def __init__(self, encoded_space_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(32, 3, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 64, 3, 
            stride=2, output_padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 3, stride=2, 
            padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 3, stride=2, 
            padding=1, output_padding=1)
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        #x = torch.sigmoid(x)
        return x
    

def normalize_second_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()

class FastGanDecoder(nn.Module):
    def __init__(self, ngf=128, z_dim=256, nc=3, img_resolution=256, lite=False):
        super().__init__()
        self.img_resolution = img_resolution
        self.z_dim = z_dim

        # channel multiplier
        nfc_multi = {2: 16, 4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5,
                     512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ngf)

        # layers
        self.init = InitLayer(z_dim, channel=nfc[2], sz=4)

        UpBlock = UpBlockSmall if lite else UpBlockBig

        self.feat_8   = UpBlock(nfc[4], nfc[8])
        self.feat_16  = UpBlock(nfc[8], nfc[16])
        self.feat_32  = UpBlock(nfc[16], nfc[32])
        self.feat_64  = UpBlock(nfc[32], nfc[64])
        self.feat_128 = UpBlock(nfc[64], nfc[128])
        self.feat_256 = UpBlock(nfc[128], nfc[256])

        self.se_64  = SEBlock(nfc[4], nfc[64])
        self.se_128 = SEBlock(nfc[8], nfc[128])
        self.se_256 = SEBlock(nfc[16], nfc[256])

        self.to_big = conv2d(nfc[img_resolution], nc, 3, 1, 1, bias=True)

        if img_resolution > 256:
            self.feat_512 = UpBlock(nfc[256], nfc[512])
            self.se_512 = SEBlock(nfc[32], nfc[512])
        if img_resolution > 512:
            self.feat_1024 = UpBlock(nfc[512], nfc[1024])

    def forward(self, input):#, c, **kwargs):
        # map noise to hypersphere as in "Progressive Growing of GANS"
        # input = normalize_second_moment(input[:, 0])


        feat_4 = self.init(input)
        feat_8 = self.feat_8(feat_4)
        feat_16 = self.feat_16(feat_8)
        feat_32 = self.feat_32(feat_16)
        feat_64 = self.se_64(feat_4, self.feat_64(feat_32))
        feat_128 = self.se_128(feat_8,  self.feat_128(feat_64))

        if self.img_resolution == 32:
            feat_last = feat_32

        if self.img_resolution == 64:
            feat_last = feat_64

        if self.img_resolution >= 128:
            feat_last = feat_128

        if self.img_resolution >= 256:
            feat_last = self.se_256(feat_16, self.feat_256(feat_last))

        if self.img_resolution >= 512:
            feat_last = self.se_512(feat_32, self.feat_512(feat_last))

        if self.img_resolution >= 1024:
            feat_last = self.feat_1024(feat_last)
        

        return self.to_big(feat_last)


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

def UpsampleBlock(cin, cout, scale_factor=2):
    return [
        nn.Upsample(scale_factor=scale_factor),
        nn.Conv2d(cin, cout, 3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.LeakyReLU(0.2, inplace=True),
    ]

def lin_block(in_feat, out_feat, normalize=True):
    layers = [nn.Linear(in_feat, out_feat)]
    layers.append(nn.BatchNorm1d(out_feat, 0.8))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers

def shape_layers(cin, cout, ngf, init_sz):
    return [
        nn.Linear(cin, ngf*2 * init_sz ** 2),
        Reshape(*(-1, ngf*2, init_sz, init_sz)),
        get_norm_layer()(ngf*2),
        *UpsampleBlock(ngf*2, ngf),
        get_norm_layer()(ngf),
        *UpsampleBlock(ngf, cout),
        get_norm_layer()(cout),
    ]

def texture_layers(cin, cout, ngf, init_sz):
    return [
        nn.Linear(cin, ngf*2 * init_sz ** 2),
        Reshape(*(-1, ngf*2, init_sz, init_sz)),
        *UpsampleBlock(ngf*2, ngf*2),
        *UpsampleBlock(ngf*2, ngf),
        nn.Conv2d(ngf, cout, 3, stride=1, padding=1),
        nn.BatchNorm2d(cout),
    ]

class Patch2Image(nn.Module):
    ''' take in patch and copy n_up times to form the full image'''
    def __init__(self, patch_sz,  n_up):
        super(Patch2Image, self).__init__()
        self.patch_sz = patch_sz
        self.n_up = n_up

    def forward(self, x):
        assert x.shape[-1]==self.patch_sz, f"inp.patch_sz ({x.shape[-1]}): =/= self.patch_sz ({self.patch_sz})"
        x = torch.cat([x]*self.n_up, -1)
        x = torch.cat([x]*self.n_up, -2)
        return x

def choose_rand_patches(x, patch_sz, dim):
    assert dim == 2 or dim == 3
    batch_sz = x.shape[0]

    # get all possible patches
    patches = x.unfold(dim, patch_sz, 1)
    n_patches = patches.shape[2]

    # for each image, choose a random patch
    idx = torch.randint(0, n_patches, (batch_sz,))

    if dim == 2:
        patches = patches[torch.arange(batch_sz), :, idx, :]
    elif dim == 3:
        patches = patches[torch.arange(batch_sz), :, :, idx]
    return patches

class RandomCrop(nn.Module):
    def __init__(self, crop_sz):
        super(RandomCrop, self).__init__()
        self.crop_sz = crop_sz

    def forward(self, x):
        img_sz = x.shape[-1]
        assert img_sz >= self.crop_sz, f"img_sz {img_sz} is too small for crop_sz {self.crop_sz}"
        x = choose_rand_patches(x, self.crop_sz, 2)
        x = choose_rand_patches(x, self.crop_sz, 2)
        return 
    
    
def toggle_grad(model, on_or_off):
    for param in model.parameters():
        param.requires_grad = on_or_off

# useful functions from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return nn.Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net

def MMD_multiscale(x, y):
    """ MMD with rationale kernel"""
    # Normalize Inputs Jointly
    device = 'cuda'
    xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2.*xx
    dyy = ry.t() + ry - 2.*yy
    dxy = rx.t() + ry - 2.*zz

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
    torch.zeros(xx.shape).to(device),
    torch.zeros(xx.shape).to(device))

    for a in [0.2, 0.5, 1 ]:
        XX += a**2 * (a**2 + dxx)**-1
        YY += a**2 * (a**2 + dyy)**-1
        XY += a**2 * (a**2 + dxy)**-1

    return torch.mean(XX + YY - 2.*XY)
