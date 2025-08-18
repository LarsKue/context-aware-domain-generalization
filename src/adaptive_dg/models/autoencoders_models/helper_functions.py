import torch
import torch.nn as nn

class Decoder(nn.Module):
    
    def __init__(self,ch_inp,ch_h1=32,ch_h2=64,ch_h3=128,ch_h4=256,ch_h5=512,enc_size=1024,env_cond=False,env_size=None):
        super(Decoder, self).__init__()
        
        self.ch_inp = ch_inp
        self.factor = 49 #5 stages halfing size 224*224 = 7*7 on last stage
        self.ch_h1 = ch_h1
        self.ch_h2 = ch_h2
        self.ch_h3 = ch_h3
        self.ch_h4 = ch_h4
        self.ch_h5 = ch_h5
        self.dim_z = enc_size
        self.env_cond = env_cond
        
        
        # decoder
        self.dim_dec = self.dim_z + env_size if self.env_cond else self.dim_z
        self.dec = nn.Linear(self.dim_dec, self.ch_h5*self.factor)
        self.tbn1 = nn.BatchNorm1d(self.ch_h5*self.factor)
        self.tconv1 = nn.ConvTranspose2d(self.ch_h5,self.ch_h4,4,2,1)
        self.tbn2 = nn.BatchNorm2d(self.ch_h4)
        self.tconv2 = nn.ConvTranspose2d(self.ch_h4,self.ch_h3,4,2,1)
        self.tbn3 = nn.BatchNorm2d(self.ch_h3)
        self.tconv3 = nn.ConvTranspose2d(self.ch_h3,self.ch_h2,4,2,1)
        self.tbn4 = nn.BatchNorm2d(self.ch_h2)
        self.tconv4 = nn.ConvTranspose2d(self.ch_h2,self.ch_h1,4,2,1)
        self.tbn5 = nn.BatchNorm2d(self.ch_h1)
        self.tconv5 = nn.ConvTranspose2d(self.ch_h1,self.ch_inp,4,2,1)      

        self.relu = nn.ReLU()
        self.acti = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.25)
        
    def forward(self, X):
        X = self.dec(X)
        X = self.tbn1(X)
        X = self.relu(X)
        X = self.dropout(X)
        X = X.view(-1,self.ch_h5,7,7) # again hardcoded for simplicity
        
        X = self.tconv1(X)
        X = self.tbn2(X)
        X = self.relu(X)
        X = self.dropout(X)
        
        X = self.tconv2(X)
        X = self.tbn3(X)
        X = self.relu(X)
        X = self.dropout(X)
        
        X = self.tconv3(X)
        X = self.tbn4(X)
        X = self.relu(X)
        X = self.dropout(X)
        
        X = self.tconv4(X)
        X = self.tbn5(X)
        X = self.relu(X)
        X = self.dropout(X)
        
        X = self.tconv5(X)
        X = self.acti(X)
        return X
    
class Encoder(nn.Module):
    def __init__(self, ch_inp,ch_h1=32,ch_h2=64,ch_h3=128,ch_h4=256,ch_h5=512,enc_size=1024,env_cond=False,env_size=None):
        super(Encoder, self).__init__()
            
        self.ch_inp = ch_inp
        self.factor = 49 #5 stages halfing size 224*224 = 7*7 on last stage
        self.ch_h1 = ch_h1
        self.ch_h2 = ch_h2
        self.ch_h3 = ch_h3
        self.ch_h4 = ch_h4
        self.ch_h5 = ch_h5
        self.dim_z = enc_size
        self.env_cond = env_cond
        
        # encoder
        self.conv1 = nn.Conv2d(self.ch_inp,self.ch_h1,4,2,1)
        self.bn1 = nn.BatchNorm2d(self.ch_h1)
        self.conv2 = nn.Conv2d(self.ch_h1,self.ch_h2,4,2,1)
        self.bn2 = nn.BatchNorm2d(self.ch_h2)
        self.conv3 = nn.Conv2d(self.ch_h2,self.ch_h3,4,2,1)
        self.bn3 = nn.BatchNorm2d(self.ch_h3)
        self.conv4 = nn.Conv2d(self.ch_h3,self.ch_h4,4,2,1)
        self.bn4 = nn.BatchNorm2d(self.ch_h4)
        self.conv5 = nn.Conv2d(self.ch_h4,self.ch_h5,4,2,1)
        self.bn5 = nn.BatchNorm2d(self.ch_h5)
        self.enc = nn.Linear(self.ch_h5*self.factor, self.dim_z)
        
        self.relu = nn.ReLU()
        self.acti = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.25)
    
    def forward(self, X):
        X = self.conv1(X)
        X = self.bn1(X)
        X = self.relu(X)
        X = self.dropout(X)

        X = self.conv2(X)
        X = self.bn2(X)
        X = self.relu(X)
        X = self.dropout(X)

        X = self.conv3(X)
        X = self.bn3(X)
        X = self.relu(X)
        X = self.dropout(X)

        X = self.conv4(X)
        X = self.bn4(X)
        X = self.relu(X)
        X = self.dropout(X)

        X = self.conv5(X)
        X = self.bn5(X)
        X = self.relu(X)
        X = self.dropout(X)

        X = X.view(X.shape[0],-1)
        X = self.enc(X)
        return X