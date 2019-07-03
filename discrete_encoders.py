import numpy as np
import time

import torch
import torch.nn as nn
from torch.nn import functional as F


class IdEncoder(nn.Module):
    def __init__(self):
        super(IdEncoder,self).__init__()

    def encode(self,x):
        return x

    
class HTEncoder(nn.Module):
    def __init__(self, load_dir):
        super(HTEncoder,self).__init__()
        self.emb_mean = torch.from_numpy(np.load(load_dir + 'emb_mean.npy')).cuda()
        self.emb_mean = self.emb_mean.view(1,4096)

    def encode(self,x):
        # return (x>self.emb_mean).float().cuda()
        return (x>0.07).float().cuda()


    
class RandomEncoder(nn.Module):
    def __init__(self,dim,LOAD_PATH):
        super(RandomEncoder,self).__init__()    
        self.project_mat =  torch.randn(4096,dim).cuda()
        self.emb_mean = torch.from_numpy(np.load(LOAD_PATH + 'emb_mean.npy')).cuda()
        self.emb_mean = self.emb_mean.view(1,4096)
        
    def encode(self, x):
        random_project_emb = torch.matmul(x - self.emb_mean, self.project_mat)
        discrete_emb = (random_project_emb > 0.).float().cuda()
        return random_project_emb + (discrete_emb-random_project_emb).detach()

    

class PCAEncoder(nn.Module):
    def __init__(self,dim,LOAD_PATH):
        super(PCAEncoder,self).__init__()
        np_project_mat = np.load(LOAD_PATH + 'trans_mat.npy')
        self.project_mat = torch.from_numpy(np_project_mat[:,:dim]).cuda()
        self.emb_mean = torch.from_numpy(np.load(LOAD_PATH + 'emb_mean.npy')).cuda()
        self.emb_mean = self.emb_mean.view(1,4096)
        
    def encode(self, x):
        pca_emb = torch.matmul(x - self.emb_mean, self.project_mat)
        discrete_emb = (pca_emb > 0.).float().cuda()
        return pca_emb + (discrete_emb-pca_emb).detach()


    
class LinearAutoEncoder(nn.Module):
    def __init__(self,dim):
        super(LinearAutoEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(2* 2048, dim),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(dim, 2*2048)
        )

    def forward(self, x):
        logits = self.encoder(x)
        latent_code = (logits>0.).float().cuda()
        to_decoder = logits+(latent_code-logits).detach()
        predict = self.decoder(2.*to_decoder-1.)
        return predict

    def encode(self, x):
        logits = self.encoder(x)
        latent_code = (logits>0.).float().cuda()
        to_decoder = logits+(latent_code-logits).detach()
        return to_decoder


    
class NonlinearAutoEncoder(nn.Module):
    def __init__(self, dim, fc_dim=2048):             #fc_dim for dimension of fully-connect layers
        super(NonlinearAutoEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(2* 2048, config.fc_dim),
            nn.Tanh(),
            nn.Linear(config.fc_dim, dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(dim, 2*2048)
        )

    def forward(self, x):
        logits = self.encoder(x)
        latent_code = (logits>0.).float().cuda()
        to_decoder = logits+(latent_code-logits).detach()
        predict = self.decoder(2.*to_decoder-1.)
        return predict

    def encode(self, x):
        logits = self.encoder(x)
        latent_code = (logits>0.).float().cuda()
        to_decoder = logits+(latent_code-logits).detach()
        return to_decoder

