import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import *
import time

class GAT_gate(torch.nn.Module):
    def __init__(self, n_in_feature, n_out_feature):
        super(GAT_gate, self).__init__()
        self.W = nn.Linear(n_in_feature, n_out_feature)
        #self.A = nn.Parameter(torch.Tensor(n_out_feature, n_out_feature))
        self.A = nn.Parameter(torch.zeros(size=(n_out_feature, n_out_feature)))
        self.gate = nn.Linear(n_out_feature*2, 1)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x, adj):
        h = self.W(x)
        batch_size = h.size()[0]
        N = h.size()[1]
        e = torch.einsum('ijl,ikl->ijk', (torch.matmul(h,self.A), h))
        e = e + e.permute((0,2,1))
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        #attention = F.dropout(attention, self.dropout, training=self.training)
        #h_prime = torch.matmul(attention, h)
        attention = attention*adj
        h_prime = F.relu(torch.einsum('aij,ajk->aik',(attention, h)))
       
        coeff = torch.sigmoid(self.gate(torch.cat([x,h_prime], -1))).repeat(1,1,x.size(-1))
        retval = coeff*x+(1-coeff)*h_prime
        return retval

class GConv(torch.nn.Module):
    def __init__(self, n_in_feature, n_out_feature):
        super(GConv, self).__init__()
        self.W = nn.Linear(n_in_feature, n_out_feature)
    
    def forward(self, x, adj):
        x = self.W(x)
        x = torch.einsum('xjk,xkl->xjl', (adj.clone(), x))
        #x = torch.bmm(adj, x)
        return F.relu(x)

class GConv_gate(torch.nn.Module):
    def __init__(self, n_in_feature, n_out_feature):
        super(GConv_gate, self).__init__()
        self.W = nn.Linear(n_in_feature, n_out_feature)
        self.gate = nn.Linear(n_out_feature*2, 1)
    
    def forward(self, x, adj):
        m = self.W(x)
        m = F.relu(torch.einsum('xjk,xkl->xjl', (adj.clone(), m)))
        coeff = torch.sigmoid(self.gate(torch.cat([x,m], -1))).repeat(1,1,x.size(-1))
        retval = coeff*x+(1-coeff)*m

        #x = torch.bmm(adj, x)
        return retval

class GGNN(torch.nn.Module):
    def __init__(self, n_in_feature, n_out_feature):
        super(GGNN, self).__init__()
        self.W = nn.Linear(n_in_feature, n_out_feature)
        self.C = nn.GRUCell(n_out_feature, n_out_feature)
    
    def forward(self, x, adj):
        m = self.W(x)
        m = torch.einsum('xjk,xkl->xjl', (adj.clone(), m))
        x_size = x.size()
        m = m.view(-1, m.size(-1))
        x = x.view(-1, x.size(-1))
        x = self.C(m, x)
        #hs = C(hs, ms)
        x = x.view(x_size[0], x_size[1], m.size(-1))
        #x = torch.bmm(adj, x)
        return x

class ConcreteDropout(nn.Module):
    def __init__(self, weight_regularizer=1e-6,
                 dropout_regularizer=1e-5, init_min=0.1, init_max=0.1):
        super(ConcreteDropout, self).__init__()

        
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        
        init_min = np.log(init_min) - np.log(1. - init_min)
        init_max = np.log(init_max) - np.log(1. - init_max)
        
        self.p_logit = nn.Parameter(torch.empty(1).uniform_(init_min, init_max))
        
    def forward(self, x1, layer, x2=None, x3=None):
        p = torch.sigmoid(self.p_logit)
        if x2 is None or x3 is None:
            out = layer(self._concrete_dropout(x1, p))
        else:
            out = layer(self._concrete_dropout(x1, p), x3)-layer(self._concrete_dropout(x1, p), x2)
        
        sum_of_square = 0
        for param in layer.parameters():
            sum_of_square += torch.sum(torch.pow(param, 2))
        
        weights_regularizer = self.weight_regularizer * sum_of_square / (1 - p)
        
        dropout_regularizer = p * torch.log(p)
        dropout_regularizer += (1. - p) * torch.log(1. - p)
        
        input_dimensionality = x1[0].numel() # Number of elements of first item in batch
        dropout_regularizer *= self.dropout_regularizer * input_dimensionality
        
        regularization = weights_regularizer + dropout_regularizer
        return out, regularization
        
    def _concrete_dropout(self, x, p):
        eps = 1e-7
        temp = 0.1

        unif_noise = torch.rand_like(x)

        drop_prob = (torch.log(p + eps)
                    - torch.log(1 - p + eps)
                    + torch.log(unif_noise + eps)
                    - torch.log(1 - unif_noise + eps))
        
        drop_prob = torch.sigmoid(drop_prob / temp)
        random_tensor = 1 - drop_prob
        retain_prob = 1 - p
        
        x  = torch.mul(x, random_tensor)
        x /= retain_prob
        
        return x
        
