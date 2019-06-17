import numpy as np
import torch
from torch.autograd import Variable
from scipy import sparse
import os.path
import time
import torch.nn as nn
#from rdkit.Contrib.SA_Score.sascorer import calculateScore
#from rdkit.Contrib.SA_Score.sascorer
#import deepchem as dc

N_atom_features = 28

def create_var(tensor, requires_grad=None): 
    if requires_grad is None: 
        #return Variable(tensor)
        return Variable(tensor).cuda()
    else: 
        return Variable(tensor,requires_grad=requires_grad).cuda()

def ensure_shared_grads(model, shared_model, gpu=False):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None and not gpu:
            return
        elif not gpu:
            shared_param._grad = param.grad
        else:
            if param.grad is None:
                continue
            shared_param._grad = param.grad.cpu()

def preprocessor(data, device):

    #each data of protein-ligand complex has different size.
    #unite the size of tensor as the largest number of atoms by padding zero elements
    
    max_natoms = np.amax(np.array([len(d[0]) for d in data]))
    c_hs = np.zeros((len(data), max_natoms, N_atom_features*2), np.uint8)
    c_adjs1 = np.zeros((len(data),  max_natoms, max_natoms), np.uint8)
    c_adjs2 = np.zeros((len(data),  max_natoms, max_natoms))
    c_valid = np.zeros((len(data),  max_natoms), np.uint8)
    
    for idx, (hs, adj, distance_matrix, n_ligand_atoms) in enumerate(data):
        #hs : initial node state
        #adj : adjacency matrix, note that only ligand-ligand, protein-protein is filled
        #distance_miatrix :  distance matrix, note that only ligand-protein is filled
        #n_ligand_atoms : number of ligand atoms
        adj = np.copy(adj.todense())
        distance_matrix = np.copy(distance_matrix.todense())
        adj += np.eye(len(adj)).astype(np.uint8)
        c_hs[idx, :n_ligand_atoms,:N_atom_features] = hs[:n_ligand_atoms]
        c_hs[idx, n_ligand_atoms:len(adj),N_atom_features:] = hs[n_ligand_atoms:]

        adj1 = np.copy(adj)
        adj1[:n_ligand_atoms,n_ligand_atoms:] = 0
        c_adjs1[idx, :len(adj), :len(adj)] = adj1
        c_adjs2[idx, :len(adj), :len(adj)] = np.copy(distance_matrix)
        c_valid[idx,:n_ligand_atoms] = 1
    
    c_adjs2[c_adjs2<0.00001] = 1e10
    c_hs = create_var(torch.from_numpy(np.array(c_hs))).to(device).float()
    c_adjs1 = create_var(torch.from_numpy(np.array(c_adjs1))).to(device).float()
    c_adjs2 = create_var(torch.from_numpy(np.array(c_adjs2))).to(device).float()
    c_valid = create_var(torch.from_numpy(np.array(c_valid))).to(device).float()
    
    return c_hs, c_adjs1, c_adjs2, c_valid

def set_cuda_visible_device(ngpus):
    import subprocess
    import os
    empty = []
    for i in range(8):
        command = 'nvidia-smi -i '+str(i)+' | grep "No running" | wc -l'
        output = subprocess.check_output(command, shell=True).decode("utf-8")
        #print('nvidia-smi -i '+str(i)+' | grep "No running" | wc -l > empty_gpu_check')
        if int(output)==1:
            empty.append(i)
    if len(empty)<ngpus:
        print ('avaliable gpus are less than required')
        exit(-1)
    cmd = ''
    for i in range(ngpus):        
        cmd+=str(empty[i])+','
    return cmd

def cal_auc(true, pred):    
    from sklearn.metrics import roc_auc_score
    true = np.array([[1,0] if true[i]==0 else [0,1] for i in range(len(true))])
    return roc_auc_score(true, pred)

def cal_R2(true, pred):
    from sklearn.metrics import r2_score
    import scipy
    r2_1 = r2_score(true, pred)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(true, pred)
    r2_2 = r_value*r_value

    return r2_1, r2_2
    
def initialize_model(model, device, load_save_file=False):
    if load_save_file:
        model.load_state_dict(torch.load(load_save_file)) 
    else:
        for param in model.parameters():
            if param.dim() == 1:
                continue
                nn.init.constant(param, 0)
            else:
                #nn.init.normal(param, 0.0, 0.15)
                nn.init.xavier_normal_(param)

    if torch.cuda.device_count() > 1:
      print("Let's use", torch.cuda.device_count(), "GPUs!")
      # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
      model = nn.DataParallel(model)
    model.to(device)
    return model
