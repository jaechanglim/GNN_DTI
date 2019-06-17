import pickle
import collections
from collections import OrderedDict
from gnn import gnn
import time
import numpy as np
import random
from utils import *
import torch.nn as nn
from multiprocessing import Pool
import sys
import time
import os
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
num_epochs = 10000
lr = 0.0001

ngpus = 1
batch_size = 32*ngpus

#with open ('tmp_data', 'rb') as fp:
with open ('../dude_data/data', 'rb') as fp:
    dude_data = pickle.load(fp)
with open ('../pdbbind_data/data', 'rb') as fp:
    pdbbind_data = pickle.load(fp)

dude_gene =  list(OrderedDict.fromkeys([k.split('_')[0] for k in dude_data.keys()]))
random.seed(0)
random.shuffle(dude_gene)
test_dude_gene = ['egfr', 'parp1', 'fnta', 'aa2ar', 'pygm', 'kith', 'met', 'abl1', 'ptn1', 'casp3', 'hdac8', 'grik1', 'kpcb', 'ada', 'pyrd', 'ace', 'aces', 'pgh1', 'aldr', 'kit', 'fa10', 'pa2ga', 'fgfr1', 'cp3a4', 'wee1']
train_dude_gene = [p for p in dude_gene if p not in test_dude_gene]
print (len(train_dude_gene), len(test_dude_gene))
train_dude_active_keys = [k for k in dude_data.keys() if 'CHEMBL' in k and k.split('_')[0] in train_dude_gene]
train_dude_inactive_keys = [k for k in dude_data.keys() if 'CHEMBL' not in k and k.split('_')[0] in train_dude_gene]
test_dude_keys = [k for k in dude_data.keys() if k.split('_')[0] in test_dude_gene]
end = time.time()
for p in test_dude_gene:
    count = 0
    for k in test_dude_keys:
        if p in k and 'CHEMBL' in k:
            count+=1
    print (p, count)

f = open('../pdbbind_refined/result.txt')
lines = f.read().split('\n')[:-1]
f.close()
lines = [l.split() for l in lines]
lines = [l for l in lines if l[0] + '_' + l[-2] in pdbbind_data.keys() and not (2<float(l[-1])<4)]
pdbbind_pdbs = list(OrderedDict.fromkeys([l[0] for l in lines]))
train_pdbbind_pdbs = pdbbind_pdbs[:int(len(pdbbind_pdbs)*0.75)]
test_pdbbind_pdbs = pdbbind_pdbs[int(len(pdbbind_pdbs)*0.75):]


#train_pdbbind_active_keys = [l[0] + '_' + l[-2] for l in lines if l[0] in train_pdbbind_pdbs]
train_pdbbind_active_keys = [l[0] + '_' + l[-2] for l in lines if l[0] in train_pdbbind_pdbs and float(l[-1])<2.0]
train_pdbbind_inactive_keys = [l[0] + '_' + l[-2] for l in lines if l[0] in train_pdbbind_pdbs and float(l[-1])>4.0]

test_pdbbind_active_keys = [l[0] + '_' + l[-2] for l in lines if l[0] in test_pdbbind_pdbs and float(l[-1])<2.0]
test_pdbbind_inactive_keys = [l[0] + '_' + l[-2] for l in lines if l[0] in test_pdbbind_pdbs and float(l[-1])>4.0]
test_pdbbind_keys = test_pdbbind_active_keys + test_pdbbind_inactive_keys



print ('*******************')
print ('num dude data', len(dude_data))
print ('num dude active train', len(train_dude_active_keys))        
print ('num dude inactive train', len(train_dude_inactive_keys))        
print ('num dude test', len(test_dude_keys))        
print ('*******************')
print ('num pdbbind data', len(pdbbind_data))
print ('num pdbbind active train', len(train_pdbbind_active_keys))        
print ('num pdbbind inactive train', len(train_pdbbind_inactive_keys))        
print ('num pdbbind test', len(test_pdbbind_keys))        
cmd = set_cuda_visible_device(ngpus)
os.environ['CUDA_VISIBLE_DEVICES']=cmd[:-1]

model = gnn()
print ('number of parameters : ', sum(p.numel() for p in model.parameters() if p.requires_grad))

for param in model.parameters():
    if param.dim() == 1:
        continue
        nn.init.constant(param, 0)
    else:
        #nn.init.normal(param, 0.0, 0.15)
        nn.init.xavier_normal(param)
model.load_state_dict(torch.load('save/save_215.pt')) 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
max_loss = 10000000.0
loss_fn = nn.CrossEntropyLoss()
#loss_fn = nn.MSELoss()
#loss_fn = nn.BCELoss()
st = time.time()
train_losses = [] 
test_losses1 = [] 
test_losses2 = [] 
train_true = []
train_pred = []
test_true1 = []
test_pred1 = []
test_true2 = []
test_pred2 = []
#for batch in range(1):
w = open('test_dude.txt', 'w')
#for batch in range(10):
model.eval()
for batch in range(int(len(test_dude_keys)/batch_size)+1):
    keys = [random.choice(test_dude_keys) for i in range(batch_size)]
    if batch==int(len(test_dude_keys)/batch_size):
        keys = test_dude_keys[batch_size*batch:]
    else:
        keys = test_dude_keys[batch_size*batch:batch_size*(batch+1)]
                    
    input = [dude_data[k] for k in keys]
    affinity = [1 if 'CHEMBL' in k else 0 for k in keys]
    affinity = create_var(torch.from_numpy(np.array(affinity)).float())
    affinity = torch.stack((affinity, 1-affinity), -1)
    
    data = preprocessor(input, device)

    with torch.no_grad(): 
        pred = model.test_model(data)
    loss = torch.mean(-affinity*torch.log(pred))

    #loss = (create_var(torch.from_numpy(train_data[batch][1]).float())-pred).pow(2).mean()
    #loss.backward()
    #optimizer.step()

    test_losses1.append(loss.data.cpu().numpy())
    test_true1.append(affinity.data.cpu().numpy())
    test_pred1.append(pred.data.cpu().numpy())
    pred = pred.data.cpu().numpy()
    affinity = affinity.data.cpu().numpy()
    
    for i in range(len(keys)):
        w.write(keys[i] + '\t' + str(affinity[i][0])  +'\t' + str(pred[i,0]) +'\t' + str(pred[i,1]) + '\n')
w.close()

w = open('test_pdbbind.txt', 'w')
#for batch in range(10):
for batch in range(int(len(test_pdbbind_keys)/batch_size)+1):
#for batch in range(int(len(test_keys)/batch_size)):
    if batch==int(len(test_pdbbind_keys)/batch_size):
        keys = test_pdbbind_keys[batch_size*batch:]
    else:    
        keys = test_pdbbind_keys[batch_size*batch:batch_size*(batch+1)]
    input = [pdbbind_data[k] for k in keys]
    affinity = [1 if k in test_pdbbind_active_keys else 0 for k in keys]
    affinity = create_var(torch.from_numpy(np.array(affinity)).float())
    affinity = torch.stack((affinity, 1-affinity), -1)
    
    data = preprocessor(input, device)

    with torch.no_grad(): 
        pred = model.test_model(data)
    loss = torch.mean(-affinity*torch.log(pred))

    #loss = (create_var(torch.from_numpy(train_data[batch][1]).float())-pred).pow(2).mean()
    #loss.backward()
    #optimizer.step()

    test_losses2.append(loss.data.cpu().numpy())
    test_true2.append(affinity.data.cpu().numpy())
    test_pred2.append(pred.data.cpu().numpy())
    pred = pred.data.cpu().numpy()
    affinity = affinity.data.cpu().numpy()
    
    for i in range(len(keys)):
        if keys[i] in test_pdbbind_active_keys:
            w.write(keys[i] + '_active\t' + str(affinity[i][0]) + '\t' + str(pred[i,0]) +'\t' + str(pred[i,1]) + '\n')
        else:
            w.write(keys[i] + '_inactive\t' + str(affinity[i][0]) + '\t' + str(pred[i,0]) +'\t' + str(pred[i,1]) + '\n')
w.close()

