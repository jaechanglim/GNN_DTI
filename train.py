import pickle
from gnn import gnn
import time
import numpy as np
import random
from utils import *
import torch.nn as nn
import time
import os
from sklearn.metrics import roc_auc_score
import argparse
import time
now = time.localtime()
s = "%04d-%02d-%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
print (s)

parser = argparse.ArgumentParser()
parser.add_argument("--lr", help="learning rate", type=float, default = 0.0001)
parser.add_argument("--epoch", help="epoch", type=int, default = 10000)
parser.add_argument("--ngpu", help="number of gpu", type=int, default = 1)
parser.add_argument("--batch_size", help="batch_size", type=int, default = 32)
parser.add_argument("--n_graph_layer", help="number of GNN layer", type=int, default = 4)
parser.add_argument("--d_graph_layer", help="dimension of GNN layer", type=int, default = 140)
parser.add_argument("--n_FC_layer", help="number of FC layer", type=int, default = 4)
parser.add_argument("--d_FC_layer", help="dimension of FC layer", type=int, default = 128)
parser.add_argument("--dude_data_fpath", help="file path of dude data", type=str)
parser.add_argument("--pdbbind_data_fpath", help="file path of pdbbind data", type=str)
parser.add_argument("--save_dir", help="save directory of model parameter", type=str, default = './save/')
parser.add_argument('--no_linear_interpolation', dest='no_linear_interpolation', action='store_true')
parser.add_argument('--CDO', dest='CDO', action='store_true')
parser.add_argument("--CDO_l", help="weight regularizer", type=float, default = 1e-4)
parser.add_argument("--CDO_N", help="dropout regularizer", type=float, default = 1e6)
parser.add_argument("--initial_mu", help="initial value of mu", type=float, default = 4.0)
parser.add_argument("--initial_dev", help="initial value of dev", type=float, default = 1.0)
parser.add_argument("--dropout_rate", help="dropout_rate", type=float, default = 0.0)
parser.add_argument("--GNN", help="which graph neural network will be used", type=str, default='GGNN', choices=['GGNN', 'GConv', 'GConv_gate', 'GAT_gate'])
parser.add_argument("--key_dir", help="key directory", type=str, default='keys')
args = parser.parse_args()
print(f"""\
lr : {args.lr}
epoch : {args.epoch}
ngpu : {args.ngpu}
batch_size : {args.batch_size}
n_graph_layer : {args.n_graph_layer}
d_graph_layer : {args.d_graph_layer}
n_FC_layer : {args.n_FC_layer}
d_FC_layer : {args.d_FC_layer}
dude_data_fpath : {args.dude_data_fpath}
pdbbind_data_fpath : {args.pdbbind_data_fpath}
save_dir : {args.save_dir}
no_linear_interpolation : {args.no_linear_interpolation}
CDO : {args.CDO}
CDO_l : {args.CDO_l}
CDO_N : {args.CDO_N}
initial_mu : {args.initial_mu}
initial_dev : {args.initial_dev}
dropout_rate : {args.dropout_rate}
GNN : {args.GNN}
key_dir : {args.key_dir}
""")
#hyper parameters
num_epochs = args.epoch
lr = args.lr
ngpu = args.ngpu
batch_size = args.batch_size
batch_size_per_gpu = batch_size//ngpu
dude_data_fpath = args.dude_data_fpath
pdbbind_data_fpath = args.pdbbind_data_fpath
no_linear_interpolation = args.no_linear_interpolation
save_dir = args.save_dir
#make save dir if it doesn't exist
if not os.path.isdir(save_dir):
    os.system('mkdir ' + save_dir)

print ('linear interpolation will be used') if not no_linear_interpolation else print ('linear interpolation will not be used')
#read data. data is stored in format of dictionary. Each key has information about protein-ligand complex.
with open (dude_data_fpath, 'rb') as fp:
    dude_data = pickle.load(fp)
with open (pdbbind_data_fpath, 'rb') as fp:
    pdbbind_data = pickle.load(fp)

#read keys of training set and test set
with open(args.key_dir+'/train_dude_gene.pkl', 'rb') as fp:
    train_dude_gene = pickle.load(fp)
with open(args.key_dir+'/test_dude_gene.pkl', 'rb') as fp:
    test_dude_gene = pickle.load(fp)
with open(args.key_dir+'/train_dude_active_keys.pkl', 'rb') as fp:
    train_dude_active_keys = pickle.load(fp)
with open(args.key_dir+'/train_dude_inactive_keys.pkl', 'rb') as fp:
    train_dude_inactive_keys = pickle.load(fp)
with open(args.key_dir+'/test_dude_keys.pkl', 'rb') as fp:
    test_dude_keys = pickle.load(fp)
with open(args.key_dir+'/train_pdbbind_active_keys.pkl', 'rb') as fp:
    train_pdbbind_active_keys = pickle.load(fp)
with open(args.key_dir+'/train_pdbbind_inactive_keys.pkl', 'rb') as fp:
    train_pdbbind_inactive_keys = pickle.load(fp)
with open(args.key_dir+'/test_pdbbind_keys.pkl', 'rb') as fp:
    test_pdbbind_keys = pickle.load(fp)
with open(args.key_dir+'/test_pdbbind_active_keys.pkl', 'rb') as fp:
    test_pdbbind_active_keys = pickle.load(fp)
with open(args.key_dir+'/test_pdbbind_inactive_keys.pkl', 'rb') as fp:
    test_pdbbind_inactive_keys = pickle.load(fp)


#print simple statistics about dude data and pdbbind data
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

#initialize model
cmd = set_cuda_visible_device(ngpu)
os.environ['CUDA_VISIBLE_DEVICES']=cmd[:-1]
model = gnn(args)
print ('number of parameters : ', sum(p.numel() for p in model.parameters() if p.requires_grad))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = initialize_model(model, device)

#optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#loss function
loss_fn = nn.BCELoss()

for epoch in range(num_epochs):
    st = time.time()
    #collect losses of each iteration
    train_losses = [] 
    test_losses1 = [] 
    test_losses2 = [] 

    #collect true label of each iteration
    train_true = []
    test_true1 = []
    test_true2 = []
    
    #collect predicted label of each iteration
    train_pred = []
    test_pred1 = []
    test_pred2 = []
    
    model.train()
    for batch in range(3000):
        model.zero_grad()
        
        #sample active and inactive keys 1:1 for dude data and pdbbind
        #ratio of pdbbind data set and dude set is 1:1
        keys1 = [train_dude_active_keys[np.random.randint(0, len(train_dude_active_keys))] for i in range(int(batch_size/2))]
        keys2 = [train_pdbbind_active_keys[np.random.randint(0, len(train_pdbbind_active_keys))] for i in range(int(batch_size/2))]
        keys3 = [train_dude_inactive_keys[np.random.randint(0, len(train_dude_inactive_keys))] for i in range(int(batch_size/2))]
        keys4 = [train_pdbbind_inactive_keys[np.random.randint(0, len(train_pdbbind_inactive_keys))] for i in range(int(batch_size/2))]
 
        #get corresponding data for each key
        active_input1 = [dude_data[k] for k in keys1]
        active_input2 = [pdbbind_data[k] for k in keys2]
        inactive_input1 = [dude_data[k] for k in keys3]
        inactive_input2 = [pdbbind_data[k] for k in keys4]
        
        #convert raw data of initial node state, adjacency matrix, distance matrix and indice to tensor
        active_data = preprocessor(active_input1+active_input2, device)
        inactive_data = preprocessor(inactive_input1+inactive_input2, device)
        
        #train neural network
        if no_linear_interpolation:
            pred, regularization = model.train_model(active_data, inactive_data, None)
            affinity = create_var(torch.from_numpy(np.array([1.0 for i in range(batch_size)] + [0.0 for i in range(batch_size)])))
            affinity = affinity.float()
        else:
            #randomly sample mixing ratio between active graph and inactive graph
            mixing_ratio = create_var(torch.from_numpy(np.random.uniform(0.0,1.0,batch_size))).float()
            pred, regularization = model.train_model(active_data, inactive_data, mixing_ratio)
            affinity = mixing_ratio

        loss = loss_fn(pred, affinity) + regularization
        loss.backward()
        optimizer.step()
        
        #collect loss, true label and predicted label
        train_losses.append(loss.data.cpu().numpy())
        train_true.append(affinity.data.cpu().numpy())
        train_pred.append(pred.data.cpu().numpy())
    
    model.eval()
    for batch in range(500):
        keys = [test_dude_keys[np.random.randint(0, len(test_dude_keys))] for i in range(batch_size)]
        input = [dude_data[k] for k in keys]
        affinity = [1 if 'CHEMBL' in k else 0 for k in keys]
        affinity = create_var(torch.from_numpy(np.array(affinity)).float())
        
        data = preprocessor(input, device)

        with torch.no_grad(): 
            pred = model.test_model(data)
        loss = loss_fn(pred, affinity)

        test_losses1.append(loss.data.cpu().numpy())
        test_true1.append(affinity.data.cpu().numpy())
        test_pred1.append(pred.data.cpu().numpy())
        
    for batch in range(int(len(test_pdbbind_keys)/batch_size)+1):
        if batch==int(len(test_pdbbind_keys)/batch_size):
            keys = test_pdbbind_keys[batch_size*batch:]
        else:    
            keys = test_pdbbind_keys[batch_size*batch:batch_size*(batch+1)]
        input = [pdbbind_data[k] for k in keys]
        affinity = [1 if k in test_pdbbind_active_keys else 0 for k in keys]
        affinity = create_var(torch.from_numpy(np.array(affinity)).float())
        
        data = preprocessor(input, device)

        with torch.no_grad(): 
            pred = model.test_model(data)
        loss = loss_fn(pred, affinity)

        test_losses2.append(loss.data.cpu().numpy())
        test_true2.append(affinity.data.cpu().numpy())
        test_pred2.append(pred.data.cpu().numpy())
        
    train_losses = np.mean(np.array(train_losses))
    test_losses1 = np.mean(np.array(test_losses1))
    test_losses2 = np.mean(np.array(test_losses2))

    train_true = np.concatenate(train_true, 0)
    train_pred = np.concatenate(train_pred, 0)
    test_true1 = np.concatenate(test_true1, 0)
    test_pred1 = np.concatenate(test_pred1, 0)
    test_true2 = np.concatenate(test_true2, 0)
    test_pred2 = np.concatenate(test_pred2, 0)
    #train_roc = roc_auc_score(train_true, train_pred) 
    test_roc1 = roc_auc_score(test_true1, test_pred1) 
    test_roc2 = roc_auc_score(test_true2, test_pred2) 
    end = time.time()
    print ("%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f" \
    %(epoch, train_losses, test_losses1, test_losses2, test_roc1, test_roc2, end-st))
    name = save_dir + '/save_'+str(epoch)+'.pt'
    torch.save(model.state_dict(), name)
