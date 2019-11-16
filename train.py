import pickle
from gnn import gnn
import time
import numpy as np
import utils
import torch.nn as nn
import torch
import time
import os
from sklearn.metrics import roc_auc_score
import argparse
import time
from torch.utils.data import DataLoader                                     
from dataset import MolDataset, collate_fn, DTISampler
now = time.localtime()
s = "%04d-%02d-%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
print (s)

parser = argparse.ArgumentParser()
parser.add_argument("--lr", help="learning rate", type=float, default = 0.0001)
parser.add_argument("--epoch", help="epoch", type=int, default = 10000)
parser.add_argument("--ngpu", help="number of gpu", type=int, default = 1)
parser.add_argument("--batch_size", help="batch_size", type=int, default = 32)
parser.add_argument("--num_workers", help="number of workers", type=int, default = 7)
parser.add_argument("--n_graph_layer", help="number of GNN layer", type=int, default = 4)
parser.add_argument("--d_graph_layer", help="dimension of GNN layer", type=int, default = 140)
parser.add_argument("--n_FC_layer", help="number of FC layer", type=int, default = 4)
parser.add_argument("--d_FC_layer", help="dimension of FC layer", type=int, default = 128)
parser.add_argument("--dude_data_fpath", help="file path of dude data", type=str, default='data/')
parser.add_argument("--save_dir", help="save directory of model parameter", type=str, default = './save/')
parser.add_argument("--initial_mu", help="initial value of mu", type=float, default = 4.0)
parser.add_argument("--initial_dev", help="initial value of dev", type=float, default = 1.0)
parser.add_argument("--dropout_rate", help="dropout_rate", type=float, default = 0.0)
parser.add_argument("--train_keys", help="train keys", type=str, default='keys/train_keys.pkl')
parser.add_argument("--test_keys", help="test keys", type=str, default='keys/test_keys.pkl')
args = parser.parse_args()
print (args)

#hyper parameters
num_epochs = args.epoch
lr = args.lr
ngpu = args.ngpu
batch_size = args.batch_size
dude_data_fpath = args.dude_data_fpath
save_dir = args.save_dir

#make save dir if it doesn't exist
if not os.path.isdir(save_dir):
    os.system('mkdir ' + save_dir)

#read data. data is stored in format of dictionary. Each key has information about protein-ligand complex.
with open (args.train_keys, 'rb') as fp:
    train_keys = pickle.load(fp)
with open (args.test_keys, 'rb') as fp:
    test_keys = pickle.load(fp)

#print simple statistics about dude data and pdbbind data
print (f'Number of train data: {len(train_keys)}')
print (f'Number of test data: {len(test_keys)}')

#initialize model
if args.ngpu>0:
    cmd = utils.set_cuda_visible_device(args.ngpu)
    os.environ['CUDA_VISIBLE_DEVICES']=cmd[:-1]
model = gnn(args)
print ('number of parameters : ', sum(p.numel() for p in model.parameters() if p.requires_grad))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = utils.initialize_model(model, device)

#train and test dataset
train_dataset = MolDataset(train_keys, args.dude_data_fpath)
test_dataset = MolDataset(test_keys, args.dude_data_fpath)
num_train_chembl = len([0 for k in train_keys if 'CHEMBL' in k])
num_train_decoy = len([0 for k in train_keys if 'CHEMBL' not in k])
train_weights = [1/num_train_chembl if 'CHEMBL' in k else 1/num_train_decoy for k in train_keys]
train_sampler = DTISampler(train_weights, len(train_weights), replacement=True)                     
train_dataloader = DataLoader(train_dataset, args.batch_size, \
     shuffle=False, num_workers = args.num_workers, collate_fn=collate_fn,\
     sampler = train_sampler)
test_dataloader = DataLoader(test_dataset, args.batch_size, \
     shuffle=False, num_workers = args.num_workers, collate_fn=collate_fn)

#optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#loss function
loss_fn = nn.BCELoss()

for epoch in range(num_epochs):
    st = time.time()
    #collect losses of each iteration
    train_losses = [] 
    test_losses = [] 

    #collect true label of each iteration
    train_true = []
    test_true = []
    
    #collect predicted label of each iteration
    train_pred = []
    test_pred = []
    
    model.train()
    for i_batch, sample in enumerate(train_dataloader):
        model.zero_grad()
        H, A1, A2, Y, V, keys = sample 
        H, A1, A2, Y, V = H.to(device), A1.to(device), A2.to(device),\
                            Y.to(device), V.to(device)
        
        #train neural network
        pred = model.train_model((H, A1, A2, V))

        loss = loss_fn(pred, Y) 
        loss.backward()
        optimizer.step()
        
        #collect loss, true label and predicted label
        train_losses.append(loss.data.cpu().numpy())
        train_true.append(Y.data.cpu().numpy())
        train_pred.append(pred.data.cpu().numpy())
        #if i_batch>10 : break
    
    model.eval()
    for i_batch, sample in enumerate(test_dataloader):
        model.zero_grad()
        H, A1, A2, Y, V, keys = sample 
        H, A1, A2, Y, V = H.to(device), A1.to(device), A2.to(device),\
                          Y.to(device), V.to(device)
        
        #train neural network
        pred = model.train_model((H, A1, A2, V))

        loss = loss_fn(pred, Y) 
        
        #collect loss, true label and predicted label
        test_losses.append(loss.data.cpu().numpy())
        test_true.append(Y.data.cpu().numpy())
        test_pred.append(pred.data.cpu().numpy())
        #if i_batch>10 : break
        
    train_losses = np.mean(np.array(train_losses))
    test_losses = np.mean(np.array(test_losses))
    
    train_pred = np.concatenate(np.array(train_pred), 0)
    test_pred = np.concatenate(np.array(test_pred), 0)
    
    train_true = np.concatenate(np.array(train_true), 0)
    test_true = np.concatenate(np.array(test_true), 0)

    train_roc = roc_auc_score(train_true, train_pred) 
    test_roc = roc_auc_score(test_true, test_pred) 
    end = time.time()
    print ("%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f" \
    %(epoch, train_losses, test_losses, train_roc, test_roc, end-st))
    name = save_dir + '/save_'+str(epoch)+'.pt'
    torch.save(model.state_dict(), name)
