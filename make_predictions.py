import numpy as np
import torch
import torch.nn as nn

import torch.utils.data.dataloader as Dataloader
from model import Net, Net_2, Net_2_cpu
from Dataset import *
from argparse import ArgumentParser
import torch.optim as optim
from METRICS import *
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import sklearn
import sklearn.metrics
parser = ArgumentParser(description='Read in device, distance map location, 1d sequence location and label location')
parser.add_argument('--device', default='cpu')
parser.add_argument('--distance_map', default='distance_map')
parser.add_argument('--seq', default='1D_sequence')
parser.add_argument('--label', default='labels')
args = parser.parse_args()


DEVICE = torch.device(args.__dict__['device'])
path = args.__dict__['distance_map']
path_feat = args.__dict__['seq']
path_ground_truth = args.__dict__['label']
model_path = 'Saved_Model/DistDom.pth'
nThreads = 4
dataset = my_dataset()
dataset.initialize(path,path_feat,path_ground_truth)
# print(dataset.__getitem__(1)['ground_truth'].size())
train_set_size = math.ceil(dataset.__len__() * .80)
validation_set_size = math.floor(dataset.__len__() * .20)
# print(train_set_size, validation_set_size)
train_set, validation_set = random_split(dataset, [train_set_size, validation_set_size], generator=torch.Generator().manual_seed(42))

train_loader = torch.utils.data.DataLoader(train_set,batch_size=1,shuffle=None,num_workers=nThreads)
validation_loader = torch.utils.data.DataLoader(validation_set,batch_size=1,shuffle=None,num_workers=nThreads)

test_loader = torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=None,num_workers=nThreads)
#Model define and optmizers
model = Net_2_cpu().to(device=DEVICE)
model.load_state_dict(torch.load(model_path,map_location=DEVICE))
weight_ratio = 1.0
learning_rate = 0.0001
loss_weights = [1.0,weight_ratio]
class_weights = torch.FloatTensor(loss_weights).to(device=DEVICE)
criterion = nn.CrossEntropyLoss().to(device=DEVICE)
optimizer = optim.Adam(model.parameters(),lr=learning_rate,weight_decay = 0)
validation_loss_list = []
validation_precision_list = []
validation_recall_list = []
validation_f1_list = []
validation_mattcoeff_list = []
validation_precision_max = 0
f1_total = 0
recall_total = 0
precision_total = 0
running_prec = []
running_rec = []
running_f1 = []
running_no_skill = 0
pred_list = []
label_list = []
with torch.no_grad():
    for i, data in enumerate(test_loader):


        model.eval()

        dist = data['dist'].to(device=DEVICE)
        seq = data['one_hot_sequence'].to(device=DEVICE)
        label = data['ground_truth'].to(device=DEVICE)
        name = data['name']
        name_feat = data['name_feats']
        print(name)

        dist = dist.float()
        seq = torch.transpose(seq, 1, 2)
        seq = seq.float()

        label = label.long()

        output_final = model(dist, seq)
        output_final = torch.squeeze(output_final, -1)
        loss = criterion(output_final, label)
        running_validation_precision = calculate_precision(label, torch.argmax(output_final, dim=1))
        running_validation_recall = calculate_recall(label, torch.argmax(output_final, dim=1))
        running_validation_F1 = calculate_F1(label, torch.argmax(output_final, dim=1))
        running_validation_mattcoeff = calculate_mattcoeff(label, torch.argmax(output_final, dim=1))

        x, y, z = calculate_precision_recall_by_threshold(label,output_final,threshold=0.03)
        print('Name: {0} Precision: {1} Recall: {2} F1: {3} for 0.03 threshold'.format(str(name[0].split('.txt')[0]),x,y,z))
        pred_np, pred_np_prob = save_prediction(label,output_final,threshold=0.03)
        np.savetxt(os.path.join('Saved_Predictions',str(name[0].split('.txt')[0])+'.txt'),pred_np)


        running_validation_loss = loss.item()
        f1_total += running_validation_F1
        precision_total += running_validation_precision
        recall_total += running_validation_recall
        pred_list.append(output_final)
        label_list.append(label)


