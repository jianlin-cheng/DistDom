import torch
import torch.utils.data as data
import os
import glob
import math
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import random_split




class my_dataset(data.Dataset):
    def initialize(self, distance_map, feat,labels):
        # assert os.path.isdir(root_path), '%s is not a valid directory' % root_path

        # List all JPEG images
        distance_maps_path = os.path.join(distance_map, '*')
        feats_path = os.path.join(feat,'*.feat')
        labels_path = os.path.join(labels,'*')
        self.distance_maps = sorted(glob.glob(distance_maps_path))
        self.feats = sorted(glob.glob(feats_path))
        self.labels = sorted(glob.glob(labels_path))
        self.size = len(self.distance_maps)
        self.size_feats = len(self.feats)
        self.size_labels = len(self.labels)

        # Define a transform to be applied to each image
        # Here we resize the image to 224X224 and convert it to Tensor
        self.transform = transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.485), (0.229))])
        self.transform_one_hot = transforms.Compose([ transforms.ToTensor()])

    def __getitem__(self, index):
        # Loads a single data point from the dataset
        # Supporting integer indexing in range from 0 to the size of the dataset (exclusive)

        path = self.distance_maps[index % self.size]
        path_feats = self.feats[index % self.size_feats]
        path_labels = self.labels[index % self.size_labels]
        name = path.split('/')[-1]
        name_feats = path_feats.split('/')[-1]
        name_labels = path_labels.split('/')[-1]
        # label = int(((path.split('/')[-1]).split('.')[0])=='cat')   # Ectract label from the filename
        # img = Image.open(path).convert('RGB')                       # Load the image and convert to RGB
        dist = np.loadtxt(path)
        # dist = np.expand_dims(dist,axis=1)
        # with open(path_fasta) as handle:
        #   for record in SeqIO.parse(handle, "fasta"):
        #     one_hot_sequence = one_hot_encode(record.seq)

        # dist = np.moveaxis(dist,-1,0)
        # img = self.transform(img)                                   # Apply the defined transform
        dist = self.transform(dist)
        # print(len(one_hot_sequence))
        # one_hot_sequence = pad_sequence(one_hot_sequence)
        # one_hot_sequence = self.transform_one_hot(one_hot_sequence)
        one_hot_sequence = np.loadtxt(path_feats)
        # one_hot_sequence = np.swapaxes(one_hot_sequence,0,1)
        one_hot_sequence = self.transform(one_hot_sequence)

        ground_truth = torch.load(path_labels)
        return {'dist': dist, 'one_hot_sequence': one_hot_sequence , 'ground_truth': ground_truth, 'name': name, 'name_feats': name_feats, 'name_labels': name_labels}

    def __len__(self):
        # Provides the size of the dataset

        return self.size


# path = 'distance_maps'
# path_feat = 'feat_4073'
# path_ground_truth = 'labels_4073'
#
# dataset = my_dataset()
# dataset.initialize(path,path_feat,path_ground_truth)
# print(dataset.__getitem__(1)['ground_truth'].size())
