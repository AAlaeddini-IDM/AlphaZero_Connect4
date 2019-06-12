#!/usr/bin/env python

from alpha_net_c4 import ConnectNet, train
import os
import pickle
import numpy as np
import torch

def train_chessnet(net_to_train="c4_current_net_trained2_iter5.pth.tar",save_as="c4_current_net_trained2_iter6.pth.tar"):
    # gather data
    datasets = []
    data_path = "./datasets/iter6/"
    for idx,file in enumerate(os.listdir(data_path)):
        filename = os.path.join(data_path,file)
        with open(filename, 'rb') as fo:
            datasets.extend(pickle.load(fo, encoding='bytes'))
   
    
    '''
    data_path = "./datasets/iter0/"
    datasets = []
    with open(os.path.join(data_path,"dataset_cpu5_7"), 'rb') as fo:
            datasets.extend(pickle.load(fo, encoding='bytes'))
    with open(os.path.join(data_path,"dataset_cpu5_12"), 'rb') as fo:
            datasets.extend(pickle.load(fo, encoding='bytes')) 
    '''
    datasets = np.array(datasets)
    
    # train net
    net = ConnectNet()
    cuda = torch.cuda.is_available()
    if cuda:
        net.cuda()
    current_net_filename = os.path.join("./model_data/",\
                                    net_to_train)
    checkpoint = torch.load(current_net_filename)
    net.load_state_dict(checkpoint['state_dict'])
    train(net, datasets, epoch_start=0, epoch_stop=200, cpu=0)
    # save results
    torch.save({'state_dict': net.state_dict()}, os.path.join("./model_data/",\
                                    save_as))

if __name__=="__main__":
    train_chessnet()