from __future__ import print_function, division
import sys
import os
import time
import copy
import configparser
from collections import Counter
import pickle
import random
import shutil
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms,utils
from PIL import Image
import matplotlib.pyplot as plt
import warnings
from itertools import product 
import glob 
import argparse
import pandas as pd

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

sys.path.append("scripts/utils")
from helper_funcs import *
from torchsampler import ImbalancedDatasetSampler

# method was adapted from https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
def train_model(model, model_name, dataloader, criterion, optimizer, device, num_epochs, save_freq, outdir, num_workers):
    perf_history = []
    model.train()  # Set model to training mode

    step = 0
    for epoch in range(1,num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 20)
        
        # Each epoch has a training and validation phase
        since = time.time()

        running_loss = 0.0
        running_corrects = 0
        train_sample_counter = 0

        # Iterate over data.
        batch_time = time.time()

        for i, (images,labels,_) in enumerate(dataloader):

            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(images)
            
            if model_name == "inception":
                outputs = outputs[0]
            
            loss = criterion(outputs, labels)
            preds = torch.round(torch.sigmoid(outputs)).detach()

            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * images.size(0)
            train_sample_counter += images.size(0)
            running_corrects += torch.sum(preds == labels.data)
            running_accuracy = (running_corrects.double()/train_sample_counter).item()

            # per batch info
            if i % (num_workers * 100) == 0 and i != 0:
                batch_time_elapsed = time.time() - batch_time
                print('Step {}, time {:.0f}m {:.0f}s'.format(step, batch_time_elapsed // 60, batch_time_elapsed % 60))
                print('Training accuracy is ',running_accuracy)
                batch_time = time.time()
                
            if step % save_freq == 0 and step != 0:
                torch.save(model.state_dict(),"{}/model_step-{}.pt".format(outdir,step))
                perf_history.append((step,running_loss/train_sample_counter,running_accuracy))
                
                df = pd.DataFrame(perf_history,columns=['step','loss','accuracy'])
                df.to_csv("{}/training_loss.txt".format(outdir),index=False)

                running_loss = 0.0
                running_corrects = 0
                train_sample_counter = 0

            step +=1 



def prep_data(train_slides,tiles,num_workers,batch_size,pretrained):
    print("\n" + '-' * 20)
    print("Creating datasets")
    print('-' * 20)

    tiles_df = pd.read_csv(tiles)
    tiles_df['slide'] = tiles_df['slide'].apply(lambda x: str(x))

    train_df = tiles_df[tiles_df['slide'].isin(train_slides)]
    u_train_slides = np.unique(train_df['slide'])

    data_transforms = get_data_transforms(pretrained)

    dataset = TileDataset(data_transforms['train'],train_df)
    train_sampler = ImbalancedDatasetSampler(dataset,callback_get_label=get_label)

    dataset_size = len(train_sampler)
    dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                sampler=train_sampler,
                                drop_last=True)

    print("There are {} training slides with {} tiles".format(len(u_train_slides),len(train_df)))

    return dataloader 


def print_arguments(args):
    print("Input arguments:")
    keys = sorted(vars(args))
    for key in keys:
        print("{:16} {}".format(key, vars(args)[key]))
    print("*** End Input arguments ***")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_slides", type=str)
    parser.add_argument("--tiles", type=str)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--save_freq", type=int, default=200)

    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("--fold", type=str)
    parser.add_argument("--exp_dir", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--checkpoint_path", type=str)

    args = parser.parse_args()

    outdir = "%s/%s/fold-%s"%(args.exp_dir,args.name,args.fold)

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    elif not args.overwrite:
    	print("Run exists, quitting")
    	sys.exit(0)

    print_arguments(args)

    args_file =  os.path.join(outdir, "arguments.csv")
    f = open(args_file,"w")
    f.write(str(args.__dict__))
    f.close()

    # get the data
    train_slides = [str(x) for x in pd.read_csv(args.train_slides,header=None)[0].values]
    dataloader = prep_data(train_slides,args.tiles,args.num_workers,args.batch_size,args.pretrained)
    
    # initialize model
    model = get_model(args.model,args.pretrained,args.num_classes)   

    if args.checkpoint_path is not None:
        model.load_state_dict(torch.load(args.checkpoint_path,map_location="cpu"))
        print("loading checkpoint ",args.checkpoint_path)

    # define optimizer and loss function
    criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
       
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # train
    print("\n" + '-' * 20)
    print("Training model")
    print('-' * 20)
    
    train_model(model,args.model,dataloader, criterion, optimizer, device, args.num_epochs,args.save_freq,outdir,args.num_workers)


