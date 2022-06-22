# import stuff
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
from ast import literal_eval
import sys

sys.path.append("scripts")
from utils.helper_funcs import *
from torchsampler import ImbalancedDatasetSampler

# method was adapted from https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
def test_model(model, checkpoint_path, dataloader, results_name, model_name):

    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path,map_location="cpu"))
    else:
        print("no checkpoint loaded")
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
       
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    torch.set_grad_enabled(False)

    all_paths = []
    all_preds = []
    all_losses = []
    all_labels = []

    # Iterate over data.
    batch_time = time.time()

    for i, (images,labels,paths) in enumerate(dataloader):
        if (i+1) % 100 == 0:
            print("Evaluating batch ",i+1)
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        outputs = model(images)

        losses = criterion(outputs, labels).detach().cpu().numpy().flatten()
        preds = outputs.detach().cpu().numpy().flatten()
        
        all_paths.extend(paths)
        all_losses.extend(list(losses))
        all_preds.extend(list(preds))
        all_labels.extend(list(labels.cpu().numpy().flatten()))

        df = pd.DataFrame([x for x in zip(all_paths,all_labels,all_preds,all_losses)])
        df.columns = ['path','label','logit','loss']

        df.to_csv(results_name,index=False)

def prep_data(slides_file,tiles,num_workers,batch_size,pretrained,balanced):
    print("\n" + '-' * 20)
    print("Creating datasets")
    print('-' * 20)

    tiles_df = pd.read_csv(tiles)

    tiles_df['slide'] = tiles_df['slide'].apply(lambda x: str(x))

    eval_slides = [str(x) for x in pd.read_csv(slides_file,header=None)[0].values]
    slide_df = tiles_df[tiles_df['slide'].isin(eval_slides)]

    u_slides = np.unique(slide_df['slide'])

    data_transforms = get_data_transforms(pretrained)

    dataset = TileDataset(data_transforms['val'],slide_df)
    if balanced:
        sampler = ImbalancedDatasetSampler(dataset,callback_get_label=get_label) 
    else:
        sampler = None
    dataloader = DataLoader(dataset,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 sampler=sampler)

    print("There are {} tiles from {} slides to be evaluated".format(len(slide_df),len(u_slides)))
    return dataloader


def print_arguments(args):
    print("Input arguments:")
    keys = sorted(vars(args))
    for key in keys:
        print("{:16} {}".format(key, vars(args)[key]))
    print("*** End Input arguments ***")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--slides", type=str, default=None)
    parser.add_argument("--tiles", type=str)
    parser.add_argument("--outdir", type=str,default="results")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--run_dir", type=str)
    parser.add_argument("--checkpoint", type=int, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--phase", type=str, default="eval")
    parser.add_argument("--baseline", action="store_true", default=False)
    parser.add_argument("--balanced", action="store_true", default=False)
    args = parser.parse_args()


    f = open("{}/arguments.csv".format(args.run_dir),"r")
    run_args = literal_eval(f.readlines()[0])
    f.close()

    print_arguments(args)
    print()

    outdir = "{}/{}".format(args.run_dir,args.outdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)


    # get the data
    dataloader = prep_data(args.slides,args.tiles,args.num_workers,args.batch_size,run_args['pretrained'],args.balanced)


    # initialize model
    model = get_model(run_args['model'],run_args['pretrained'],run_args['num_classes'])   

    # test model
    if args.baseline:
        checkpoint = "step-0"
        print("Now evaluating imagenet weights")

        results_name = "{}/{}_preds_ckp-{}.csv".format(outdir,args.phase,checkpoint)
        if os.path.exists(results_name):
            print("already saved")
        else:
            test_model(model, None, dataloader, results_name, run_args['model'])

    elif args.checkpoint_path is None:
        checkpoint_paths = glob.glob("{}/model*00.pt".format(args.run_dir))
        np.random.shuffle(checkpoint_paths)
        print("Testing {} checkpoints".format(len(checkpoint_paths)))
        for checkpoint_path in checkpoint_paths:
            checkpoint = checkpoint_path.split("_")[-1][:-3]
            print("Now evaluating checkpoint ",checkpoint_path)
            results_name = "{}/{}_preds_ckp-{}.csv".format(outdir,args.phase,checkpoint)
            if os.path.exists(results_name):
                print("already saved")
            else:
                test_model(model, checkpoint_path, dataloader, results_name, run_args['model'])

    else:
        if os.path.islink(args.checkpoint_path):
            mod = args.checkpoint_path.split("_")[-1][:-3]
            args.checkpoint_path = os.readlink(args.checkpoint_path)
        checkpoint = args.checkpoint_path.split("_")[-1][:-3]
        print("Now evaluating checkpoint ",args.checkpoint_path)

        results_name = "{}/{}_preds_ckp-{}.csv".format(outdir,args.phase,checkpoint)
        print(results_name)
        if os.path.exists(results_name):
            print("already saved")
        else:
            test_model(model, args.checkpoint_path, dataloader, results_name, run_args['model'])
            
