# code taken from https://github.com/kazuto1011/grad-cam-pytorch
# with very minor changes 

from __future__ import print_function

import copy
import os.path as osp

import click
import cv2
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
import pandas as pd 
 
import argparse
import sys, os
sys.path.append("scripts")

from utils.helper_funcs import *

from utils.gradcam import (
    BackPropagation,
    Deconvnet,
    GradCAM,
    GuidedBackPropagation,
    occlusion_sensitivity,
)

def load_images(image_paths):
    images = []
    raw_images = []
    print("Images:")
    for i, image_path in enumerate(image_paths):
        print("\t#{}: {}".format(i, image_path))
        image, raw_image = preprocess(image_path)
        images.append(image)
        raw_images.append(raw_image)
    return images, raw_images

def preprocess(image_path):

    raw_image = cv2.imread(image_path)
    # raw_image = cv2.resize(raw_image, (512,) * 2)

    image = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(raw_image[..., ::-1].copy())
    return image, raw_image

def gen_gradcam(model_name, target_layer, df, output_dir, checkpoint_path, slide, clabel):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = get_model(model_name,True,1)   

    model.load_state_dict(torch.load(checkpoint_path,map_location="cpu"))
    model.to(device)
    model.eval()

    target_class = 0 

    # Images
    df = df.sample(frac=1)
    image_paths = df['path'].values
    image_names = [os.path.basename(x).split(".")[0] for x in df['path'].values]

    labels = df['label'].values
    # images, raw_images = load_images(image_paths)
    # images = torch.stack(images).to(device)

    print("Guided Grad-CAM:")
    ids_ = torch.LongTensor([[target_class]]).to(device)

    gcam = GradCAM(model=model)
    gbp = GuidedBackPropagation(model=model)


    count = 0
    for image_path, image_name, label in zip(image_paths,image_names,labels):
        count += 1
        if count % 99 == 0:
            print("processing image {}/{}".format(count+1,len(image_paths)))
        
        file_name = "{}_{}_guided-gradcam-{}".format(
                        image_name, label, target_layer)

        if not os.path.exists(os.path.join(output_dir,slide,file_name)):
            image, raw_image = preprocess(image_path)
            image = image.unsqueeze(0).to(device)

            probs, ids = gcam.forward(image,clabel)
            _ = gbp.forward(image,clabel)

            # Guided Backpropagation
            gbp.backward(ids=ids_)
            gradients = gbp.generate()

            # Grad-CAM
            gcam.backward(ids=ids_)
            regions = gcam.generate(target_layer=target_layer)

            # Guided Grad-CAM
            save_gradient(
                filename=osp.join(
                    output_dir,slide,
                    file_name,
                ),
                raw_gradient=torch.mul(regions, gradients)[0]
            )

def save_gradient(filename, raw_gradient):
    gradient = raw_gradient.cpu().numpy().transpose(1, 2, 0)
    np.save(filename,gradient)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--target_layer", type=str)
    parser.add_argument("--slide", type=str)
    parser.add_argument("--clabel", type=int)
    args = parser.parse_args()
    print(args)

    df = pd.read_csv(args.results)

    df['slide'] = df['path'].apply(lambda x: x.split("/")[3].split("_files")[0])
    
    if args.slide is None:
        slides = np.unique(df['slide'])
    else:
        slides = [args.slide]

    for slide in slides:
        tmp_df = df[df['slide'] == slide]

        if not os.path.exists(os.path.join(args.output_dir,slide)):
            os.makedirs(os.path.join(args.output_dir,slide))

        print("slide:",slide)
        gen_gradcam(args.model_name, args.target_layer, tmp_df, args.output_dir, args.checkpoint_path, slide, args.clabel)

