import pandas as pd 
import glob 
import numpy as np 
import sys
import argparse
import os, shutil
import matplotlib.pyplot as plt
import cv2
import scipy.io
from PIL import Image

def gen_gradcam_img(gradient):
    cmap = plt.cm.get_cmap("Greys")

    grad_img = Image.fromarray((cmap(gradient) *255).astype(np.uint8))

    return grad_img

def process_gradcam(file_name):
    gradient = np.load(file_name)
    # gradient = np.abs(gradient)
    gradient = np.clip(gradient,gradient.mean() - 3 * gradient.std(), gradient.mean() + 3 * gradient.std())

    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient = gradient.mean(axis=2)

    # gradient = gradient.flatten()
    return gradient


input_dir, output_dir = sys.argv[1:]

files = glob.glob("{}/*.npy".format(input_dir))
for file_name in files:
    gradient = process_gradcam(file_name)
    grad_img = gen_gradcam_img(gradient)

    output = "{}/{}".format(output_dir,os.path.basename(file_name[:-4]))
    grad_img.save(output)