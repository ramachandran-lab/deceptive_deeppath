from staintools import stain_normalizer
from skimage import io, transform
import numpy as np
import pandas as pd
import glob
import os 
import sys
import argparse
import pickle 
import warnings

warnings.filterwarnings('error')

def read_macenko(macenko_pkl):
    macenko_file = open(macenko_pkl,"rb")
    macenko = pickle.load(macenko_file)
    macenko_file.close()

    return macenko

def read_image(f):
    image = io.imread(f)
    image = image[:,:,:3]
    image = image.astype(np.uint8)
    return image

def transform_macenko(macenko,tile_dir,tile_size,img_extension):
    files = glob.glob("{}/*.{}".format(tile_dir,img_extension))

    m_out_dir = os.path.join(tile_dir,"macenko") 
    if not os.path.exists(m_out_dir):
        os.makedirs(m_out_dir)

    # iterate through the files
    for i, f in enumerate(files):
        if i % 1000 == 0:
            print("Now normalizing tile %d out of %d"%(i+1,len(files)))
        
        # only normalize if output doesn't already exist
        image_basename = os.path.basename(f)
        image_outname = os.path.join(m_out_dir,image_basename)
        if not os.path.exists(image_outname):
            image = read_image(f)

            # skip tiles with the wrong dimensions
            if image.shape != (tile_size,tile_size,3):
                print("Skipping: tile {} with dimensions {}".format(f,image.shape))
            else:
                try:
                    m_image = macenko.transform(image)
                    io.imsave(image_outname,m_image)
                # skip tiles that give a macenko error
                except Exception as e:
                    print("Skipping: tile {} with error {}".format(f,str(e)))   

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--macenko_file", type=str)
    parser.add_argument("--tile_dir", type=str)
    parser.add_argument("--tile_size", type=int)
    parser.add_argument("--img_extension", type=str,default="jpeg")

    args = parser.parse_args()

    print("Now processing directory",args.tile_dir)

    # get macenko stain matrix
    macenko = read_macenko(args.macenko_file)

    # transform all tiles and write out normalized tiles
    transform_macenko(macenko,args.tile_dir,args.tile_size,args.img_extension)
    
    print("Done")