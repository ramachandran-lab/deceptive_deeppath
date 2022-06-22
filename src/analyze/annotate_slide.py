import pandas as pd 
import xml.etree.ElementTree as ET
from scipy import misc
import math
import cv2
import glob 
import numpy as np 
import sys
import matplotlib.pyplot as plt 
import seaborn as sns
import argparse
import os
from PIL import Image, ImageDraw
from matplotlib import cm
import openslide
from matplotlib.colors import LinearSegmentedColormap
def rgb_int2tuple(rgbint):
    return (rgbint // 256 // 256 % 256, rgbint // 256 % 256, rgbint % 256)

def get_coordinates(xml, ds, color="65535"):
    xml_tree = ET.parse(xml)
    xml_root = xml_tree.getroot()
    anno_coordinates = []
    colors = []
    ids = []
    count = 0
    for annotation in xml_root.iter('Annotation'):
        # print(annotation.get("LineColor"))
        # if annotation.get("LineColor") == color:
        if True:
            for region in annotation.iter('Region'):
                coordinates = []
                for node in region.getiterator():
                    if node.tag == 'Vertex':
                        x, y = float(float(node.get("X")))/float(ds), float(float(node.get("Y")))/float(ds) # normalizing by zoom level
                        coordinates.append([x,y])
                anno_coordinates.append(coordinates)
                colors.append(annotation.get("LineColor"))
                ids.append(region.get("Text"))

    return anno_coordinates, colors, ids

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--slide", type=str)
    parser.add_argument("--xml", type=str)
    parser.add_argument("--svs", type=str)
    parser.add_argument("--level", type=int,default=2)
    parser.add_argument("--out_dir", type=str, default=".")

    args = parser.parse_args()


    # add text
    from PIL import ImageFont
    font = ImageFont.truetype("/usr/share/fonts/abattis-cantarell/Cantarell-Bold.otf", 50)

    # read svs as well
    slide = openslide.OpenSlide(args.svs)
    img = slide.read_region((0,0),args.level,(slide.level_dimensions[args.level][0],slide.level_dimensions[args.level][1])).convert("RGB")
    xml_ds = int(slide.level_downsamples[args.level])
    anno_coordinates, colors, ids = get_coordinates(args.xml,xml_ds)
    for coords,c,i in zip(anno_coordinates,colors,ids):
        a = tuple([tuple(x) for x in coords])
        draw = ImageDraw.Draw(img)
        draw.line(a,fill="#1a2e5b",width=10,joint="curve")
        draw.text(tuple(coords[0]),"{}".format(i),(255,255,255),font=font)

    img.save(os.path.join(args.out_dir,"{}_slide.png".format(args.slide)))
