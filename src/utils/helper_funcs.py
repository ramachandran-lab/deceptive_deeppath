from PIL import Image
from torchvision import transforms,utils
import torchvision.models as models
import pandas as pd
from torch.utils.data import Dataset
import torch 
import torch.nn as nn
import scipy.io
import numpy as np
from skimage.measure import label, regionprops, find_contours
import cv2 
import random
import glob 

def get_label(dataset,idx):
    return dataset.labels[idx]

def get_subtype_label(dataset,idx):
    labs = dataset.label_array[idx]
    pos_lab_idx = np.where(labs == 1)[0]

    # if none of the labels are pos, just assign to last category
    if len(pos_lab_idx) == 0:
        pos_lab_idx = [5]    
    
    dom_label = np.random.choice(pos_lab_idx)
    return dom_label

class TypeTileDataset(Dataset):
    def __init__(self, tf, df, subtypes=['acinar','lepidic','solid','papillary','micropapillary','mucinous']):
        self.tf = tf
        self.tiles = df['path'].values
        self.labels = {}
        self.subtypes = subtypes 
        for s in subtypes:
            self.labels[s] = df[s].values 
        
        label_array = np.stack([self.labels[s] for s in subtypes])
        self.label_array = label_array.transpose() 


    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        path = self.tiles[idx]

        image = Image.open(path)
        image = self.tf(image)

        targets = []
        for s in self.subtypes:
            targets.append(float(self.labels[s][idx]))

        path = self.tiles[idx]
        targets = torch.tensor(targets)
        return image, targets, path

class MultiTileDataset(Dataset):
    def __init__(self, tf, df):
        self.tf = tf
        self.tiles = df['path'].values
        self.tumor_labels = df['tumor_label'].values
        self.met_labels = df['met_label'].values
        self.kras_labels = df['kras_label'].values

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        path = self.tiles[idx]

        image = Image.open(path)
        image = self.tf(image)               
       
        tumor_target = float(self.tumor_labels[idx])
        met_target = float(self.met_labels[idx])
        kras_target = float(self.kras_labels[idx])

        path = self.tiles[idx]
        return image, tumor_target, met_target, kras_target, path


class TileDataset(Dataset):
    def __init__(self, tf, df):
        self.tf = tf
        self.tiles = df['path'].values
        self.labels = df['label'].values

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        path = self.tiles[idx]

        image = Image.open(path)
        image = self.tf(image)               
       
        target = float(self.labels[idx])
        path = self.tiles[idx]
        return image, target, path


class TileSubDataset(Dataset):
    def __init__(self, tf, df):
        self.tf = tf
        self.tiles = df['path'].values
        self.labels = df['label'].values
        self.subtypes = df['subtype'].values

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        path = self.tiles[idx]

        image = Image.open(path)
        image = self.tf(image)

        target = float(self.labels[idx])
        subtype = float(self.subtypes[idx])
        path = self.tiles[idx]
        return image, target, subtype, path

def get_data_transforms(pretrained):
    data_transforms = {
        'train': [transforms.RandomHorizontalFlip(),
             transforms.RandomVerticalFlip(),
             transforms.ToTensor()],
        'val': [transforms.ToTensor()]}

    if pretrained:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        for phase in ['train','val']:
            data_transforms[phase].append(transforms.Normalize(mean,std))

    for phase in ['train','val']:
        data_transforms[phase] = transforms.Compose(data_transforms[phase])

    return data_transforms


class SubtypeModel(nn.Module):
    def __init__(self,model_name,pretrained,num_classes):
        super(SubtypeModel, self).__init__()
        self.cnn = get_model(model_name,pretrained,num_classes)
        
        self.fc = nn.Linear(self.cnn.fc.in_features+1, num_classes)
        self.cnn.fc = nn.Identity()
 
    def forward(self, image, subtype):
        x1 = self.cnn(image)
        x2 = subtype.float() 
        
        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        return x


def get_model(model_name,pretrained,num_classes):
    if model_name == "resnet18" or model_name == "resnet":
        model = models.resnet18(pretrained)
    elif model_name == "resnet34":
        model = models.resnet34(pretrained)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained)
    elif model_name == "inception":
        model = models.inception_v3(pretrained)
    elif model_name == "shufflenet":
        model = models.shufflenet_v2_x1_0(pretrained)
    else:
        raise NotImplementedError("{} is not implemented".format(model_name))

    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs,num_classes)

    return model


class GCTileDataset(Dataset):
    def __init__(self, tf, df, gc_dir, threshold, shuffle=False, reverse_mask=False, gc_file=None):
        self.tf = tf
        self.tiles = df['path'].values
        self.labels = df['label'].values
        self.shuffle = shuffle
        self.gc_dir = gc_dir
        self.threshold = threshold
        if self.shuffle:
            gc_list = pd.read_csv(gc_file,header=None)[0].values
            self.gc_paths = list(gc_list)
        self.reverse_mask = reverse_mask


    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        path = self.tiles[idx]
        
        image = Image.open(path)
        image = self.tf(image)

        if self.shuffle:
            gc_path = random.choice(self.gc_paths)
        else:
            slide = path.split("/")[3].split("_")[0]
            tile = path.split("/")[-1].split(".")[0]
            gc_path = glob.glob("{}/{}/{}_*.npy".format(self.gc_dir,slide,tile))[0]

        gradient = self.process_gradcam(gc_path)

        # if the gradient has nans, convert to 0
        if np.isnan(np.sum(gradient)):
            gradient[np.isnan(gradient)] = 0 

        gc_mask = np.ones(gradient.shape)

        if self.reverse_mask:
            gc_mask[gradient < self.threshold] = 0
        else:
            gc_mask[gradient >= self.threshold] = 0

        gc_mask = cv2.resize(gc_mask,(256,256))
        final_gc_mask = np.ones(gc_mask.shape)
        final_gc_mask[gc_mask != 1] = 0


        tensor_gc_mask = torch.tensor(final_gc_mask)
        target = float(self.labels[idx])
        path = self.tiles[idx]

        return image, tensor_gc_mask, target, path

    def process_gradcam(self,file_name,abs_flag=True):
        gradient = np.load(file_name)
        gradient = np.clip(gradient,gradient.mean() - 3 * gradient.std(), gradient.mean() + 3 * gradient.std())

        if abs_flag:
            gradient = np.abs(gradient)
        gradient = gradient.mean(axis=2)

        gradient -= gradient.min()
        gradient /= gradient.max()

        return gradient
