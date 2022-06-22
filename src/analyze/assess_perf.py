import pandas as pd 
import glob 
from sklearn.metrics import roc_auc_score
import numpy as np 
import sys
from torch.nn import BCEWithLogitsLoss
import torch 
import argparse
import os

def weighted_acc(df,label_col,pred_col):
    true_pos = df[(df[label_col] == 1) & (df[pred_col] == 1)]
    true_neg = df[(df[label_col] == 0) & (df[pred_col] == 0)]

    all_pos = df[(df[label_col] == 1)]
    all_neg = df[(df[label_col] == 0)]

    if len(all_pos) == len(df):
        acc = len(true_pos) / len(all_pos)
    elif len(all_neg) == len(df):
        acc = len(true_neg) / len(all_neg)
    else:
        acc = 0.5 * (len(true_pos)/len(all_pos) + len(true_neg)/len(all_neg))

    return acc

def auc(y_true,y_pred):
    if len(np.unique(y_true)) == 1:
        return np.nan
    else:
        return roc_auc_score(y_true,y_pred)  
def process_file(f, slide_agg):
    criterion = BCEWithLogitsLoss(reduction="none")

    df = pd.read_csv(f)
    df['slide'] = df['path'].apply(lambda x: x.split("/")[3].split("_files")[0])

    # exclude_slides = pd.read_csv("exclude.txt",header=None)[0].values
    # exclude_slides =  [str(x) for x in exclude_slides]
    # df = df[~df['slide'].isin(exclude_slides)]

    run_name = f.split("/")[2]
    fold = f.split("/")[3]
    step = int(f.split("/")[-1].split("-")[-1][:-4])
    phase = f.split("/")[-1].split("_")[0]

    df['prob'] = df['logit'].apply(lambda x: 1/(1 + np.exp(-x)))
    df['label_pred'] = df['prob'] > 0.5
    df['label_pred'] = df['label_pred'].apply(lambda x: int(x))

    tile_auc = auc(df['label'],df['prob'])
    tile_acc = weighted_acc(df,"label","label_pred")
    tile_loss = np.mean(df['loss'])
    
    tile_data = [tile_auc,tile_acc,tile_loss]
    slide_data = []

    if slide_agg and phase != "patch":
        df['slide'] = df['path'].apply(lambda x: x.split("/")[3].split("_files")[0])
        df = df.groupby("slide").median()
        df['label'] = (df['label'] > 0).astype("float")
        del df['prob'], df['loss'], df['label_pred']

        df['mean_logit_prob'] = df['logit'].apply(lambda x: 1/(1 + np.exp(-x)))
        df['mean_logit_loss'] = criterion(torch.from_numpy(df['logit'].values),torch.from_numpy(df['label'].values))

        df['label_pred'] = df['logit'] > 0
        df['label_pred'] = df['label_pred'].apply(lambda x: int(x))

        slide_auc = auc(df['label'],df['mean_logit_prob'])
        slide_acc = weighted_acc(df,"label","label_pred")
        slide_loss = np.mean(df['mean_logit_loss'])

        slide_data = [slide_auc,slide_acc,slide_loss]

    return [run_name,fold,step,phase] + tile_data + slide_data


def process_files(files, slide_agg):
    data = []
    for f in files:
        # print(f)
        perf_stats = process_file(f,slide_agg)
        data.append(perf_stats)

    cols = ['name','fold','step','phase','tile_auc','tile_acc','tile_loss']
    if slide_agg:
        cols += ['slide_auc','slide_acc','slide_loss']

    df = pd.DataFrame(data,columns=cols)

    return df

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str)
    parser.add_argument("--slide_agg", action='store_true')
    parser.add_argument("--train_loss", type=str)
    parser.add_argument("--out_file", type=str,default="summary.csv")
    parser.add_argument("--symlink", action="store_true")
    parser.add_argument("--model_dir", type=str)

    args = parser.parse_args()

    files = glob.glob("{}/*pred*".format(args.results_dir))

    print("Now processing {} files from {}".format(len(files),args.results_dir))

    df = process_files(files,args.slide_agg)

    if args.train_loss is not None:
        train_df = pd.read_csv(args.train_loss)
        train_df.columns = ['step','tile_loss','tile_acc']
        train_df['phase'] = "train"
        train_df['fold'] = df.iloc[0]["fold"]
        train_df['name'] = df.iloc[0]["name"]

        df = pd.concat([df,train_df])

    df.to_csv(os.path.join(args.results_dir,args.out_file),index=False)


    df = df[df['phase'] == "val"]
    
    if args.symlink:
        metrics = ['auc','acc','loss']
        ascending = [False,False,True]
        levels = ['tile']
        if args.slide_agg: levels += ['slide']
        for level in levels:
            for metric, a in zip(metrics, ascending):
                df = df.sort_values("{}_{}".format(level,metric),ascending=a)
                model_info = df.iloc[0]
                print(model_info)
                src = "{}/model_step-{}.pt".format(args.model_dir,model_info['step'])
                src = os.path.abspath(src)
                target = "{}/model_best-val-{}-{}.pt".format(args.results_dir,level,metric)
                if os.path.exists(target):
                    os.unlink(target)
                os.symlink(src,target)

    print("done")
