import pandas as pd 
import glob 
from sklearn.metrics import roc_auc_score
import numpy as np 
import sys
import argparse
import os
import warnings
from sklearn.metrics import confusion_matrix

def perf_measure(y_true, y_pred):
    c = confusion_matrix(y_true,y_pred)
    TN = c[0][0]
    FN = c[1][0]
    TP = c[1][1]
    FP = c[0][1]

    sensitivity  = TP / (TP+FN)
    specificity  = TN / (TN+FP)
    # recall = TP / (TP+FP)

    return sensitivity, specificity

def weighted_acc(y_true,y_pred):
    tpr, tnr = perf_measure(y_true,y_pred)
    acc = (tpr + tnr) / 2
    return acc

def auc(y_true,y_pred):
    if len(np.unique(y_true)) == 1:
        return np.nan
    else:
        return roc_auc_score(y_true,y_pred)

def process_files(exp_dir,phase,n_bootstraps,slide_agg,ci=0.05):

    files = glob.glob("{}/fold-*/results/{}_*".format(exp_dir,phase))
    print((files))
    data = []
    for f in files:
        tmp_df = pd.read_csv(f)
        tmp_df['file'] = f
        data.append(tmp_df)

    df = pd.concat(data)

    if phase == "patch":
        df['slide'] = df['path'].apply(lambda x: x.split("/")[2].split("-")[0])
    else:
        df['slide'] = df['path'].apply(lambda x: x.split("/")[3].split("_")[0])

    if not slide_agg:
        print("tile level")
        df = df.groupby("path").median()
        df = df.reset_index()
        df['prob'] = df['logit'].apply(lambda x: 1/(1 + np.exp(-x)))
        df['label_pred'] = df['prob'] > 0.5
        df['label_pred'] = df['label_pred'].apply(lambda x: int(x))
        
        bootstrap(df.copy(),n_bootstraps,ci)
    else: 
        df = df.groupby(["slide","file"]).median().reset_index()
        df = df.groupby("slide").median()

        df['prob'] = df['logit'].apply(lambda x: 1/(1 + np.exp(-x)))

        df['label_pred'] = df['logit'] > 0
        df['label_pred'] = df['label_pred'].apply(lambda x: int(x))

        print("slide level")
        # print(df[(df['label'] == 1) & (df['label_pred'] == 0)])
        # pvals(df,n_perms)
        bootstrap(df.copy(),n_bootstraps,ci)


# def slide_perf(df,n_perm)

def bootstrap(df,n_bootstraps,ci):
    rng_seed = 42  # control reproducibility
    rng = np.random.RandomState(rng_seed)

    bootstrapped_aucs = []
    bootstrapped_accs = []
    bootstrapped_tprs = []
    bootstrapped_tnrs = []

    y_true = df['label'].values.astype("int")
    y_pred = df['label_pred'].values
    y_prob = df['prob'].values

    true_tpr, true_tnr = perf_measure(y_true,y_pred)

    true_auc = roc_auc_score(y_true, y_prob)
    true_acc = weighted_acc(y_true, y_pred)

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            continue

        auc = roc_auc_score(y_true[indices], y_prob[indices])
        acc = weighted_acc(y_true[indices], y_pred[indices])
        tpr, tnr = perf_measure(y_true[indices], y_pred[indices])

        bootstrapped_aucs.append(auc)
        bootstrapped_accs.append(acc)
        bootstrapped_tprs.append(tpr)
        bootstrapped_tnrs.append(tnr)

    print_ci(bootstrapped_accs,"acc",true_acc,ci)
    print_ci(bootstrapped_aucs,"auc",true_auc,ci)

def print_ci(scores,metric,value,ci):
    sorted_scores = np.array(scores)
    sorted_scores.sort()

    confidence_upper = sorted_scores[int(ci * len(sorted_scores))]
    confidence_lower = sorted_scores[int((1-ci) * len(sorted_scores))]
    print("CI {}: {:0.3f} [{:0.3f} - {:0.3}]".format(
        metric,value,confidence_lower, confidence_upper))


if __name__ == "__main__":

    model, phase, exp, slide, ci = sys.argv[1:]
    if slide == "True":
        slide_agg = True
    else:
        slide_agg = False

    exp_dir = "experiments/{}/{}_moffitt".format(exp,model)
    n_bootstraps = 100
    print(model,phase)
    process_files(exp_dir,phase,n_bootstraps,slide_agg,float(ci))
    print()
