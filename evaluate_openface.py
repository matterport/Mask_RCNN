#!/usr/bin/env python2
#
# Copyright 2015-2016 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This implements the standard LFW verification experiment.

import math
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

import os
import sys

import argparse

from scipy import arange


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'tag', type=str, help='The label/tag to put on the ROC curve.')
    parser.add_argument('workDir', type=str,
                        help='The work directory with labels.csv and reps.csv.')
    pairsDefault = os.path.expanduser("~/openface/data/lfw/pairs.txt")
    parser.add_argument('--lfwPairs', type=str,
                        default=os.path.expanduser(
                            "~/openface/data/lfw/pairs.txt"),

                        help='Location of the LFW pairs file from http://vis-www.cs.umass.edu/lfw/pairs.txt')
    args = parser.parse_args()

    if not os.path.isfile(args.lfwPairs):
        print("""
Error in LFW evaluation code. (Source: <openface>/evaluation/lfw.py)

The LFW evaluation requires a file containing pairs of faces to evaluate.
Download this file from http://vis-www.cs.umass.edu/lfw/pairs.txt
and place it in the default location ({})
or pass it as --lfwPairs.
""".format(pairsDefault))
        sys.exit(-1)

    print("Loading embeddings.")
    fname = "{}/labels.csv".format(args.workDir)
    paths = pd.read_csv(fname, header=None).as_matrix()[:, 1]
    paths = map(os.path.basename, paths)  # Get the filename.
    # Remove the extension.
    paths = map(lambda path: os.path.splitext(path)[0], paths)
    fname = "{}/reps.csv".format(args.workDir)
    rawEmbeddings = pd.read_csv(fname, header=None).as_matrix()
    embeddings = dict(zip(*[paths, rawEmbeddings]))

    pairs = loadPairs(args.lfwPairs, embeddings)
    verifyExp(args.workDir, pairs, embeddings)
    plotVerifyExp(args.workDir, args.tag)


def loadPairs(pairsFname, embeddings):
    print("  + Reading pairs.")
    pairs = []
    with open(pairsFname, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            if check_pair_exists(pair, embeddings):
                pairs.append(pair)
    # assert(len(pairs) == 6000)
    print(len(pairs))
    return np.array(pairs)


def check_pair_exists(pair, embeddings):
    if len(pair) == 3:
        name1 = "{}_{}".format(pair[0], pair[1].zfill(4))
        name2 = "{}_{}".format(pair[0], pair[2].zfill(4))
        actual_same = True
    elif len(pair) == 4:
        name1 = "{}_{}".format(pair[0], pair[1].zfill(4))
        name2 = "{}_{}".format(pair[2], pair[3].zfill(4))
        actual_same = False
    else:
        raise Exception(
            "Unexpected pair length: {}".format(len(pair)))

    return True if name1 in embeddings and name2 in embeddings else False


def getEmbeddings(pair, embeddings):
    if len(pair) == 3:
        name1 = "{}_{}".format(pair[0], pair[1].zfill(4))
        name2 = "{}_{}".format(pair[0], pair[2].zfill(4))
        actual_same = True
    elif len(pair) == 4:
        name1 = "{}_{}".format(pair[0], pair[1].zfill(4))
        name2 = "{}_{}".format(pair[2], pair[3].zfill(4))
        actual_same = False
    else:
        raise Exception(
            "Unexpected pair length: {}".format(len(pair)))

    (x1, x2) = (embeddings[name1], embeddings[name2])
    return (x1, x2, actual_same)


def writeROC(fname, thresholds, embeddings, pairsTest):
    with open(fname, "w") as f:
        f.write("threshold,tp,tn,fp,fn,tpr,fpr\n")
        tp = tn = fp = fn = 0
        for threshold in thresholds:
            tp = tn = fp = fn = 0
            for pair in pairsTest:
                (x1, x2, actual_same) = getEmbeddings(pair, embeddings)
                diff = x1 - x2
                dist = np.dot(diff.T, diff)
                predict_same = dist < threshold

                if predict_same and actual_same:
                    tp += 1
                elif predict_same and not actual_same:
                    fp += 1
                elif not predict_same and not actual_same:
                    tn += 1
                elif not predict_same and actual_same:
                    fn += 1

            if tp + fn == 0:
                tpr = 0
            else:
                tpr = float(tp) / float(tp + fn)
            if fp + tn == 0:
                fpr = 0
            else:
                fpr = float(fp) / float(fp + tn)
            f.write(",".join([str(x)
                              for x in [threshold, tp, tn, fp, fn, tpr, fpr]]))
            f.write("\n")
            if tpr == 1.0 and fpr == 1.0:
                # No further improvements.
                f.write(",".join([str(x)
                                  for x in [4.0, tp, tn, fp, fn, tpr, fpr]]))
                return


def getDistances(embeddings, pairsTrain):
    list_dist = []
    y_true = []
    for pair in pairsTrain:
        (x1, x2, actual_same) = getEmbeddings(pair, embeddings)
        diff = x1 - x2
        dist = np.dot(diff.T, diff)
        list_dist.append(dist)
        y_true.append(actual_same)
    return np.asarray(list_dist), np.array(y_true)


def evalThresholdAccuracy(embeddings, pairs, threshold):
    distances, y_true = getDistances(embeddings, pairs)
    y_predict = np.zeros(y_true.shape)
    y_predict[np.where(distances < threshold)] = 1

    y_true = np.array(y_true)
    accuracy = accuracy_score(y_true, y_predict)
    return accuracy, pairs[np.where(y_true != y_predict)]


def findBestThreshold(thresholds, embeddings, pairsTrain):
    bestThresh = bestThreshAcc = 0
    distances, y_true = getDistances(embeddings, pairsTrain)
    for threshold in thresholds:
        y_predlabels = np.zeros(y_true.shape)
        y_predlabels[np.where(distances < threshold)] = 1
        accuracy = accuracy_score(y_true, y_predlabels)
        if accuracy >= bestThreshAcc:
            bestThreshAcc = accuracy
            bestThresh = threshold
        else:
            # No further improvements.
            return bestThresh
    return bestThresh


def verifyExp(workDir, pairs, embeddings):
    print("  + Computing accuracy.")
    folds = KFold(n=len(pairs), n_folds=10, shuffle=False)
    thresholds = arange(0, 4, 0.01)

    if os.path.exists("{}/accuracies.txt".format(workDir)):
        print("{}/accuracies.txt already exists. Skipping processing.".format(workDir))
    else:
        accuracies = []
        with open("{}/accuracies.txt".format(workDir), "w") as f:
            f.write('fold, threshold, accuracy\n')
            for idx, (train, test) in enumerate(folds):
                print(idx, test)
                fname = "{}/l2-roc.fold-{}.csv".format(workDir, idx)
                writeROC(fname, thresholds, embeddings, pairs[test])

                bestThresh = findBestThreshold(
                    thresholds, embeddings, pairs[train])
                accuracy, pairs_bad = evalThresholdAccuracy(
                    embeddings, pairs[test], bestThresh)
                accuracies.append(accuracy)
                f.write('{}, {:0.2f}, {:0.2f}\n'.format(
                    idx, bestThresh, accuracy))
            avg = np.mean(accuracies)
            std = np.std(accuracies)
            f.write('\navg, {:0.4f} +/- {:0.4f}\n'.format(avg, std))
            print('    + {:0.4f}'.format(avg))


def getAUC(fprs, tprs):
    sortedFprs, sortedTprs = zip(*sorted(zip(*(fprs, tprs))))
    sortedFprs = list(sortedFprs)
    sortedTprs = list(sortedTprs)
    if sortedFprs[-1] != 1.0:
        sortedFprs.append(1.0)
        sortedTprs.append(sortedTprs[-1])
    return np.trapz(sortedTprs, sortedFprs)


def plotOpenFaceROC(workDir, plotFolds=True, color=None):
    fs = []
    for i in range(10):
        rocData = pd.read_csv("{}/l2-roc.fold-{}.csv".format(workDir, i))
        fs.append(interp1d(rocData['fpr'], rocData['tpr']))
        x = np.linspace(0, 1, 1000)
        if plotFolds:
            foldPlot, = plt.plot(x, fs[-1](x), color='grey', alpha=0.5)
        else:
            foldPlot = None

    fprs = []
    tprs = []
    for fpr in np.linspace(0, 1, 1000):
        tpr = 0.0
        for f in fs:
            v = f(fpr)
            if math.isnan(v):
                v = 0.0
            tpr += v
        tpr /= 10.0
        fprs.append(fpr)
        tprs.append(tpr)
    if color:
        meanPlot, = plt.plot(fprs, tprs, color=color)
    else:
        meanPlot, = plt.plot(fprs, tprs)
    AUC = getAUC(fprs, tprs)
    return foldPlot, meanPlot, AUC


def plotVerifyExp(workDir, tag):
    print("Plotting.")

    fig, ax = plt.subplots(1, 1)

    openbrData = pd.read_csv("comparisons/openbr.v1.1.0.DET.csv")
    openbrData['Y'] = 1 - openbrData['Y']
    # brPlot = openbrData.plot(x='X', y='Y', legend=True, ax=ax)
    brPlot, = plt.plot(openbrData['X'], openbrData['Y'])
    brAUC = getAUC(openbrData['X'], openbrData['Y'])

    foldPlot, meanPlot, AUC = plotOpenFaceROC(workDir, color='k')

    humanData = pd.read_table(
        "comparisons/kumar_human_crop.txt", header=None, sep=' ')
    humanPlot, = plt.plot(humanData[1], humanData[0])
    humanAUC = getAUC(humanData[1], humanData[0])

    deepfaceData = pd.read_table(
        "comparisons/deepface_ensemble.txt", header=None, sep=' ')
    dfPlot, = plt.plot(deepfaceData[1], deepfaceData[0], '--',
                       alpha=0.75)
    deepfaceAUC = getAUC(deepfaceData[1], deepfaceData[0])

    # baiduData = pd.read_table(
    #     "comparisons/BaiduIDLFinal.TPFP", header=None, sep=' ')
    # bPlot, = plt.plot(baiduData[1], baiduData[0])
    # baiduAUC = getAUC(baiduData[1], baiduData[0])

    eigData = pd.read_table(
        "comparisons/eigenfaces-original-roc.txt", header=None, sep=' ')
    eigPlot, = plt.plot(eigData[1], eigData[0])
    eigAUC = getAUC(eigData[1], eigData[0])

    ax.legend([humanPlot, dfPlot, brPlot, eigPlot,
               meanPlot, foldPlot],
              ['Human, Cropped [AUC={:.3f}]'.format(humanAUC),
               # 'Baidu [{:.3f}]'.format(baiduAUC),
               'DeepFace Ensemble [{:.3f}]'.format(deepfaceAUC),
               'OpenBR v1.1.0 [{:.3f}]'.format(brAUC),
               'Eigenfaces [{:.3f}]'.format(eigAUC),
               'OpenFace {} [{:.3f}]'.format(tag, AUC),
               'OpenFace {} folds'.format(tag)],
              loc='lower right')

    plt.plot([0, 1], color='k', linestyle=':')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    # plt.ylim(ymin=0,ymax=1)
    plt.xlim(xmin=0, xmax=1)

    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='k', linestyle='-', alpha=0.2)
    plt.minorticks_on()
    # fig.savefig(os.path.join(workDir, "roc.pdf"))
    fig.savefig(os.path.join(workDir, "roc.png"))

if __name__ == '__main__':
    main()
