# -*- coding: utf-8 -*-

import os
import torch
import time
import argparse

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='testidate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
nowTime = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

featureType = "GraphFeatures"
featureNumDict = {"GraphFeatures": 2052}
model_path = "./model/"
Result_Step = 5
learningRate = 0.01
num_supports = 1
MARGIN = 0.2
epochs = 20

outPutDim = 256
hardMode = False
batch_size = 5
# ------------------------------------sketch /Image Truly Image For display-------------------------

imageImgPath = r"train/image/Image"
imageImgTestPath=r"test/image/Image"
sketchImgTrainPath = r"train/sketch/Image"
sketchImgTestPath = r"test/sketch/Image"

# ------------------------------------sketch Features For train and test-------------------------
sketchVPath = "train/sketch/GraphFeatures"

sketchVPathTest = "test/sketchGraphFeatures"


# ------------------------------------image Features For train and test-------------------------

imageVPath = "train/image/GraphFeatures"

imageVPathTest = "test/image/GraphFeatures"


shuffleListTest = os.listdir(sketchVPathTest)
batchesTest = len(shuffleListTest)
shuffleListTest = [int(x.split(".")[0]) for x in shuffleListTest]

imgTestList = os.listdir(imageImgTestPath)

print(shuffleListTest)
