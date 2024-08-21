#!/usr/bin/python
# -*- coding:utf-8 -*-

import os

from torch_geometric.loader import DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.optim as optim
from utils import compute_view_specific_distance, loadDataDirectTest, calculate_accuracy, datasetTestSketch, \
    datasetTestImage, datasetTrain
from get_input import loadData
from models import GCNAttention, TripletAttentionNet
from loss import TripletLoss
from torch.optim import lr_scheduler
from config import *
import tqdm
import numpy as np
import random
import time
import os


# def loadDataDirect(shuffleList, batchIndex):
#     """Load citation network dataset (cora only for now)"""
#     '''
#             image_list: images captured according to bounding boxes
#             label_list: labels of image_list
#             category_list: the center coordinates and the number of bounding boxes
#             '''
#     batchIndex = shuffleList[batchIndex]
#     image_list_arc, label_list_arc, bbox_list_arc, total_image_arc, adj_arc, corr_arc = loadData(
#         os.path.join(sketchVPath, str(batchIndex) + ".csv"),
#         os.path.join(sketchImgTrainPath, str(batchIndex).zfill(12) + ".png"))
#
#     image_list_pos, label_list_pos, bbox_list_pos, total_image_pos, adj_pos, corr_pos = loadData(
#         os.path.join(imageVPath, str(batchIndex) + ".csv"),
#         os.path.join(imageImgPath, str(batchIndex).zfill(12) + ".jpg"))
#
#     shuffleNum = batchIndex
#     while shuffleNum == batchIndex:
#         shuffleNum = shuffleList[random.randint(0, len(shuffleList) - 1)]
#
#     image_list_neg, label_list_neg, bbox_list_neg, total_image_neg, adj_neg, corr_neg = loadData(
#         os.path.join(imageVPath, str(shuffleNum) + ".csv"),
#         os.path.join(imageImgPath, str(shuffleNum).zfill(12) + ".jpg"))
#
#     return image_list_arc, label_list_arc, bbox_list_arc, total_image_arc, adj_arc, corr_arc, image_list_pos, label_list_pos, bbox_list_pos, total_image_pos, adj_pos, corr_pos, image_list_neg, label_list_neg, bbox_list_neg, total_image_neg, adj_neg, corr_neg


np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
margin = 1.
embedding_net = GCNAttention(gcn_input_shape=featureDim, gcn_output_shape=outPutDim)

model = TripletAttentionNet(embedding_net)
if args.cuda:
    print("Cuda")
    model.cuda()

loss_fn = TripletLoss(margin)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)

Result_Step = 1

t_total = time.time()

maxModel = "0"

dataset_train = datasetTrain("data")
dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, follow_batch=['x_a', 'x_p', 'x_n'])

dataset_sketch_test = datasetTestSketch("data")
dataset_image_test = datasetTestImage("data")
dataloader_sketch = DataLoader(dataset_sketch_test, batch_size=args.batch_size * 3, shuffle=False)
dataloader_image = DataLoader(dataset_image_test, batch_size=args.batch_size * 3, shuffle=False)

for epoch in range(args.epochs + 1):
    model.train()
    running_loss = 0.0
    for i, batch in enumerate(dataloader_train):
        batch.cuda()
        t = time.time()
        batch.img_a = batch.img_a.view(-1, 3, 128, 128)
        batch.adj_a = batch.adj_a.view(-1, 15, 15)
        batch.img_p = batch.img_p.view(-1, 3, 128, 128)
        batch.adj_p = batch.adj_p.view(-1, 15, 15)
        batch.img_n = batch.img_n.view(-1, 3, 128, 128)
        batch.adj_n = batch.adj_n.view(-1, 15, 15)

        output1, output2, output3 = model(batch)

        loss = loss_fn(output1, output2, output3)

        # 2.1 loss regularization

        # 2.2 back propagation
        optimizer.zero_grad()  # reset gradient
        loss.backward()
        optimizer.step()  # update parameters of net
        running_loss += loss.item()
        # 3. update parameters of net
        if (i % args.batch_size) == 0 or (batch + 1) == batches:
            # optimizer the net
            print('Epoch: {:04d}'.format(epoch + 1), 'Batch: {:04d}'.format(i + 1),
                  'loss_train: {:.4f}'.format(running_loss / args.batch_size),
                  'time: {:.4f}s'.format(time.time() - t))
            running_loss = 0.0

        torch.save(model.state_dict(), "model/model_" + str(epoch) + ".pth")
    # -----------------evaluate---------------------------
    epoch_name = str(epoch)

    aList = []
    pList = []
    model.eval()

    with torch.no_grad():

        for batch in tqdm.tqdm(dataloader_sketch):
            if args.cuda:
                batch.cuda()
            batch.img = batch.img.view(-1, 3, 128, 128)
            batch.adj = batch.adj.view(-1, 15, 15)
            a = model.get_embedding(batch)
            aList.append(a.cpu().numpy())

        for batch in tqdm.tqdm(dataloader_image):
            if args.cuda:
                batch.cuda()
            batch.img = batch.img.view(-1, 3, 128, 128)
            batch.adj = batch.adj.view(-1, 15, 15)
            p = model.get_embedding(batch)
            pList.append(p.cpu().numpy())

        aList = np.concatenate(aList)
        pList = np.concatenate(pList)

        dis = compute_view_specific_distance(aList, pList)

        top1, top5, top10, top20 = calculate_accuracy(dis, epoch_name)
        print("top1, top5, top10:", top1, top5, top10)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
