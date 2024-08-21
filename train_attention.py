#!/usr/bin/python
# -*- coding:utf-8 -*-

import os

from torch.nn import TripletMarginLoss
from torch_geometric.loader import DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch.optim as optim
from utils import compute_view_specific_distance, calculate_accuracy, datasetTestSketch, \
    datasetTestImage, datasetTrain
from models import GCNAttention, TripletAttentionNet
from torch.optim import lr_scheduler
from config import *
import tqdm
import numpy as np
import time


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


optimizer = optim.Adam(model.parameters(),
                       lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
loss_fn = TripletMarginLoss(margin)

t_total = time.time()

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
        if args.cuda:
            batch.cuda()
        t = time.time()
        batch.img_a = batch.img_a.view(-1, 3, 128, 128)
        batch.adj_a = batch.adj_a.view(-1, 15, 15)
        batch.img_p = batch.img_p.view(-1, 3, 128, 128)
        batch.adj_p = batch.adj_p.view(-1, 15, 15)
        batch.img_n = batch.img_n.view(-1, 3, 128, 128)
        batch.adj_n = batch.adj_n.view(-1, 15, 15)

        optimizer.zero_grad()
        output_a, output_p, output_n = model(batch)

        loss = loss_fn(output_a, output_p, output_n)
        running_loss += loss.item()

        loss.backward()
        optimizer.step()  # update parameters of net

        # 3. update parameters of net
        if (i % 5) == 0:
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
