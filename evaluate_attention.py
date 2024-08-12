# -*- coding: utf-8 -*-


import os

from torch.utils.data import DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from utils import compute_view_specific_distance, calculate_accuracy, datasetTestSketch, datasetTestImage
from models import GCNAttention, TripletAttentionNet
from config import *
import tqdm
import numpy as np

# -------------------------------------------------------------------------------------------------------------------------------------------
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
embedding_net = GCNAttention(gcn_input_shape=featureDim, gcn_output_shape=outPutDim)

model = TripletAttentionNet(embedding_net)
if args.cuda:
    model.cuda()

dataset_sketch = datasetTestSketch(sketchImgTestPath, sketchVPathTest, shuffleList)
dataset_image = datasetTestImage(imageImgTestPath, imageVPathTest, shuffleList)

dataloader_sketch = DataLoader(dataset_sketch, batch_size=batch_size, shuffle=False)
dataloader_image = DataLoader(dataset_image, batch_size=batch_size, shuffle=False)

MaxEpoch = 'epoch1'
for i in os.listdir("model"):
    if not i.startswith("model"):
        continue

    print(os.path.join("model", i))
    model.load_state_dict(torch.load(os.path.join("model", i)))
    model.eval()
    epoch_name = "Epoch " + str(i)

    aList = []
    pList = []

    with torch.no_grad():

        for batch in dataloader_sketch:
            # image_list, label_list, bbox_list, img, adj, corr = loadDataDirectTest("sketch",
            #                                                                        shuffleListTest,
            #                                                                        batchIndex)
            print(len(batch))
            print(batch.shape)
            a = model.get_embedding(batch)
            print(a)
            aList.append(a.cpu().numpy()[0])

        aList = np.array(aList)

        for batch in dataloader_image:
            # image_list, label_list, bbox_list, img, adj, corr = loadDataDirectTest("image",
            #                                                                        shuffleListTest,
            #                                                                        batchIndex)
            p = model.get_embedding(batch)
            pList.append(p.cpu().numpy()[0])

        pList = np.array(pList)

        dis = compute_view_specific_distance(aList, pList)

        top1, top5, top10, top20 = calculate_accuracy(dis, epoch_name)
        print("top1, top5, top10:", top1, top5, top10)
