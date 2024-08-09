# -*- coding: utf-8 -*-


import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from utils import loadDataDirectTest, compute_view_specific_distance, calculate_accuracy
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

        for batchIndex in tqdm.tqdm(range(batchesTest)):
            image_list, label_list, bbox_list, img, adj, corr = loadDataDirectTest("sketch",
                                                                                   shuffleListTest,
                                                                                   batchIndex)
            a = model.get_embedding(image_list, label_list, bbox_list, img, adj, corr)
            aList.append(a.cpu().numpy()[0])

        aList = np.array(aList)

        for batchIndex in tqdm.tqdm(range(batchesTest)):
            image_list, label_list, bbox_list, img, adj, corr = loadDataDirectTest("image",
                                                                                   shuffleListTest,
                                                                                   batchIndex)
            p = model.get_embedding(image_list, label_list, bbox_list, img, adj, corr)
            pList.append(p.cpu().numpy()[0])

        pList = np.array(pList)

        dis = compute_view_specific_distance(aList, pList)

        top1, top5, top10, top20 = calculate_accuracy(dis, epoch_name)
        print("top1, top5, top10:", top1, top5, top10)
