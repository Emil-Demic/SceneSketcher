# -*- coding: utf-8 -*-


import os

from torch_geometric.loader import DataLoader

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


dataset_sketch_test = datasetTestSketch("data")
dataset_image_test = datasetTestImage("data")
dataloader_sketch = DataLoader(dataset_sketch_test, batch_size=args.batch_size, shuffle=False)
dataloader_image = DataLoader(dataset_image_test, batch_size=args.batch_size, shuffle=False)


MaxEpoch = 'epoch1'
for i in os.listdir("model"):
    if not i.startswith("model"):
        continue

    print(os.path.join("model", i))
    # model.load_state_dict(torch.load(os.path.join("model", i), map_location=torch.device('cpu')))
    model.load_state_dict(torch.load(os.path.join("model", i)))
    if args.cuda:
        print("Cuda")
        model.cuda()
    model.eval()
    epoch_name = "Epoch " + str(i)

    aList = []
    pList = []

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
