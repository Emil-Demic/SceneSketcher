# -*- coding: utf-8 -*-


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import scipy.spatial.distance as ssd
from get_input import loadData
from models import GCNAttention, TripletAttentionNet
from config import *
import tqdm
import numpy as np


def loadDataDirectTest(mode, shuffleList, batchIndex):
    batchIndex = shuffleList[batchIndex]
    if mode == "sketch":

        image_list, label_list, bbox_list, img, adj, corr, area_dict, category_list = loadData(
            os.path.join(sketchVPathTest, str(batchIndex) + ".csv"),
            os.path.join(sketchImgTestPath, str(batchIndex).zfill(12) + ".jpg"))
    else:
        image_list, label_list, bbox_list, img, adj, corr, area_dict, category_list = loadData(
            os.path.join(imageVPathTest, str(batchIndex) + ".csv"),
            os.path.join(imageImgTestPath, str(batchIndex).zfill(12) + ".jpg"))

    return image_list, label_list, bbox_list, img, adj, corr, area_dict, category_list


def compute_view_specific_distance(sketch_feats, image_feats):
    return ssd.cdist(sketch_feats, image_feats, 'sqeuclidean')


def outputHtml(sketchindex, indexList, mode="test"):
    imageNameList = shuffleListTest
    sketchPath = sketchImgTestPath
    imgPath = imageImgTestPath

    tmpLine = "<tr>"

    tmpLine += "<td><image src='%s' width=256 /></td>" % (
        os.path.join(sketchPath, str(shuffleListTest[sketchindex]).zfill(12) + ".png"))
    for i in indexList:
        if i != sketchindex:
            tmpLine += "<td><image src='%s' width=256 /></td>" % (
                os.path.join(imgPath, str(imageNameList[i]).zfill(12) + ".jpg"))
        else:
            tmpLine += "<td ><image src='%s' width=256   style='border:solid 2px red' /></td>" % (
                os.path.join(imgPath, str(imageNameList[i]).zfill(12) + ".jpg"))

    return tmpLine + "</tr>"


def calculate_accuracy(dist, epoch_name):
    top1 = 0
    top5 = 0
    top10 = 0
    top20 = 0
    tmpLine = ""
    for i in range(dist.shape[0]):
        rank = dist[i].argsort()
        if rank[0] == i:
            top1 = top1 + 1
        if i in rank[:5]:
            top5 = top5 + 1
        if i in rank[:10]:
            top10 = top10 + 1
        if i in rank[:20]:
            top20 = top20 + 1
        tmpLine += outputHtml(i, rank[:10]) + "\n"
    num = dist.shape[0]
    print(epoch_name + ' top1: ' + str(top1 / float(num)))
    print(epoch_name + ' top5: ' + str(top5 / float(num)))
    print(epoch_name + 'top10: ' + str(top10 / float(num)))
    print(epoch_name + 'top20: ' + str(top20 / float(num)))

    htmlContent = """
       <html>
       <head></head>
       <body>
       <table>%s</table>
       </body>
       </html>""" % (tmpLine)
    with open(r"html_result/result.html", 'w+') as f:
        f.write(htmlContent)
    return top1, top5, top10, top20


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
            image_list, label_list, bbox_list, img, adj, corr, area_dict, category_list = loadDataDirectTest("sketch",
                                                                                                             shuffleListTest,
                                                                                                             batchIndex)
            a = model.get_embedding(image_list, label_list, bbox_list, img, adj, corr)
            aList.append(a.cpu().numpy()[0])

        aList = np.array(aList)

        for batchIndex in tqdm.tqdm(range(batchesTest)):
            image_list, label_list, bbox_list, img, adj, corr, area_dict, category_list = loadDataDirectTest("image",
                                                                                                             testList,
                                                                                                             batchIndex)
            p = model.get_embedding(image_list, label_list, bbox_list, img, adj, corr)
            pList.append(p.cpu().numpy()[0])

        pList = np.array(pList)

        dis = compute_view_specific_distance(aList, pList)

        top1, top5, top10, top20 = calculate_accuracy(dis, epoch_name)
        print("top1, top5, top10:", top1, top5, top10)
