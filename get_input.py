# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import torch
from torchvision.io import read_image
from torchvision.transforms.v2 import Resize, ToDtype
from torchvision.tv_tensors import BoundingBoxes, Image

from config import num_categories

word2vec = []
with open("classFeature.csv", "r") as f:
    lines = f.readlines()
    for line in lines:
        word2vec.append([float(x) for x in line.strip().split(",")])


def cosSim(x, y):
    tmp = sum(a * b for a, b in zip(x, y))
    non = np.linalg.norm(x) * np.linalg.norm(y)
    return round(tmp / float(non), 3)


def euDistance(x, y):
    x_center = ((x[0] + x[2]) / 2, (x[1] + x[3]) / 2)
    y_center = ((y[0] + y[2]) / 2, (y[1] + y[3]) / 2)

    distance = np.sqrt((x_center[0] - y_center[0]) ** 2 + (x_center[1] - y_center[1]) ** 2)
    return distance


def loadData(vpath, imgpath):
    assert os.path.exists(vpath)
    assert os.path.exists(imgpath)

    to_float = ToDtype(torch.float32, scale=True)
    img = Image(read_image(imgpath))
    resize_BB = Resize((32, 32))
    resize_global = Resize((128, 128))

    channel, img_height, img_width = img.shape
    label_list = []
    image_list = []
    bbox_list = []
    category_dict = {}

    category_list = []
    corr = torch.zeros((num_categories, num_categories))
    adj = torch.zeros((num_categories, num_categories))
    category_count = np.zeros(num_categories, dtype=np.int8)
    with open(vpath, 'r+') as f:
        lines = f.readlines()
        for line in lines[1:]:
            features = list(map(float, line.strip().split(",")))
            category = int(features[1]) - 1
            if category_count[category] == 3:
                continue
            bbox = [round(x) for x in features[2:]]
            if bbox[3] > img_height or bbox[2] > img_width:
                continue
            if (bbox[3] - bbox[1]) < 32 or (bbox[2] - bbox[0]) < 32:
                continue
            bbox_img = Image(img[:, bbox[1]:bbox[3], bbox[0]:bbox[2]])
            image_list.append(to_float(resize_BB(bbox_img)))
            normalize_bbox = [bbox[0] / img_width, bbox[1] / img_height, bbox[2] / img_width, bbox[3] / img_height]
            bbox_list.append(normalize_bbox)

            if category in category_dict.keys():
                category_dict[category].append(normalize_bbox)
            else:
                category_dict[category] = [normalize_bbox]
            corr[category, category] = 1
            label_list.append(category + 1)
            category_count[category] += 1

    for i in range(num_categories):
        if corr[i, i] == 1:
            for j in range(i + 1, num_categories):
                if corr[j, j] == 1:
                    sim = cosSim(word2vec[i], word2vec[j])
                    corr[i, j] = sim
                    corr[j, i] = sim

    for i in range(num_categories):
        category_list.append([])
    for i in range(num_categories):
        if i in category_dict.keys():
            category_list[i] = list(np.mean(category_dict[i], axis=0)) + [len(category_dict[i])]
        else:
            category_list[i] = [0, 0, 0, 0, 0]

    for i in range(num_categories):
        for j in range(i + 1, num_categories):
            if i in category_dict.keys() and j in category_dict.keys():
                distance = euDistance(category_list[i], category_list[j])
                distance = 1.0 - distance

                adj[i, j] = distance
                adj[j, i] = distance

    img = to_float(resize_global(img))

    if len(image_list) > 1:
        image_list = torch.stack(image_list)
    else:
        image_list = torch.zeros((3, 32, 32))
    label_list = torch.tensor(label_list)
    bbox_list = torch.tensor(bbox_list, dtype=torch.float32)
    # bbox_list = BoundingBoxes(bbox_list, format="XYXY", canvas_size=(img_height, img_width))

    return image_list, label_list, bbox_list, img, adj, corr
