import random

import scipy.spatial.distance as ssd
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import dense_to_sparse

from config import *
from get_input import loadData


def compute_view_specific_distance(sketch_feats, image_feats):
    return ssd.cdist(sketch_feats, image_feats, 'sqeuclidean')


def outputHtml(sketchindex, indexList):
    imageNameList = shuffleListTest
    sketchPath = sketchImgTestPath
    imgPath = imageImgTestPath

    tmpLine = "<tr>"

    tmpLine += "<td><image src='%s' width=256 /></td>" % (
        os.path.join(sketchPath, str(shuffleListTest[sketchindex]).zfill(12) + ".jpg"))
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


class datasetTrain(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(datasetTrain, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        csv_files_sketch = os.listdir("train/sketch/GraphFeatures/")
        jpg_files_sketch = os.listdir("train/sketch/Image/")
        csv_files_image = os.listdir("train/image/GraphFeatures/")
        jpg_files_image = os.listdir("train/image/Image/")
        return csv_files_sketch + jpg_files_sketch + csv_files_image + jpg_files_image

    @property
    def processed_file_names(self):
        processed_files_sketches = []
        processed_files_images = []
        for i in range(len(self.raw_file_names) // 4):
            processed_files_sketches.append(f"data_sketch_train_{i}.pt")
            processed_files_images.append(f"data_image_train_{i}.pt")
        return processed_files_sketches + processed_files_images

    def process(self):
        idx = 0

        for i in range(self.len()):
            batchIndex = shuffleList[idx]
            image_list, label_list, bbox_list, img, adj = loadData(
                # os.path.join(sketchVPath, str(batchIndex).zfill(12) + ".csv"),
                os.path.join(sketchVPath, str(batchIndex) + ".csv"),
                os.path.join(sketchImgPath, str(batchIndex).zfill(12) + ".png"))

            data_s = Data(image_list=image_list, x=label_list, bbox_list=bbox_list,
                          img=img, adj=adj)

            torch.save(data_s, os.path.join(self.processed_dir, f'data_sketch_train_{idx}.pt'))

            image_list, label_list, bbox_list, img, adj = loadData(
                # os.path.join(imageVPath, str(batchIndex).zfill(12) + ".csv"),
                os.path.join(imageVPath, str(batchIndex) + ".csv"),
                os.path.join(imageImgPath, str(batchIndex).zfill(12) + ".jpg"))

            data_i = Data(image_list=image_list, x=label_list, bbox_list=bbox_list,
                          img=img, adj=adj)

            torch.save(data_i, os.path.join(self.processed_dir, f'data_image_train_{idx}.pt'))

            idx += 1

    def len(self):
        return len(self.processed_file_names) // 2

    def get(self, idx):
        data_a = torch.load(os.path.join(self.processed_dir, f'data_sketch_train_{idx}.pt'))
        data_p = torch.load(os.path.join(self.processed_dir, f'data_image_train_{idx}.pt'))

        negative_idx = random.randint(0, self.len() - 1)
        while negative_idx == idx:
            negative_idx = random.randint(0, self.len() - 1)

        data_n = torch.load(os.path.join(self.processed_dir, f'data_image_train_{negative_idx}.pt'))

        data = Data(image_list_a=data_a.image_list, x_a=data_a.x, bbox_list_a=data_a.bbox_list, img_a=data_a.img,
                    adj_a=data_a.adj,
                    image_list_p=data_p.image_list, x_p=data_p.x, bbox_list_p=data_p.bbox_list, img_p=data_p.img,
                    adj_p=data_p.adj,
                    image_list_n=data_n.image_list, x_n=data_n.x, bbox_list_n=data_n.bbox_list, img_n=data_n.img,
                    adj_n=data_n.adj)
        return data


class datasetTestSketch(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(datasetTestSketch, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        csv_files = os.listdir("test/sketch/GraphFeatures/")
        jpg_files = os.listdir("test/sketch/Image/")
        return csv_files + jpg_files

    @property
    def processed_file_names(self):
        processed_files_sketches = []
        for i in range(len(self.raw_file_names) // 2):
            processed_files_sketches.append(f"data_sketch_test_{i}.pt")
        return processed_files_sketches

    def process(self):
        idx = 0

        for i in range(self.len()):
            batchIndex = shuffleListTest[idx]
            image_list, label_list, bbox_list, img, adj = loadData(
                os.path.join(sketchVPathTest, str(batchIndex) + ".csv"),
                # os.path.join(sketchVPathTest, str(batchIndex).zfill(12) + ".csv"),
                os.path.join(sketchImgTestPath, str(batchIndex).zfill(12) + ".png"))

            data = Data(image_list=image_list, x=label_list, bbox_list=bbox_list,
                        img=img, adj=adj)

            torch.save(data, os.path.join(self.processed_dir, f'data_sketch_test_{idx}.pt'))
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_sketch_test_{idx}.pt'))
        return data


class datasetTestImage(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(datasetTestImage, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        csv_files = os.listdir("test/image/GraphFeatures/")
        jpg_files = os.listdir("test/image/Image/")
        return csv_files + jpg_files

    @property
    def processed_file_names(self):
        processed_files_images = []
        for i in range(len(self.raw_file_names) // 2):
            processed_files_images.append(f"data_image_test_{i}.pt")
        return processed_files_images

    def process(self):
        idx = 0

        for i in range(self.len()):
            batchIndex = shuffleListTest[idx]
            image_list, label_list, bbox_list, img, adj = loadData(
                os.path.join(imageVPathTest, str(batchIndex) + ".csv"),
                # os.path.join(imageVPathTest, str(batchIndex).zfill(12) + ".csv"),
                os.path.join(imageImgTestPath, str(batchIndex).zfill(12) + ".jpg"))

            data = Data(image_list=image_list, x=label_list, bbox_list=bbox_list,
                        img=img, adj=adj)

            torch.save(data, os.path.join(self.processed_dir, f'data_image_test_{idx}.pt'))
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_image_test_{idx}.pt'))
        return data
