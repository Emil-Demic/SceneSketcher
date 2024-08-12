#!/usr/bin/python
# -*- coding:utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from layers import GraphConvolution
from utils_model import get_network
import torch
from config import num_categories
import matplotlib.pyplot as plt

LOOP_NUM = 3


class GCNAttention(nn.Module):
    def __init__(self, gcn_input_shape, gcn_output_shape):
        super(GCNAttention, self).__init__()

        self.image_bbox_extract_net = get_network("inceptionv3", num_classes=2048).cuda()
        self.global_image_extract_net = get_network("inceptionv3", num_classes=num_categories).cuda()
        self.X = nn.Parameter(torch.zeros((num_categories, num_categories), dtype=torch.float32)).cuda()
        self.linear = nn.Linear(LOOP_NUM, 1).cuda()
        nn.init.constant_(self.X, 1e-6)
        # -----------------GCN-----------------------------
        self.gc1 = GraphConvolution(gcn_input_shape, gcn_output_shape).cuda()

    def forward(self, image_list, label_list, category_list, total_image, adj, corr):
        '''
        image_list存根据bbox截取好的图像
        label_list存根据bbox截取好的图像类别标签
        category_list存类别对应的5维输入：bbox均值,个数
        '''
        label_list = label_list[0]
        gcn_input = torch.zeros((num_categories, LOOP_NUM, 2052), dtype=torch.float32, requires_grad=False)
        img_features = self.image_bbox_extract_net(image_list[0])
        full_features = torch.hstack((img_features, category_list[0]))
        category_count = np.zeros(num_categories, dtype=np.int32)
        for i, tmp_feature in enumerate(full_features):
            # plt.imshow(image_list[0][i].permute(1,2,0))
            # plt.show()
            # tmp_img = torch.from_numpy(np.transpose(np.array([image_list[0][i] / 255.0], np.float32), [0, 3, 1, 2])).type(
            #     torch.FloatTensor)
            # plt.imshow(tmp_img[0].permute(1, 2, 0))
            # plt.show()

            # if category_count[label_list[i] - 1] < LOOP_NUM:
                # gcn_input[label_list[i] - 1, category_count[label_list[i] - 1], :2048] += img_feature[0]
                # tmp_category = torch.from_numpy(np.array([category_list[i]], np.float32)).type(torch.FloatTensor)
                gcn_input[label_list[i] - 1, category_count[label_list[i] - 1], :] = tmp_feature
                category_count[label_list[i] - 1] += 1
        gcn_input = torch.transpose(gcn_input, 1, 2)
        gcn_input = self.linear(gcn_input).squeeze()

        # total_image = torch.from_numpy(np.transpose(np.array([total_image / 255.0], np.float32), [0, 3, 1, 2])).type(
        #     torch.FloatTensor)
        global_attention = self.global_image_extract_net(total_image)

        # corr = torch.from_numpy(corr).type(torch.FloatTensor)
        # adj = torch.from_numpy(adj).type(torch.FloatTensor)

        new_adj = self.X + adj[0] + corr[0]

        gcn_output = F.leaky_relu(self.gc1(gcn_input, new_adj))

        result_feature = torch.mm(global_attention, gcn_output)

        return result_feature

    def get_image_feature(self, image):
        image = torch.from_numpy(np.transpose(np.array([image / 255.0], np.float32), [0, 3, 1, 2])).type(
            torch.FloatTensor)
        return self.image_bbox_extract_net(image)


class TripletAttentionNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletAttentionNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, image_list_arc, label_list_arc, category_list_arc, total_image_arc, adj_arc, corr_arc,
                image_list_pos, label_list_pos, category_list_pos, total_image_pos, adj_pos, corr_pos,
                image_list_neg, label_list_neg, category_list_neg, total_image_neg, adj_neg, corr_neg
                ):
        output_pos = self.embedding_net(image_list_pos, label_list_pos, category_list_pos, total_image_pos, adj_pos,
                                        corr_pos)
        output_neg = self.embedding_net(image_list_neg, label_list_neg, category_list_neg, total_image_neg, adj_neg,
                                        corr_neg)
        output_arc = self.embedding_net(image_list_arc, label_list_arc, category_list_arc, total_image_arc, adj_arc,
                                        corr_arc)
        return output_arc, output_pos, output_neg

    def get_embedding(self, image_list, label_list, category_list, total_image, adj, corr):
        return self.embedding_net(image_list.cuda(), label_list.cuda(), category_list.cuda(), total_image.cuda(), adj.cuda(), corr.cuda())

    def get_image_feature(self, image):
        return self.embedding_net.get_image_feature(image)
