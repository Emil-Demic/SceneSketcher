#!/usr/bin/python
# -*- coding:utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import numel
from torch_geometric.utils import to_dense_adj, to_dense_batch

from layers import GraphConvolution
from utils_model import get_network
import torch
from config import num_categories

LOOP_NUM = 3

class GCNAttention(nn.Module):
    def __init__(self, gcn_input_shape, gcn_output_shape):
        super(GCNAttention, self).__init__()

        self.image_bbox_extract_net = get_network("inceptionv3", num_classes=2048)
        self.global_image_extract_net = get_network("inceptionv3", num_classes=num_categories)
        self.X = nn.Parameter(torch.zeros((num_categories, num_categories), dtype=torch.float32))
        self.linear = nn.Linear(LOOP_NUM, 1)
        nn.init.constant_(self.X, 1e-6)
        # -----------------GCN-----------------------------
        self.gc1 = GraphConvolution(gcn_input_shape, gcn_output_shape)

    def forward(self, image_list, label_list, bbox_list, img, adj, batch):
        '''
        image_list存根据bbox截取好的图像
        label_list存根据bbox截取好的图像类别标签
        category_list存类别对应的5维输入：bbox均值,个数
        '''
        gcn_input = torch.zeros((img.shape[0], num_categories, LOOP_NUM, 2052), dtype=torch.float32, requires_grad=False).cuda()
        img_features = self.image_bbox_extract_net(image_list)
        full_features = torch.concat((img_features, bbox_list), dim=1)
        category_count = np.zeros((img.shape[0], num_categories), dtype=np.int32)
        for b, label, tmp_feature in zip(batch, label_list, full_features):
            gcn_input[b, label - 1, category_count[b, label - 1], :] = tmp_feature
            category_count[b, label - 1] += 1
        gcn_input = torch.transpose(gcn_input, 2, 3)
        gcn_input = self.linear(gcn_input).squeeze()

        global_attention = self.global_image_extract_net(img)

        new_adj = self.X.unsqueeze(0) + adj

        gcn_output = F.leaky_relu(self.gc1(gcn_input, new_adj))

        global_attention = global_attention.unsqueeze(1)

        result_feature = torch.matmul(global_attention, gcn_output)

        return result_feature.squeeze()

    def get_image_feature(self, image):
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

    def get_embedding(self, batch):
        return self.embedding_net(batch.image_list, batch.x, batch.bbox_list, batch.img, batch.adj, batch.batch)

    def get_image_feature(self, image):
        return self.embedding_net.get_image_feature(image)
