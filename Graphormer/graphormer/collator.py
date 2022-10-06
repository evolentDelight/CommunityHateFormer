# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torch_geometric.transforms import RandomNodeSplit

splitter = RandomNodeSplit("train_rest", num_val=0.10, num_test=0.20)

def pad_1d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)

def pad_1d_unsqueeze_y(y, padlen):
    ylen = y.size(0)
    if ylen < padlen:
        new_y = y.new_zeros([padlen], dtype=y.dtype)
        new_y[:ylen] = y
        y = new_y
    return y.unsqueeze(0)

def pad_1d_unsqueeze_mask(y, padlen):
    ylen = y.size(0)
    if ylen < padlen:
        new_y = y.new_zeros([padlen], dtype=y.dtype)
        new_y[:ylen] = y
        y = new_y
    return y.unsqueeze(0)


def pad_2d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros(
            [padlen, padlen], dtype=x.dtype).fill_(float('-inf'))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)


def pad_edge_type_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_spatial_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):
    x = x + 1
    xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3, :] = x
        x = new_x
    return x.unsqueeze(0)


class Batch():
    def __init__(self, idx, attn_bias, spatial_pos, in_degree, out_degree, x, y, mask):
        super(Batch, self).__init__()
        self.idx = idx
        self.in_degree, self.out_degree = in_degree, out_degree
        self.x, self.y = x, y
        self.attn_bias, self.spatial_pos = attn_bias, spatial_pos
        self.mask = mask
        #self.weight = weight
        

    def to(self, device):
        self.idx = self.idx.to(device)
        self.in_degree, self.out_degree = self.in_degree.to(
            device), self.out_degree.to(device)
        self.x, self.y = self.x.to(device), self.y.to(device)
        self.attn_bias, self.spatial_pos = self.attn_bias.to(
            device), self.spatial_pos.long().to(device)
        self.mask = self.mask.to(device)
        #self.weight = self.weight.to(device)
        
        return self

    def __len__(self):
        return self.in_degree.size(0)


def collator(items, mode='train', max_node=512, multi_hop_max_dist=20, spatial_pos_max=20):
    items = [
        item for item in items if item is not None and item.x.size(0) <= max_node]

    items = [(item.idx, 
                item.attn_bias, 
                item.spatial_pos, 
                item.in_degree,
              item.out_degree, 
              item.x, 
              item.y, 
              item.train_mask, 
              item.test_mask, 
              item.val_mask) for item in items]
    idxs, attn_biases, spatial_poses, in_degrees, out_degrees, xs, ys, train_masks, test_masks, val_masks = zip(
        *items)

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][spatial_poses[idx] >= spatial_pos_max] = float('-inf')
    max_node_num = max(i.size(0) for i in xs)
    #max_dist = max(i.size(-2) for i in edge_inputs)
    if mode == 'train':
        mask = torch.cat([pad_1d_unsqueeze_mask(i, max_node_num) for i in train_masks])
    elif mode == "val":
        mask = torch.cat([pad_1d_unsqueeze_mask(i, max_node_num) for i in val_masks])
    else:
        mask = torch.cat([pad_1d_unsqueeze_mask(i, max_node_num) for i in test_masks])
    y = torch.cat([pad_1d_unsqueeze_y(i, max_node_num) for i in ys])
  
    #weight = torch.cat([pad_1d_unsqueeze_y(i, max_node_num) for i in weights]) 
    #print(mask.shape)
    #print(y.shape)
    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
    #print(x.shape)
    # edge_input = torch.cat([pad_3d_unsqueeze(
    #     i, max_node_num, max_node_num, max_dist) for i in edge_inputs])
    attn_bias = torch.cat([pad_attn_bias_unsqueeze(
        i, max_node_num) for i in attn_biases])
    # attn_edge_type = torch.cat(
    #     [pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types])
    spatial_pos = torch.cat([pad_spatial_pos_unsqueeze(i, max_node_num)
                        for i in spatial_poses])
    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num)
                          for i in in_degrees])
    out_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num)
                           for i in out_degrees])
    return Batch(
        idx=torch.LongTensor(idxs),
        attn_bias=attn_bias,
        #attn_edge_type=attn_edge_type,
        spatial_pos=spatial_pos,
        in_degree=in_degree,
        out_degree=out_degree,
        x=x,
        #edge_input=edge_input,
        y=y,
        mask=mask,
        #weight=weight
    )
