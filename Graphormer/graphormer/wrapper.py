# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from sys import path_importer_cache
from joblib.parallel import delayed
import torch
import numpy as np
from torch.nn.functional import embedding
import torch_geometric.datasets
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.data import Dataset, InMemoryDataset
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.lsc.pcqm4m_pyg import PygPCQM4MDataset
import pyximport
import pandas as pd
from transformers import AutoModel, AutoTokenizer, BertTokenizer, BertModel
import os.path as osp
from glob import glob
from tqdm import tqdm


pyximport.install(setup_args={'include_dirs': np.get_include()})
import algos


def convert_to_single_emb(x, offset=512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + \
        torch.arange(0, feature_num * offset, offset, dtype=torch.short)
    x = x + feature_offset
    return x


def preprocess_item(item):
    edge_index, x = item.edge_index, item.x
    N = x.size(0)

    x = convert_to_single_emb(x)

    # node adj matrix [N, N] bool
    #print('ADJ with N:', N)
    #print(N)
    adj = torch.zeros([N, N], dtype=torch.bool)
    #print(adj)
    adj[edge_index[0, :], edge_index[1, :]] = True
    #print(adj)

    # # edge feature here
    # if len(edge_attr.size()) == 1:
    #     edge_attr = edge_attr[:, None]
    # attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.short)
    # attn_edge_type[edge_index[0, :], edge_index[1, :]] = convert_to_single_emb(edge_attr) + 1 # TODO: Maybe remove this?
    
    shortest_path_result, path = algos.floyd_warshall(adj.numpy())
    #max_dist = np.amax(shortest_path_result)
    #edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.cpu().numpy())
    spatial_pos = torch.from_numpy((shortest_path_result)).short()
    attn_bias = torch.zeros(
        [N, N], dtype=torch.float)  # with graph token # EDIT + 1 removed

    # combine
    item.x = x
    item.adj = adj
    item.attn_bias = attn_bias
    #item.attn_edge_type = attn_edge_type
    item.spatial_pos = spatial_pos
    item.in_degree = adj.long().sum(dim=1).view(-1)
    item.out_degree = adj.long().sum(dim=0).view(-1)
    #item.edge_input = torch.from_numpy(edge_input).long()

    return item


class MyGraphPropPredDataset(PygGraphPropPredDataset):
    def download(self):
        super(MyGraphPropPredDataset, self).download()

    def process(self):
        super(MyGraphPropPredDataset, self).process()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            item.idx = idx
            return preprocess_item(item)
        else:
            return self.index_select(idx)


class MyPygPCQM4MDataset(PygPCQM4MDataset):
    def download(self):
        super(MyPygPCQM4MDataset, self).download()

    def process(self):
        super(MyPygPCQM4MDataset, self).process()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            item.idx = idx
            return preprocess_item(item)
        else:
            return self.index_select(idx)


class MyZINCDataset(torch_geometric.datasets.ZINC):
    def download(self):
        super(MyZINCDataset, self).download()

    def process(self):
        super(MyZINCDataset, self).process()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            item.idx = idx
            return preprocess_item(item)
        else:
            return self.index_select(idx)

class RedditDataset(Dataset):
    def __init__(self, root, transform=None, subreddit=None, variant='equal', pre_transform=RandomNodeSplit("train_rest", num_val=0.10, num_test=0.20)):
        #self.pandas_data = pd.read_parquet(glob('/home/l2hebert/reddit-text2/*.parquet'))
        print("SUBREDDIT",subreddit)
        self.pandas_data = pd.read_parquet(f'/home/l2hebert/reddit-text/data-{variant}.parquet')
        # self.pandas_data = self.pandas_data.groupby('link_id').filter(lambda x: len(x) > 2)
        # self.pandas_data['idx_path'] = self.pandas_data.groupby('link_id').ngroup()
        # self.pandas_data['idx_path'] = 'data_' + self.pandas_data['idx_path'].astype(str) + '.pt' 
        #     
        if subreddit != None:    
            self.pandas_data = self.pandas_data.loc[self.pandas_data['subreddit'] == subreddit]
        self.variant = variant
        self.weights = [0.2743,   1.0954,   4.2365,   5.0225, 15.000]
        self.pandas_data['idx'] = self.pandas_data.groupby(['link_id']).ngroup()
        self.pandas_data['idx_path'] = f'data_' + self.pandas_data['idx'].astype(str) + '.pt'
        self.ids = self.pandas_data['idx_path'].unique().tolist()
        print("LENGTH", len(self.pandas_data['idx_path'].unique().tolist()))
        super().__init__(root, transform, pre_transform)
    
    # @property
    # def pandas_data(self):
    #     print('reading data')
    #     return 

    @property
    def raw_file_names(self):
        return [idx for idx in list(self.pandas_data['idx_path'].unique())]
        #return list(self.pandas_data['path'].unique())

    @property
    def processed_file_names(self):
        return self.ids 

    def process(self):
        print("DIR", self.processed_dir)
        idx = 0
        groups = self.pandas_data.groupby('link_id')
        known_files = glob(f'/home/l2hebert/dev/processed-graphs-bert/{self.variant}/processed/*.pt')
        
        print(len(list(set(self.processed_file_names) - set(known_files))))
        print(len(set(self.processed_file_names)))
        print(len(set(known_files)))

        print(known_files[:3])
        print(self.processed_file_names[:3])


        
        #print(set(self.processed_file_names))
        
        with torch.no_grad():
            transformer = BertModel.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain").to('cuda:0')
            tokenizer = BertTokenizer.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain", truncation=True, padding=True)
            
            token = 'OTE3ODczMTY4MzA2MzAzMDc2.Ya_CAA.-0oJf9n_QcgMnPtTkrvEorexdRU'
            channel_id = '917867844220383233'
            iterator = tqdm(groups, total=groups.ngroups, desc='processing graphs')
            prefix = f'/home/l2hebert/reddit-graph/{self.variant}/'
            for name, group in iterator:
                #group = groups.get_group('bva9ed')
                path = str(group['path'].iloc[0])
                body = list(group['body'])
                
                iterator.set_postfix({'length': len(body), 'name': name}, refresh=False)
                if f'/home/l2hebert/dev/processed-graphs-bert/{self.variant}/processed/data_{idx}.pt' in known_files:
                    idx += 1
                    continue
                else:
                    known_files = [] # just to speed things up
                # if name in ['bvb6k8', 'bva9ed', 'bvdtn4']: # This just causes issues for some reason
                #     continue

                # print('LENGTH', len(body))
                # Read data from `raw_path`.
                
                res = None
                if (len(body) / 50 > 20):
                    batches = tqdm(range(0, len(body), 50), desc='big graph {' + name + '}')
                else:
                    batches = range(0, len(body), 50)
                for i in batches: 
                    temp_body = body[i:(i + 50)]
                    tokens = tokenizer(temp_body, truncation=True, padding=True, return_tensors='pt')
                    for key in tokens:
                        tokens[key] = tokens[key].to('cuda:0')
                    #print(tokens)
                    embeddings = transformer(**tokens).pooler_output

                    if res == None:
                        res = embeddings
                    else: 
                        res = torch.cat((res, embeddings), dim=0)
                        #del embeddings
                    # embeddings = embeddings.cpu()
                    del tokens
                res = res.cpu()
                try:
                    data = torch.load(path)
                except FileNotFoundError:
                    print(f'CANT FIND', path)
                    continue
                data.x = torch.cat((data.x, res), dim=1) # Verify this works
                #del res
                data.edge_index = data.edge_index.long()

                if data.edge_index.max() > data.x.size(0):
                    idx += 1
                    continue
              
                #data.edge_attr = torch.ones((data.edge_index.shape[1], 1)).short().to('cuda:0')
                #print(data)
                #print(data.edge_index)
                
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                data.idx = idx
                try:
                    data = preprocess_item(data)
                except:
                    print("skipping", name)
                    idx += 1
                    continue
                data = self.pre_transform(data)
                

                try:
                    torch.save(data, f'/home/l2hebert/dev/processed-graphs-bert/{self.variant}/processed/data_{idx}.pt')
                    idx += 1
                except IndexError:
                    print("ERROR INDEX ", name)
                
                del data
                
    def len(self):
        return len(self.processed_file_names)

    def __getitem__(self, idx):
        # if isinstance(idx, int):
        #     item = self.get(self.indices()[idx])
        #     item.idx = idx
        #     return preprocess_item(item)
        # else:
        #     return self.index_select(idx)
        try:
            data = torch.load(f'/home/l2hebert/dev/processed-graphs-bert/{self.variant}/processed/' + self.processed_file_names[idx])
        except Exception as e:
            print("GOT EXCEPTION")
            raise e
        

        # data.y[(data.y == torch.Tensor([1., 0., 0., 0., 0.])).all(dim=1)] = torch.Tensor([0, 0, 0, 0, 0])
        # data.y[(data.y == torch.Tensor([0., 1., 0., 0., 0.])).all(dim=1)] = torch.Tensor([1, 0, 0, 0, 0])
        # data.y[(data.y == torch.Tensor([0., 0., 1., 0., 0.])).all(dim=1)] = torch.Tensor([1, 1, 0, 0, 0])
        # data.y[(data.y == torch.Tensor([0., 0., 0., 1., 0.])).all(dim=1)] = torch.Tensor([1, 1, 1, 0, 0])
        # data.y[(data.y == torch.Tensor([0., 0., 0., 0., 1.])).all(dim=1)] = torch.Tensor([1, 1, 1, 1, 0])
        # data.y = data.y[:, :4]
        data.y = (torch.argmax(data.y, dim=1) + 1).float()

        # if data.y == torch.Tensor(0):
        #     data.y = torch.Tensor([0, 0, 0, 0])
        # elif data.y == torch.Tensor(1):
        #     data.y = torch.Tensor([1, 0, 0, 0])
        # elif data.y == torch.Tensor(2):
        #     data.y = torch.Tensor([1, 1, 0, 0])
        # elif data.y == torch.Tensor(3):
        #     data.y = torch.Tensor([1, 1, 1, 0])
        # else:
        #     data.y = torch.Tensor([1, 1, 1, 1])



        #data.x = data.x[:, 1:] # Hopefully this works!
        # #ys = data.y.detach().clone()
        # for i in range(len(self.weights)):
        #     data.y[data.y == i + 1] = self.weights[i] * data.y[data.y == i + 1]
        #data.weight = ys
        data.idx = idx
        return data

if __name__ == '__main__':
    for variant in ['equal', 'child-weight', 'curr-weight', 'parent-weight']:
        print(f'PROCESSING {variant}')
        data = RedditDataset(f'/home/l2hebert/dev/processed-graphs-bert/{variant}/processed/', variant=variant)
