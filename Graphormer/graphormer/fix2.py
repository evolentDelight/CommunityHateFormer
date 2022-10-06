import pandas as pd
import torch 
import tqdm 
import numpy as np
import pyximport

pyximport.install(setup_args={'include_dirs': np.get_include()})
import algos


skip_groups = ['c2a3ty', 'e4k2mp', 'eg09iq', 'hjn8o0', 'hm2woa', 'hujoqu', 'hwdvsq']

# data = data.groupby('link_id').filter(lambda x: len(x) > 2)
# data['idx_path'] = data.groupby('link_id').ngroup()
# data['idx_path'] = 'data_' + data['idx_path'].astype(str) + '.pt'

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

skipped = []
data = pd.read_parquet('/home/l2hebert/reddit-text/data_no_edge.parquet')
res = []
idx_add = 0
idx=0
for path, group in tqdm.tqdm(data.groupby('path'), total=(len(data['path'].unique()))):
    try:
        x = torch.load(path)
        if x.edge_index.max() > x.x.size(0):
            skipped += [path]
            print('skipped', path)
            continue
        x = preprocess_item(x)
        #x = torch.save(x ,path)
        group['idx_path'] = f'data_{idx}.pt'
        idx += 1
        res += [group]
    except:
        skipped += [path]
        print('skipped2', path)

data = pd.concat(res)
data = data.loc[~data['idx_path'].isin(skipped)]
data.to_parquet('/home/l2hebert/reddit-text/data_no_edge.parquet')