import torch
import pandas as pd 
from transformers import AutoModel, AutoTokenizer
import os.path as osp
from glob import glob
from tqdm import tqdm
from torch_geometric.data import Dataset
from torch_geometric.transforms import RandomNodeSplit

class RedditDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=RandomNodeSplit("train_rest", num_val=0.10, num_test=0.20)):
        #self.pandas_data = pd.read_parquet(glob('/home/l2hebert/reddit-text2/*.parquet'))
        
        self.pandas_data = pd.read_parquet('/home/l2hebert/reddit-text/data_no_edge.parquet')
        # self.pandas_data = self.pandas_data.groupby('link_id').filter(lambda x: len(x) > 2)
        # self.pandas_data['idx_path'] = self.pandas_data.groupby('link_id').ngroup()
        # self.pandas_data['idx_path'] = 'data_' + self.pandas_data['idx_path'].astype(str) + '.pt' 
        #         
        #self.pandas_data = self.pandas_data.loc[self.pandas_data['subreddit'] == 'The_Donald']
        self.weights = [0.2743,   1.0954,   4.2365,   5.0225, 15.000]
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
        return self.pandas_data['idx_path'].unique().tolist() 

    def process(self):
        print("DIR", self.processed_dir)
        idx = 0
        groups = self.pandas_data.groupby('link_id')
        known_files = glob('/home/l2hebert/dev/processed-graphs/processed/*.pt')
        
        print(len(list(set(self.processed_file_names) - set(known_files))))
        print(len(set(self.processed_file_names)))
        print(len(set(known_files)))

        print(known_files[:3])
        print(self.processed_file_names[:3])


        
        #print(set(self.processed_file_names))
        
        with torch.no_grad():
            transformer = AutoModel.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain").to('cuda:0')
            tokenizer = AutoTokenizer.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain", truncation=True, padding=True, local_files_only=True)
            
            token = 'OTE3ODczMTY4MzA2MzAzMDc2.Ya_CAA.-0oJf9n_QcgMnPtTkrvEorexdRU'
            channel_id = '917867844220383233'
            iterator = tqdm(groups, total=groups.ngroups, desc='processing graphs')
            
            for name, group in iterator:
                #group = groups.get_group('bva9ed')
                path = str(group['path'].iloc[0])
                body = list(group['body'])
                
                iterator.set_postfix({'length': len(body), 'name': name}, refresh=False)
                if f'/home/l2hebert/dev/processed-graphs/processed/data_{idx}.pt' in known_files:
                    idx += 1
                    continue
                else:
                    known_files = [] # just to speed things up
                if name in ['bvb6k8', 'bva9ed', 'bvdtn4']: # This just causes issues for some reason
                    continue

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
                data = torch.load(path)
                data.x = torch.cat((data.x, res), dim=1) # Verify this works
                #del res
                data.edge_index = data.edge_index.long()

                if data.edge_index.max() > data.x.size(0):
                    continue
              
                #data.edge_attr = torch.ones((data.edge_index.shape[1], 1)).short().to('cuda:0')
                #print(data)
                #print(data.edge_index)
                
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                data = self.pre_transform(data)
                data.idx = idx

                try:
                    torch.save(data, f'/home/l2hebert/dev/processed-graphs/processed/data_{idx}.pt')
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
            data = torch.load(f'/home/l2hebert/dev/processed-graphs/processed/' + self.processed_file_names[idx])
        except Exception as e:
            print("GOT EXCEPTION")
            raise e
        

        data.y[(data.y == torch.Tensor([1., 0., 0., 0., 0.])).all(dim=1)] = torch.Tensor([0, 0, 0, 0, 0])
        data.y[(data.y == torch.Tensor([0., 1., 0., 0., 0.])).all(dim=1)] = torch.Tensor([1, 0, 0, 0, 0])
        data.y[(data.y == torch.Tensor([0., 0., 1., 0., 0.])).all(dim=1)] = torch.Tensor([1, 1, 0, 0, 0])
        data.y[(data.y == torch.Tensor([0., 0., 0., 1., 0.])).all(dim=1)] = torch.Tensor([1, 1, 1, 0, 0])
        data.y[(data.y == torch.Tensor([0., 0., 0., 0., 1.])).all(dim=1)] = torch.Tensor([1, 1, 1, 1, 0])
        data.y = data.y[:, :4]

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
