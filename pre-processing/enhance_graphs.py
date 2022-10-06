import pandas as pd
from transformers import AutoModel, AutoTokenizer
import torch 
from torch_geometric.data import Data
from glob import glob
import os.path as osp
from tqdm import tqdm

transformer = AutoModel.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain").to('cuda:0')
tokenizer = AutoTokenizer.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain", truncation=True, padding=True, local_files_only=True)

df = pd.read_parquet(glob('/home/l2hebert/reddit-text2/*.parquet')).groupby('link_id')

id = 0
length = int(df.ngroups / 8)

i = 0
with torch.no_grad():
    for name, group in tqdm(df, total=length * (id + 1)):
        if i < length * id:
            i += 1
        elif i > length * (id + 1):
            break 

        body = list(group['body']) 
        path = group.iloc[0]['path']
        
        tokens = tokenizer(body, truncation=True, padding=True, return_tensors='pt')
        for key in tokens:
            tokens[key] = tokens[key].to('cuda:0')
        embedding = transformer(**tokens).pooler_output

        data = torch.load(path).to('cuda:0')

        data.x = torch.cat((data.x, embedding), dim=1)
        data.edge_index = data.edge_index.long()
        data.edge_attr = data.edge_attr.long()

        torch.save(data, osp.join('/home/l2hebert/dev/Graphormer/dataset/reddit/processed', f'data_{i}.pt'))


