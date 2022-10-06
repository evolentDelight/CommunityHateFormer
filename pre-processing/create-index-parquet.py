from math import isclose
import dask.dataframe as dd
from dask.distributed import Client
from numpy.lib.index_tricks import index_exp 
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import os 
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Pipeline, AutoModel
import transformers

def main():
    df = dd.read_parquet("~/reddit-processed/*")
    df2 = dd.read_parquet("~/reddit-processed-other/*")
    # df = df.loc[df['author'] != '[deleted]'] # We don't want this as it could break graph structure

    df = df.drop(['gilded', 'author_flair_text', 'author'], axis=1)

    df2['parent_id'] = -1
    df2['link_id'] = df2['name']
    df2 = df2.rename(columns={'name': 'id'})
    df2 = df2.drop(['post_hint', 'secure_media', 'media_text', 'title', 'selftext'], axis=1)

    df = df.append(df2)

    child_count = df.groupby(by=['link_id']).count()[['body']]

    merged = df.merge(child_count, on='link_id', how='left')
    merged = merged.rename(columns={'body_y': 'total_conversation_size', 'body_x': 'body'})
    merged = merged.loc[merged['total_conversation_size'] > 3]

    merged['parent_id'] = merged['parent_id'].apply(lambda x: x[3:], meta=str) 
    merged['link_id'] = merged['link_id'].apply(lambda x: x[3:], meta=str) 
    merged['comp_score'] = ((merged['offensive_score'] + merged['hate_score'] - 0.35) * 2) # goes to -0.7 to 1.3
    merged['created_utc'] = ((merged['created_utc'])) / 86400
    merged['created_utc'] = merged['created_utc'].astype(int)

    grouped = merged.groupby('link_id').mean()[['created_utc']].astype(int)
    merged2 = merged.merge(grouped, on='link_id', how='left')
    merged2 = merged2.rename(columns={'created_utc_y': 'mean_utc', 'created_utc_x': 'created_utc'})
    merged2['path'] = '/home/l2hebert/reddit-graph/' + merged2['subreddit'] + '/' + merged2['mean_utc'].astype(str) + '/' + merged2['link_id'] + '.pt'

    merged2[['body', 'path', 'link_id']].to_parquet('/home/l2hebert/reddit-text2', schema="infer") # lets pray its the same order
if __name__ == "__main__":
    client = Client(n_workers=4)
    main()