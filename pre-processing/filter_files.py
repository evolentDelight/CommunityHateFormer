import dask
import dask.dataframe as dd
from dask.distributed import Client 
from functools import partial
import numpy as np
import torch

def main(): 
    #df = dd.read_csv("/DATA/Reddit/raw/*", on_bad_lines='warn', engine='python', blocksize="10MB", dtype=types, keep_default_na=False, converters=converters, names=names)
    wanted = [
        'body',
        'score',
        'parent_id',
        'id',
        'created_utc',
        'link_id',
        'gilded',
        'subreddit',
        'author_flair_text',
        'author'
    ]
    #df = dd.read_json('/DATA/Reddit/data/RC*', blocksize='64MB')
    #df = df[wanted]
    df = dd.read_parquet('/home/l2hebert/reddit/*', columns=wanted)
    print(df.columns)
    target_reddits = [
        'The_Donald',
        'climateskeptics',
        'AmItheAsshole',
        'politics',
        'IAmA',
        'iama'
    ]
    #df = df[df['subreddit'].isin(target_reddits)]
    

    # Drop data? 
    


    df.to_parquet("/home/l2hebert/reddit-processed")

if __name__ == "__main__":
    client = Client(n_workers=24)
    print(client.dashboard_link)
    main()
    client.close()