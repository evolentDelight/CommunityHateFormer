import glob
import warnings 
from tqdm import tqdm
import pandas as pd
from functools import partial
import numpy as np
from joblib import Parallel, delayed
def main():
    def convert_retrieved(x, type):
        try: 
            return type(x)
        except:
            return np.NAN
    types = {
        'body': 'string',
        'score_hidden': 'string',
        'archieved': 'boolean', 
        'name': 'string',
        'author': 'string',
        'author_flair_text': 'string',
        'retrieved_on': 'string',
        'created_utc': np.uint32,
        'downs': 'string',
        'subreddit_id': 'string',
        'link_id': 'string',
        'parent_id': 'string',
        'score': np.int16, 
        'controversiality': np.int16,
        'gilded': np.uint16,
        'id': 'string',
        'subreddit': 'string',
        'ups': np.int16,
        'distinguished': 'string',
        'author_flair_css_class': 'string'
    }

    converters = {
        'retrieved_on': partial(convert_retrieved, type=np.uint32), 
        'score': partial(convert_retrieved, type=np.int16),
        'ups': partial(convert_retrieved, type=np.int16),
        'created_utc': partial(convert_retrieved, type=np.uint32),
        'controversiality': partial(convert_retrieved, type=np.int16),
        'gilded': partial(convert_retrieved, type=np.uint16)
    }
    file_names = glob.glob("/DATA/Reddit/data/RS_*")
    
    wanted = [
        'title',
        'selftext',
        'subreddit',
        'score',
        'created_utc',
        'post_hint',
        'secure_media',
        'name'
    ]
    target_reddits = [
        'climateskeptics',
        'AmItheAsshole',
        'politics',
        'IAmA',
        'The_Donald'
    ]
    def convert_files(x, i, name):
        
        # #df = pd.read_csv(x, converters=converters, dtype=types)
        # print(x)
        # df = pd.read_json(x, lines=True, encoding_errors='ignore')[wanted]
        print(f"processing {name} - {i}")
        x = x[wanted]
        x = x.loc[x['subreddit'].isin(target_reddits)]

        x[wanted].to_parquet("/home/l2hebert/other-reddit/" + name.split('/')[-1] + "-" + str(i) + ".parquet")

    readers = [pd.read_json(z, lines=True, chunksize=100000) for z in file_names]
    for reader, name in zip(readers, file_names):
        for i, x in enumerate(reader):
            convert_files(x, i, name) 
    

    

if __name__ == "__main__":
    print("starting!")
    main()