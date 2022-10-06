from math import isclose
import dask.dataframe as dd
from dask.distributed import Client
from joblib.parallel import delayed
from numpy.lib.index_tricks import index_exp 
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import os 
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Pipeline, AutoModel
import transformers

from joblib import Parallel, delayed

def main(path_name, child_score_weight, curr_score_weight, parent_score_weight):

    # model = AutoModel.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain", local_files_only=True)
    
    # tokenizer = AutoTokenizer.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain", local_files_only=True)

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

    grouped = merged.groupby(by=['link_id']).mean()[['created_utc']].astype(int)
    merged = merged.merge(grouped, on='link_id', how='left')
    merged = merged.rename(columns={'created_utc_y': 'mean_utc', 'created_utc_x': 'created_utc'})
    
    # merged['']
    # data_to_write = merged[['body', 'link_id']]
    # merged[['body', 'link_id']].to_parquet('/home/l2hebert/reddit-text') # we will process this on the fly
    # merged = merged.drop('body', axis=1)

   

    # def tokenized(x):
    #     data = list(x['body'])
    #     if len(data) == 0:
    #         x['embedding'] = []
    #     else:
    #         res = tokenizer(list(x['body']), padding='max_length', truncation=True, return_tensors='pt')
    #         with torch.no_grad():
    #             res = model(**res)
              
    #             x['embedding'] = res.pooler_output.tolist()
         
    #     return x

    # merged = merged.map_partitions(tokenized)
    # merged['index'] = 1 

    # grouped = merged.groupby(by=['link_id']).mean()[['created_utc']].astype(int)
    # merged = merged.merge(grouped, on='link_id', how='left')
    # merged = merged.rename(columns={'created_utc_y': 'mean_utc', 'created_utc_x': 'created_utc'})
    
  
 
    # merged['index'] = merged.groupby(by=['link_id']).cumsum()['index']
    merged['path'] = f'/home/l2hebert/reddit-graph/{path_name}/' + merged['subreddit'] + '/' + merged['mean_utc'].astype(str) + '/' + merged['link_id'] + '.pt'

    # merged = merged.persist()
    # merged[['body', 'path', 'link_id', 'index']].to_parquet('/home/l2hebert/reddit-text2', schema="infer") # lets pray its the same order
    # merged = merged.drop(['body'], axis=1)

    merged = merged.compute()
    
    client.close()
    
    grouped = merged.groupby(by=['link_id'])
    print('total groups', grouped.ngroups)
    token = 'OTE3ODczMTY4MzA2MzAzMDc2.Ya_CAA.-0oJf9n_QcgMnPtTkrvEorexdRU'
    channel_id = '917867844220383233'
    
    
    # t = tqdm(total=grouped.ngroups)
    

    def compute_weighted_scores(x, mapping):
        leafs = x.loc[(~x['index'].isin(x['parent_id']))]
        leafs = leafs.loc[leafs['parent_id'] != -1]['index']
        ids = list(leafs.index)
        leafs = list(leafs)

        while len(leafs) != 0:
            curr = leafs.pop(0)
            curr_index = ids.pop(0)
            curr_data = x.loc[x['index'] == curr]
            if len(curr_data) != 1:
                print('CURR DATA LENGTH')
                print(curr_data)
                print(x)

            children = x.loc[x['parent_id'] == curr]
            
            if len(children) != 0:
                subtree_length = children['subtree_length'].max() + 1
                children = list(child_score_weight * children['weighted_score'])
                
            else:
                subtree_length = 0
                new_score = 0
                children = []
            
            #print(float(curr_data['score']), float(curr_data['upvotes']), float(parent['score']), float(parent['upvotes']))
        
            new_score = 0
            for val in children:
                new_score += val

            if int(curr_data['parent_id']) != -1:
                parent = x.loc[x['index'] == int(curr_data['parent_id'])]
                if len(parent) != 1:
                    print("PARENT DATA LENGTH")
                    print(parent)
                    print("MAPPING")
                    print(mapping)
                    print("LOOKING FOR ", curr_data['parent_id'])
                    print(x)
                    raise Exception
                new_score += parent_score_weight * float(parent['comp_score']) * float(parent['score'])
                new_score += curr_score_weight * float(curr_data['comp_score']) * float(curr_data['score']) # worth 1x as much
                
                parent_id = int(parent['index'])
                if not parent_id in leafs:
                    leafs.append(parent_id)
                    ids.append(int(parent.index[0]))
            else:
                new_score += 0.75 * float(curr_data['comp_score']) * float(curr_data['score'])
            
            x.at[curr_index, 'weighted_score'] = new_score
            x.at[curr_index, 'subtree_length'] = subtree_length
        
        # multiple -1 should all influence each other as a proxy for no head comment
        head_nodes = x.loc[x['parent_id'] == -1]
        if len(head_nodes) != 1:
            ids = list(head_nodes['index'])
            temp_scores = []
            subtree_length = []
            for id in ids:
                curr = head_nodes.loc[head_nodes['index'] == id]
            
                if 'weighted_score' not in curr or pd.isna(curr['weighted_score'].iloc[0]):
                    temp_scores += [curr['score'].iloc[0] * curr['comp_score'].iloc[0] * 0.75]
                else:
                    temp_scores += [curr['weighted_score'].iloc[0]]
                
                if 'subtree_length' not in curr or pd.isna(curr['subtree_length'].iloc[0]):
                    subtree_length += [0]
                else:
                    subtree_length += [curr['subtree_length'].iloc[0]] 
                # if 'weighted_score' not in head_nodes.columns: # also means no children nodes, just a bunch of stranglers
                #     #head_nodes['subtree_length'] = 0
                #     x.loc[x['parent_id'] == -1, 'subtree_length'] = 0
                #     head_nodes['temp_weighted_score'] = head_nodes['score'] * head_nodes['comp_score'] * 0.75
                # else:
                #     head_nodes['temp_weighted_score'] = head_nodes['weighted_score']
            # head_nodes.loc['temp_weighted_score', :] = temp_scores 
            # head_nodes.loc['subtree_length', :] = subtree_length
            # leafs = list(head_nodes['index'])
            
            #for curr_index, score, subtree_length in zip(ids, temp_scores, subtree_length):
            for i in range(len(ids)):
                curr_index = ids[i]
                score = temp_scores[i]
                curr_subtree_length = subtree_length[i]
                rest = [x for k,x in enumerate(temp_scores) if k!=i]
                # curr = head_nodes.loc[head_nodes['index'] == id]
                # # score = float(curr['temp_weighted_score'])
                # # subtree_length = int(curr['subtree_length'])
                # #curr_index = int(curr['index'])
                # if type(curr) == pd.Series:
                #     print("type curr error")
                #     print(curr)
                #     print('doing workaround, not ideal')
                #     rest = head_nodes
                #     curr = curr.iloc[0]
                #     new_score = curr['temp_weighted_score']
                
                #rest = list(head_nodes.loc[head_nodes['index'] != curr_index]['temp_weighted_score'])
                new_score = float(score)
                for score in rest:
                    new_score += child_score_weight * score
                    
                
                x.loc[x['index'] == curr_index, 'weighted_score'] = new_score
                x.loc[x['index'] == curr_index, 'subtree_length'] = curr_subtree_length
            
            
        
        # Convert scores to catagories [normal, 0-5, 5-20, 20 - 500, 500+]

    def compute_depth(x: pd.DataFrame):
        x['depth'] = 0
        roots = list(x.loc[x['parent_id'] == -1]['index'])
        root_depth = list(x.loc[x['parent_id'] == -1]['depth'])
        # ids = list(roots.index)
        #roots = list(roots)
        
        while len(roots) != 0:
            curr = roots.pop(0)
            depth = root_depth.pop(0) + 1
           
            # curr_index = ids.pop(0)
            children = x.loc[x['parent_id'] == curr]

            for i, child in children.iterrows():

                x.at[i, 'depth'] = depth 
                root_depth.append(depth)
                roots.append(int(child['index']))
                # ids.append(int(child.index))

    def convert_to_catagories(x: pd.Series):
        normal = [1 if k <= 0 else 0 for k in x]
        small = [1 if 0 < k < 5 else 0 for k in x]
        mid = [1 if 5 <= k < 20 else 0 for k in x]
        large = [1 if 20 <= k < 500 else 0 for k in x]
        massive = [1 if 500 <= k else 0 for k in x]

        return normal, small, mid, large, massive

    global counter 
    counter = 0
    global culled 
    culled = 0


    def grouped_apply(x, name):
        # global counter
        # global culled
        # counter += 1
        # if counter % 10 == 0:
        #     t.set_postfix({'processing': name, 'culled': culled}, refresh=False)
        #     t.update(10)
        # if counter < 202750:
        #     subreddit = str(x['subreddit'].iloc[0])
        #     created_utc = str(int((x['created_utc'].sum())/len(x)))
        #     x['path'] = '/home/l2hebert/reddit-graph/' + subreddit + '/' + created_utc + '/' + name + '.pt'
        #     return x

        x['index'] = 1 
        x['index'] = x['index'].cumsum()
        x['index'] = x['index'] - 1 # so we start from 0
        mapping = x[['index', 'id']].set_index('id').to_dict('index')
        mapping = {id: value['index'] for id, value in mapping.items()}
        mapping['-1'] = -1
        mapping[name] = -1

        x = x.replace({'parent_id': mapping})
        x['parent_id'] = x['parent_id'].apply(lambda x: x if type(x) != str else -1)
        
        #x['parent_id'] = pd.to_numeric(x['parent_id'], errors='coerce', downcast='integer').fillna(-1) # replace all missing parents as if the nodes pointing to them are the head. It's not perfect
        #x['index'] = pd.to_numeric(x['index']) # to fix floating point bs
        
        x['parent_id'] = x['parent_id'].astype(np.int64)
        # x['index'] = x['index'].astype(np.int64)
        if len(x['index'].unique()) != len(x):
            print('INDEX LENGTH')
            print(len(x['index'].unique()), ' vs ', len(x))
            print(x)

        compute_weighted_scores(x, mapping)
        compute_depth(x)
        # print('DEPTH')
        # print(x.loc[x['depth'] != 0])

        #largest_length = x['subtree_length'].max()
        before = len(x)
        # print(x['subtree_length'])

        x = x.loc[(x['depth'] < 5) & (x['subtree_length'] > 1)]

        x['leaf'] = ~(x['id'].isin(x['parent_id']))

        #x = x.loc[(x['subtree_length'] > 2) | ((x['depth'] < 3) & ((x['score'] > 2) | (x['score'] < -2) | (x['index'].isin(x['parent_id']))))]
        
        if len(x) < 3:
            #print('skipping due to len 0')
            x['skip'] = True
            return x
        x['skip'] = False
        # culled += before - after
        # print('sizes', before, after)
        # print(x['subtree_length'])
        
        if 'weighted_score' not in x.columns:
            print('MISSING SCORES')
            print(x)
            return
        else:
            if len(x.loc[x['weighted_score'] == np.nan]) != 0:
                print('NAN LENGTH')
                print(x)
                return

        x['new_index'] = 1 
        x['new_index'] = x['new_index'].cumsum()
        x['new_index'] = x['new_index'] - 1 # so we start from 0
        mapping = x[['new_index', 'index']].set_index('index').to_dict('index')
        #print(mapping)
        mapping = {id: value['new_index'] for id, value in mapping.items()}
        mapping['-1'] = -1
        mapping[name] = -1

        # x = x.replace({'parent_id': mapping, 'index': mapping})
        # x['parent_id'] = x['parent_id'].apply(lambda x: x if type(x) != str else -1)
        # x['parent_id'] = x['parent_id'].astype(np.int64)

        # data = torch.Tensor(x[['score']].values) # text data eventually appended at run time     
        # leaf_mask = torch.Tensor(x[['leaf']].values)
        # edge_matrix = torch.Tensor([[], []]).long()
        
        
        # for i, row in x.iterrows():
        #     if row['parent_id'] == -1:  
        #         edges = torch.Tensor([[row['index']], [row['index']]])
        #     else:
        #         edges = torch.Tensor([[row['index'], row['index'], row['parent_id']], [row['parent_id'], row['index'], row['index']]])
            
        #     edge_matrix = torch.cat((edge_matrix, edges), dim=1).long() 
        # #print(edge_matrix)
        
        # edge_features = torch.zeros((edge_matrix.shape[1], 0))

        normal, small, mid, large, massive = convert_to_catagories(x['weighted_score'])
        x['normal'] = normal 
        x['small'] = small
        x['mid'] = mid 
        x['large'] = large 
        x['massive'] = massive
        
        # data = Data(data, edge_matrix, edge_features, torch.Tensor(x[['normal', 'small', 'mid']].values))
        
        # data.mask = leaf_mask
        # data.index = torch.Tensor(list(x['index']))
        # os.makedirs(f'/home/l2hebert/reddit-graph/{path_name}/' + str(x['subreddit'].iloc[0]) + '/' + str(x['mean_utc'].iloc[0]), exist_ok=True)
        # #x['path'] = '/home/l2hebert/reddit-graph/' + subreddit + '/' + created_utc + '/' + name + '.pt'
        # torch.save(data, str(x['path'].iloc[0]))
        
        return x
    #print(len(grouped.get_group('bs9yau')))
    #test = grouped_apply(grouped.get_group('bs9yau'), 'bs9yau')
    # print(test)
    # print(test['parent_id'])
    #print(test[['index', 'parent_id', 'new_index', 'subtree_length', 'depth']])
    #print(len(test))
    def applyParallel(df_grouped):
        result_dfs = Parallel(n_jobs=-1)(delayed(grouped_apply)(group, name) for name, group in tqdm(df_grouped, total=grouped.ngroups))
        result_dfs = [x for x in result_dfs if not x['skip'].all()]
        return pd.concat(result_dfs)

    merged = applyParallel(grouped)
    #merged['label'] = merged['']
    #merged = grouped.apply(lambda k: grouped_apply(k, k.name))
    print('done merged')
    #t.close()
    merged[['index', 'body', 'path', 'link_id', 'subreddit', 'normal', 'small', 'mid', 'large', 'massive', 'score', 'weighted_score', 'depth']].to_parquet(f'/home/l2hebert/reddit-text/data-{path_name}.parquet')
    print('DONE!')
 
if __name__ == "__main__":
    client = Client(n_workers=24)
    name = "normal" # children 0.25, 0.25 current 0.25 parent
    setups = [('equal', 0.25, 0.25, 0.25)]
    for setup in setups:
        print(setup)
        main(setup[0], child_score_weight=setup[1], curr_score_weight=setup[2], parent_score_weight=setup[3])
