import pandas as pd

data = {
    'id': [1, 2, 3, 4, 5, 6],
    'parent_id': [-1, 1, 1, 3, 3, 4],
    'score': [0.80, -0.7, 0.0, 0.5, 0.9, 0.75],
    'upvotes': [3, 10, 5, 5, -20, 15]
}

df = pd.DataFrame(data)

leafs = df.loc[~df['id'].isin(df['parent_id'])]['id']
ids = list(leafs.index)
leafs = list(leafs)
while len(leafs) != 0:
    curr = leafs.pop(0)
    curr_index = ids.pop(0)
    curr_data = df.loc[df['id'] == curr]

    children = df.loc[df['parent_id'] == curr]
    parent = df.loc[df['id'] == int(curr_data['parent_id'])]
    if len(children) != 0:
        
        children = list(0.25 * children['weighted_score'])
    else:
        children = []
    
    #print(float(curr_data['score']), float(curr_data['upvotes']), float(parent['score']), float(parent['upvotes']))
   
    new_score = 0
    for val in children:
        new_score += val

    if int(curr_data['parent_id']) != -1:
        new_score += 0.50 * float(curr_data['score']) * float(curr_data['upvotes'])
        df.at[curr_index, 'weighted_score'] = new_score
        parent_id = int(parent['id'])
        if not parent_id in leafs:
            leafs.append(parent_id)
            ids.append(int(parent.index[0]))
    else:
        new_score += 0.50 * float(curr_data['score']) * float(curr_data['upvotes'])
        df.at[curr_index, 'weighted_score'] = new_score

print(df)