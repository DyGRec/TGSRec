import json
import numpy as np
import pandas as pd
import gzip

def preprocess(data_name, meta_path):
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_l = []
    idx_list = []
    u_map = {}
    i_map = {}

    u_ind = 0
    i_ind = 0
    with open(data_name, 'r') as f:
        for line in f:
            one_interaction = line.strip().split("\t")
            u = int(one_interaction[0])
            i = int(one_interaction[1])

            if u not in u_map:
                u_map[u] = u_ind
                u_ind += 1
            if i not in i_map:
                i_map[i] = i_ind
                i_ind += 1

    i_meta_map = {}
    with open(meta_path, 'r', encoding='latin-1') as f:
        for line in f:
            one_item_meta = line.strip().split("|")
            item_id = int(one_item_meta[0])
            if item_id not in i_meta_map:
                i_meta_map[item_id] = one_item_meta[1]

    data = []
    with open(data_name,'r') as f:
        for idx, line in enumerate(f):
            one_interaction = line.strip().split("\t")
            data.append([int(one_interaction[0]), int(one_interaction[1]), float(one_interaction[3])])
    sorted_data = sorted(data, key=lambda x:x[2])
    
    for idx, eachinter in enumerate(sorted_data):
        u = u_map[eachinter[0]]
        i = i_map[eachinter[1]]
        ts = eachinter[2]
        feat = np.array([0 for _ in range(8)])
        u_list.append(u)
        i_list.append(i)
        ts_list.append(ts)
        label_list.append(1)
        idx_list.append(idx)
            
        feat_l.append(feat)

    user_ind_id_map = {v:k for k, v in u_map.items()}
    item_ind_id_map = {v:{'item_id': k, 'title': i_meta_map.get(k, '')} for k, v in i_map.items()}
    return pd.DataFrame({'u': u_list, 
                         'i':i_list, 
                         'ts':ts_list, 
                         'label':label_list, 
                         'idx':idx_list}), np.array(feat_l), user_ind_id_map, item_ind_id_map



def reindex(df):
    assert(df.u.max() - df.u.min() + 1 == len(df.u.unique()))
    assert(df.i.max() - df.i.min() + 1 == len(df.i.unique()))
    
    upper_u = df.u.max() + 1
    new_i = df.i + upper_u
    
    new_df = df.copy()
    print(new_df.u.max())
    print(new_df.i.max())
    
    new_df.i = new_i
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1
    
    print(new_df.u.max())
    print(new_df.i.max())
    
    return new_df



def run(data_name):
    PATH = './datasets/ml-100k/u.data'
    meta_path = './datasets/ml-100k/u.item'
    OUT_DF = './processed/ml_{}.csv'.format(data_name)
    #OUT_FEAT = './processed/ml_{}.npy'.format(data_name)
    #OUT_NODE_FEAT = './processed/ml_{}_node.npy'.format(data_name)
    u_map_file = './processed/{}_u_map.json'.format(data_name)
    i_map_file = './processed/{}_i_map.json'.format(data_name)
    
    df, feat, u_ind_id_map, i_ind_id_map = preprocess(PATH, meta_path)
    with open(u_map_file, 'w') as f:
        f.write(json.dumps(u_ind_id_map, sort_keys=True, indent=4))
    with open(i_map_file, 'w') as f:
        f.write(json.dumps(i_ind_id_map, sort_keys=True, indent=4))

    new_df = reindex(df)
    
    #print(feat.shape)
    #empty = np.zeros(feat.shape[1])[np.newaxis, :]
    #feat = np.vstack([empty, feat])
    #
    #max_idx = max(new_df.u.max(), new_df.i.max())
    #rand_feat = np.zeros((max_idx + 1, feat.shape[1]))
    #
    #print(feat.shape)
    new_df.to_csv(OUT_DF, index=False)
    #np.save(OUT_FEAT, feat)
    #np.save(OUT_NODE_FEAT, rand_feat)
    


run('ml-100k')
