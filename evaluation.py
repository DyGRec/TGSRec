import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

import multiprocessing
import metrics
import torch
from tqdm import tqdm


Ks = [1, 5, 10, 20, 40, 50, 60, 70, 80, 90, 100]

def eval_one_user(x):
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
            'hit_ratio': np.zeros(len(Ks)), 'auc': 0., 'mrr': 0.}
    preds = np.transpose(x[0])

    pos_label = np.ones(1)

    num_preditems = x[1]

    uit = x[2]
    rec_items = x[3]

    num_neg_sample_items = x[4]
    num_candidate_items = x[5]

    labels = np.zeros(num_preditems)
    labels[0] = 1
    scaler = MinMaxScaler()
    posterior = np.transpose(scaler.fit_transform(np.transpose([preds])))[0]
    r = []
    rankeditems = np.argsort(-preds)[:max(Ks)]
    for i in rankeditems:
        if i == 0:
            r.append(1)
        else:
            r.append(0)
    if num_neg_sample_items != -1:
        r = rank_corrected(np.array(r), num_preditems, num_candidate_items)

    precision, recall, ndcg, hit_ratio = [], [], [], []
    for K in Ks:
        precision.append(metrics.precision_at_k(r, K))
        recall.append(metrics.recall_at_k(r, K, 1))
        ndcg.append(metrics.ndcg_at_k(r, K))
        hit_ratio.append(metrics.hit_at_k(r, K))
    auc = metrics.auc(ground_truth=labels, prediction=posterior)
    mrr = metrics.mrr(r)


    result['precision'] += precision
    result['recall'] += recall
    result['ndcg'] += ndcg
    result['hit_ratio'] += hit_ratio
    result['auc'] += auc
    result['mrr'] += mrr

    return (result, rankeditems[:max(Ks)], uit, rec_items)


def rank_corrected(r, m, n):
    pos_ranks = np.argwhere(r==1)[:,0]
    corrected_r = np.zeros_like(r)
    for each_sample_rank in list(pos_ranks):
        corrected_rank = int(np.floor(((n-1)*each_sample_rank)/m))
        if corrected_rank >= len(corrected_r) - 1:
            continue
        corrected_r[corrected_rank] = 1
    assert sum(corrected_r) <= 1
    return corrected_r


def eval_users(tgrec, src, dst, ts, train_src, train_dst, args):
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
            'hit_ratio': np.zeros(len(Ks)), 'auc': 0., 'mrr': 0.}
    cores = multiprocessing.cpu_count() // 2
    userset = set(src)
    train_itemset = set(train_dst)
    pos_edges = {}
    for u, i, t in zip(src, dst, ts):
        if i not in train_itemset:
            continue
        if u in pos_edges:
            pos_edges[u].add((i, t))
        else:
            pos_edges[u] = set([(i, t)])
    train_pos_edges = {}
    for u, i in zip(train_src, train_dst):
        if u in train_pos_edges:
            train_pos_edges[u].add(i)
        else:
            train_pos_edges[u] = set([i])

    pool = multiprocessing.Pool(cores)
    batch_users = 5

    preds_list = []
    preds_len_preditems = []
    preds_uit = []
    preds_rec_items = []
    preds_sampled_neg = []
    preds_num_candidates = []

    test_outputs = []

    num_interactions = 0
    num_test_instances = 0
    with torch.no_grad():
        tgrec = tgrec.eval()
        batch_src_l = []
        batch_test_items = []
        batch_ts = []
        #batch_len = []
        batch_i = 0
        for u, i, t in zip(src, dst, ts):
            num_test_instances += 1
            if u not in train_src or i not in train_itemset or u not in pos_edges:
                continue
            num_interactions += 1
            batch_i += 1

            pos_items = [i]
            pos_ts = [t]
            src_l = [u for _ in range(len(pos_items))]
            pos_label = np.ones(len(pos_items))

            interacted_dst = train_pos_edges[u]

            neg_candidates = list(train_itemset - set(pos_items) - interacted_dst)
            if args.negsampleeval == -1:
                neg_items = neg_candidates
            else:
                neg_items = list(np.random.choice(neg_candidates, size=args.negsampleeval, replace=False))
            #neg_items = list(train_itemset - set(pos_items))
            neg_ts = [t for _ in range(len(neg_items))]
            neg_src_l = [u for _ in range(len(neg_items))]

            batch_src_l += src_l + neg_src_l
            batch_test_items += pos_items + neg_items
            batch_ts += pos_ts + neg_ts
            #batch_len.append(len(src_l+neg_src_l))

            test_items = np.array(batch_test_items)
            test_ts = np.array(batch_ts)
            test_src_l = np.array(batch_src_l)

            pred_scores = tgrec(test_src_l, test_items, test_ts, args.n_degree)
            preds = pred_scores.cpu().numpy()
            #start_ind = 0
            #for i_len in batch_len:
            preds_list.append(preds)
            preds_len_preditems.append(len(src_l+neg_src_l))
            preds_uit.append((u,i,t))
            rec_items = []
            rec_items += pos_items + neg_items
            preds_rec_items.append(rec_items)
            preds_sampled_neg.append(args.negsampleeval)
            preds_num_candidates.append(len(pos_items+neg_candidates))
                #start_ind = i_len
            batch_src_l = []
            batch_test_items = []
            batch_ts = []
            #batch_len = []

            if len(preds_list) % batch_users == 0 or num_test_instances == len(ts):

                batchset_predictions = zip(preds_list, preds_len_preditems, preds_uit, preds_rec_items, preds_sampled_neg, preds_num_candidates)
                batch_preds = pool.map(eval_one_user, batchset_predictions)
                for oneresult in batch_preds:
                    re = oneresult[0]
                    result['precision'] += re['precision']
                    result['recall'] += re['recall']
                    result['ndcg'] += re['ndcg']
                    result['hit_ratio'] += re['hit_ratio']
                    result['auc'] += re['auc']
                    result['mrr'] += re['mrr']

                    uit = oneresult[2]
                    pred_rank_list = oneresult[1]
                    rec_items = oneresult[3]

                    one_pred_result = {"u_ind": int(uit[0]), "u_pos_gd": int(uit[1]), "timestamp": float(uit[2])}
                    one_pred_result["predicted"] = [int(rec_items[int(rec_ind)]) for rec_ind in pred_rank_list]
                    test_outputs.append(one_pred_result)


                preds_list = []
                preds_len_preditems = []
                preds_uit = []
                preds_rec_items = []
                preds_sampled_neg = []
                preds_num_candidates = []
                batch_src_l = []
                batch_test_items = []
                batch_ts = []
                #batch_len = []
            #pred_prob = pred_scores.sigmoid()

            #neg_label = np.zeros(len(neg_items))

            #preds = pred_scores.cpu().numpy()
            #preds_list.append(preds)
            #preds_len_preditems.append(len(neg_items)+1)
            #scaler = MinMaxScaler()
            #posterior = np.transpose(scaler.fit_transform(np.transpose([preds])))[0]
            #posterior = -np.sort(-posterior)
            ##posterior = pred_prob.cpu().numpy()
            #labels = np.concatenate([pos_label, neg_label])

            #r = []
            #rankeditems = list((-preds).argsort())
            #for i in rankeditems:
            #    if i == 0:
            #        r.append(1)
            #    else:
            #        r.append(0)

            #precision, recall, ndcg, hit_ratio = [], [], [], []
            #for K in Ks:
            #    precision.append(metrics.precision_at_k(r, K))
            #    recall.append(metrics.recall_at_k(r, K, len(pos_items)))
            #    ndcg.append(metrics.ndcg_at_k(r, K))
            #    hit_ratio.append(metrics.hit_at_k(r, K))
            #auc = metrics.auc(ground_truth=r, prediction=posterior)
            #mrr = metrics.mrr(r)

            #result['precision'] += precision
            #result['recall'] += recall
            #result['ndcg'] += ndcg
            #result['hit_ratio'] += hit_ratio
            #result['auc'] += auc
            #result['mrr'] += mrr


    result['precision'] /= num_interactions
    result['recall'] /= num_interactions
    result['ndcg'] /= num_interactions
    result['hit_ratio'] /= num_interactions
    result['auc'] /= num_interactions
    result['mrr'] /= num_interactions
    print('num_interactions: ', num_interactions)

    return result, test_outputs


def eval_one_epoch(hint, tgrec, sampler, src, dst, ts, label):
    val_acc, val_ap, val_f1, val_auc = [], [], [], []
    with torch.no_grad():
        tgrec = tgrec.eval()
        TEST_BATCH_SIZE=1024
        num_test_instance = len(src)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
        for k in range(num_test_batch):
            # percent = 100 * k / num_test_batch
            # if k % int(0.2 * num_test_batch) == 0:
            #     logger.info('{0} progress: {1:10.4f}'.format(hint, percent))
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance - 1, s_idx + TEST_BATCH_SIZE)
            src_l_cut = src[s_idx:e_idx]
            dst_l_cut = dst[s_idx:e_idx]
            ts_l_cut = ts[s_idx:e_idx]
            # label_l_cut = label[s_idx:e_idx]

            size = len(src_l_cut)
            dst_l_fake = sampler.sample_neg(src_l_cut)

            pos_prob, neg_prob = tgrec.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, NUM_NEIGHBORS)
            
            pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
            pred_label = pred_score > 0.5
            true_label = np.concatenate([np.ones(size), np.zeros(size)])
            
            val_acc.append((pred_label == true_label).mean())
            val_ap.append(average_precision_score(true_label, pred_score))
            val_f1.append(f1_score(true_label, pred_label))
            val_auc.append(roc_auc_score(true_label, pred_score))
    return np.mean(val_acc), np.mean(val_ap), np.mean(val_f1), np.mean(val_auc)

