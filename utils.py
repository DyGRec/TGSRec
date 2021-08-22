import numpy as np
import pandas as pd

### Utility function and class
class EarlyStopMonitor(object):
    def __init__(self, max_round=3, higher_better=True, tolerance=1e-3):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        self.epoch_count += 1
        
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1
        return self.num_round >= self.max_round

class RandEdgeSampler(object):
    def __init__(self, src_list, dst_list, ts_list):
        self.edges = {}
        for u, i in zip(src_list, dst_list):
            if u in self.edges:
                self.edges[u].add(i)
            else:
                self.edges[u] = set([i])
        self.src_list = np.unique(src_list)
        self.dst_list = np.unique(dst_list)
        df = pd.DataFrame({'u': src_list, 'i': dst_list, 'ts': ts_list})

        self.items_popularity = df.groupby('i')['i'].count().to_dict()
        self.time_sorted_df = df.sort_values(by=['ts'])

    def sample(self, size):
        src_index = np.random.randint(0, len(self.src_list), size)
        dst_index = np.random.randint(0, len(self.dst_list), size)
        return self.src_list[src_index], self.dst_list[dst_index]

    def sample_neg(self, src_list):
        dst_set = set(self.dst_list)
        neg_dst = np.zeros(len(src_list), dtype=int)
        for ind, u in enumerate(src_list):
            random_neg = np.random.choice(list(dst_set-self.edges[u]), 1)[0]
            neg_dst[ind] = random_neg
        return neg_dst

    def popularity_based_sample_neg(self, src_list):
        dst_set = set(self.dst_list)
        neg_dst = np.zeros(len(src_list), dtype=int)
        for ind, u in enumerate(src_list):
            neg_candidate = list(dst_set - self.edges[u])
            neg_candidate_pop_items = []
            neg_candidate_pop = []
            for neg_item in neg_candidate:
                neg_candidate_pop_items.append(neg_item)
                neg_candidate_pop.append(self.items_popularity[neg_item])

            total_popularity = sum(neg_candidate_pop)
            neg_candidate_pop_prob = [pop / total_popularity for pop in neg_candidate_pop]

            random_neg = np.random.choice(neg_candidate_pop_items, size=1, p=neg_candidate_pop_prob)
            neg_dst[ind] = random_neg
        return neg_dst

    def timelypopularity_based_sample_neg(self, src_list, ts_list):
        neg_dst = np.zeros(len(src_list), dtype=int)
        range_ind = 1500
        all_ts = self.time_sorted_df['ts'].values
        for ind, u in enumerate(src_list):
            ts_cut = ts_list[ind]
            min_ts_diff_ind = np.argmin(abs(all_ts - ts_cut))
            prev_min_ind = max(0, min_ts_diff_ind - range_ind)
            later_max_ind = min(self.time_sorted_df.shape[0], min_ts_diff_ind + range_ind)

            timely_selected_df = self.time_sorted_df.iloc[prev_min_ind:later_max_ind, :]

            selected_items_popularity = timely_selected_df.groupby('i')['i'].count().to_dict()
            neg_candidate = list(set(list(selected_items_popularity.keys())) - self.edges[u])
            if len(neg_candidate) == 0:
                print(selected_items_popularity)
                print(self.edges[u])
                print(prev_min_ind, later_max_ind)
            neg_candidate_pop_items = []
            neg_candidate_pop = []
            for neg_item in neg_candidate:
                neg_candidate_pop_items.append(neg_item)
                neg_candidate_pop.append(selected_items_popularity[neg_item])
            
            total_popularity = sum(neg_candidate_pop)
            neg_candidate_pop_prob = [pop / total_popularity for pop in neg_candidate_pop]

            random_neg = np.random.choice(neg_candidate_pop_items, size=1, p=neg_candidate_pop_prob)
            neg_dst[ind] = random_neg

        return neg_dst
