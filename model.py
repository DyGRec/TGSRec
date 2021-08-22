import logging

import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F

class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        #self.layer_norm = torch.nn.LayerNorm(dim1 + dim2)
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        #x = self.layer_norm(x)
        h = self.act(self.fc1(x))
        return self.fc2(h)


class ScaledDotProductAttention(torch.nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(attn_dropout)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn) # [n * b, l_q, l_k]
        attn = self.dropout(attn) # [n * b, l_v, d]
                
        output = torch.bmm(attn, v)
        
        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        #output = self.layer_norm(output)
        
        return output, attn
    

class MapBasedMultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.wq_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)
        self.wk_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)
        self.wv_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)
        
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        
        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.weight_map = nn.Linear(2 * d_k, 1, bias=False)
        
        nn.init.xavier_normal_(self.fc.weight)
        
        self.dropout = torch.nn.Dropout(dropout)
        self.softmax = torch.nn.Softmax(dim=2)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.wq_node_transform(q).view(sz_b, len_q, n_head, d_k)
        
        k = self.wk_node_transform(k).view(sz_b, len_k, n_head, d_k)
        
        v = self.wv_node_transform(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        q = torch.unsqueeze(q, dim=2) # [(n*b), lq, 1, dk]
        q = q.expand(q.shape[0], q.shape[1], len_k, q.shape[3]) # [(n*b), lq, lk, dk]
        
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        k = torch.unsqueeze(k, dim=1) # [(n*b), 1, lk, dk]
        k = k.expand(k.shape[0], len_q, k.shape[2], k.shape[3]) # [(n*b), lq, lk, dk]
        
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv
        
        mask = mask.repeat(n_head, 1, 1) # (n*b) x lq x lk
        
        ## Map based Attention
        #output, attn = self.attention(q, k, v, mask=mask)
        q_k = torch.cat([q, k], dim=3) # [(n*b), lq, lk, dk * 2]
        attn = self.weight_map(q_k).squeeze(dim=3) # [(n*b), lq, lk]
        
        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn) # [n * b, l_q, l_k]
        attn = self.dropout(attn) # [n * b, l_q, l_k]
        
        # [n * b, l_q, l_k] * [n * b, l_v, d_v] >> [n * b, l_q, d_v]
        output = torch.bmm(attn, v)
        
        output = output.view(n_head, sz_b, len_q, d_v)
        
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.act(self.fc(output)))
        output = self.layer_norm(output + residual)

        return output, attn
    
def expand_last_dim(x, num):
    view_size = list(x.size()) + [1]
    expand_size = list(x.size()) + [num]
    return x.view(view_size).expand(expand_size)


class TimeEncode(torch.nn.Module):
    def __init__(self, expand_dim, factor=5):
        super(TimeEncode, self).__init__()
        #init_len = np.array([1e8**(i/(time_dim-1)) for i in range(time_dim)])
        
        time_dim = expand_dim
        self.factor = factor
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())
        
        #self.dense = torch.nn.Linear(time_dim, expand_dim, bias=False)

        #torch.nn.init.xavier_normal_(self.dense.weight)
        
    def forward(self, ts):
        # ts: [N, L]
        batch_size = ts.size(0)
        seq_len = ts.size(1)
                
        ts = ts.view(batch_size, seq_len, 1)# [N, L, 1]
        map_ts = ts * self.basis_freq.view(1, 1, -1) # [N, L, time_dim]
        map_ts += self.phase.view(1, 1, -1)
        
        harmonic = torch.cos(map_ts)

        return harmonic #self.dense(harmonic)

class DisentangleTimeEncode(torch.nn.Module):
    def __init__(self, components, expand_dim, factor=5):
        super(DisentangleTimeEncode, self).__init__()
        
        time_dim = expand_dim
        self.factor = factor
        self.components = components

        init_freq = np.zeros((self.components, time_dim))
        for i in range(self.components):
            #span_range = 2 + int(np.exp(i))
            span_range = 10 + i
            init_freq[i, :] = 1 / span_range ** np.linspace(0, (span_range - 1), time_dim)
        self.basis_freq = torch.nn.Parameter(torch.from_numpy(init_freq).float())
        self.phase = torch.nn.Parameter(torch.zeros(self.components, time_dim).float())

    def forward(self, ts):
        # ts: [N, L]
        batch_size = ts.size(0)
        seq_len = ts.size(1)

        ts = ts.view(batch_size, seq_len, 1, 1) # [N, L, 1, 1]
        map_ts = ts * self.basis_freq.view(1, self.components, -1) #[N, L, components, time_dim]
        map_ts += self.phase.view(1, self.components, -1)

        harmonic = torch.cos(map_ts)

        return harmonic
    
    
    
class PosEncode(torch.nn.Module):
    def __init__(self, expand_dim, seq_len):
        super().__init__()
        
        self.pos_embeddings = nn.Embedding(num_embeddings=seq_len, embedding_dim=expand_dim)
        
    def forward(self, ts):
        # ts: [N, L]
        order = ts.argsort()
        ts_emb = self.pos_embeddings(order)
        return ts_emb
    

class EmptyEncode(torch.nn.Module):
    def __init__(self, expand_dim):
        super().__init__()
        self.expand_dim = expand_dim
        
    def forward(self, ts):
        out = torch.zeros_like(ts).float()
        out = torch.unsqueeze(out, dim=-1)
        out = out.expand(out.shape[0], out.shape[1], self.expand_dim)
        return out


class LSTMPool(torch.nn.Module):
    def __init__(self, feat_dim, time_dim):
        super(LSTMPool, self).__init__()
        self.feat_dim = feat_dim
        self.time_dim = time_dim
        
        self.att_dim = feat_dim + time_dim
        
        self.act = torch.nn.ReLU()
        
        self.lstm = torch.nn.LSTM(input_size=self.att_dim, 
                                  hidden_size=self.feat_dim, 
                                  num_layers=1, 
                                  batch_first=True)
        self.merger = MergeLayer(feat_dim, feat_dim, feat_dim, feat_dim)

    def forward(self, src, src_t, seq, seq_t,  mask):
        # seq [B, N, D]
        # mask [B, N]
        seq_x = torch.cat([seq, seq_t], dim=2)
            
        _, (hn, _) = self.lstm(seq_x)
        
        hn = hn[-1, :, :] #hn.squeeze(dim=0)

        out = self.merger.forward(hn, src)
        return out, None
    

class MeanPool(torch.nn.Module):
    def __init__(self, feat_dim):
        super(MeanPool, self).__init__()
        self.feat_dim = feat_dim
        self.act = torch.nn.ReLU()
        self.merger = MergeLayer(feat_dim, feat_dim, feat_dim, feat_dim)
        
    def forward(self, src, src_t, seq, seq_t, mask):
        # seq [B, N, D]
        # mask [B, N]
        src_x = src
        seq_x = torch.cat([seq], dim=2) #[B, N, D]
        hn = seq_x.mean(dim=1) #[B, D]
        output = self.merger(hn, src_x)
        return output, None
    

class AttnModel(torch.nn.Module):
    """Attention based temporal layers
    """
    def __init__(self, feat_dim, time_dim, 
                 attn_mode='prod', n_head=2, drop_out=0.1):
        """
        args:
          feat_dim: dim for the node features
          time_dim: dim for the time encoding
          attn_mode: choose from 'prod' and 'map'
          n_head: number of heads in attention
          drop_out: probability of dropping a neural.
        """
        super(AttnModel, self).__init__()
        
        self.feat_dim = feat_dim
        self.time_dim = time_dim
        
        self.model_dim = (feat_dim + time_dim)
        #self.edge_fc = torch.nn.Linear(self.edge_in_dim, self.feat_dim, bias=False)

        self.merger = MergeLayer(self.model_dim, feat_dim, feat_dim, feat_dim)

        #self.act = torch.nn.ReLU()
        
        assert(self.model_dim % n_head == 0)
        self.logger = logging.getLogger(__name__)
        self.attn_mode = attn_mode
        
        if attn_mode == 'prod':
            self.multi_head_target = MultiHeadAttention(n_head, 
                                             d_model=self.model_dim, 
                                             d_k=self.model_dim // n_head, 
                                             d_v=self.model_dim // n_head, 
                                             dropout=drop_out)
            self.logger.info('Using scaled prod attention')
            
        elif attn_mode == 'map':
            self.multi_head_target = MapBasedMultiHeadAttention(n_head, 
                                             d_model=self.model_dim, 
                                             d_k=self.model_dim // n_head, 
                                             d_v=self.model_dim // n_head, 
                                             dropout=drop_out)
            self.logger.info('Using map based attention')
        else:
            raise ValueError('attn_mode can only be prod or map')
        
        
    def forward(self, src, src_t, seq, seq_t, mask):
        """"Attention based temporal attention forward pass
        args:
          src: float Tensor of shape [B, D]
          src_t: float Tensor of shape [B, Dt], Dt == D
          seq: float Tensor of shape [B, N, D]
          seq_t: float Tensor of shape [B, N, Dt]
          mask: boolean Tensor of shape [B, N], where the true value indicate a null value in the sequence.

        returns:
          output, weight

          output: float Tensor of shape [B, D]
          weight: float Tensor of shape [B, N]
        """

        src_ext = torch.unsqueeze(src, dim=1) # src [B, 1, D]
        q = torch.cat([src_ext, src_t], dim=2) # [B, 1, D + Dt] -> [B, 1, D]
        k = torch.cat([seq, seq_t], dim=2) # [B, 1, D + Dt] -> [B, 1, D]
        
        mask = torch.unsqueeze(mask, dim=2) # mask [B, N, 1]
        mask = mask.permute([0, 2, 1]) #mask [B, 1, N]

        # # target-attention
        output, attn = self.multi_head_target(q=q, k=k, v=k, mask=mask) # output: [B, 1, D + Dt], attn: [B, 1, N]
        output = output.squeeze()
        attn = attn.squeeze()

        output = self.merger(output, src)
        return output, attn


class TGRec(torch.nn.Module):
    def __init__(self, ngh_finder, n_nodes, args,
                 attn_mode='prod', use_time='time', agg_method='attn', node_dim=32, time_dim=32,
                 num_layers=3, n_head=4, null_idx=0, num_heads=1, drop_out=0.1, seq_len=None):
        super(TGRec, self).__init__()
        
        self.num_layers = num_layers 
        self.ngh_finder = ngh_finder
        self.null_idx = null_idx
        self.logger = logging.getLogger(__name__)
        #self.n_feat_th = torch.nn.Parameter(torch.from_numpy(np.random.random((n_nodes, node_dim)).astype(np.float32)))
        self.node_hist_embed = torch.nn.Embedding(n_nodes, node_dim)
        torch.nn.init.uniform_(self.node_hist_embed.weight, a=-1.0, b=1.0)
        
        self.feat_dim = node_dim
        
        self.use_time = use_time
        
        #self.n_feat_dim = self.feat_dim
        #self.e_feat_dim = self.feat_dim
        #self.model_dim = self.feat_dim
        self.n_feat_dim = node_dim
        self.model_dim = self.n_feat_dim + time_dim
        
        self.use_time = use_time
        self.time_att_weights = torch.nn.Parameter(torch.from_numpy(np.random.rand(node_dim, time_dim)).float())

        self.merge_layer = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim, self.feat_dim)
        
        if agg_method == 'attn':
            self.logger.info('Aggregation uses attention model')
            self.attn_model_list = torch.nn.ModuleList([AttnModel(self.n_feat_dim, 
                                                               time_dim,
                                                               attn_mode=attn_mode, 
                                                               n_head=n_head, 
                                                               drop_out=drop_out) for _ in range(num_layers)])
        elif agg_method == 'lstm':
            self.logger.info('Aggregation uses LSTM model')
            self.attn_model_list = torch.nn.ModuleList([LSTMPool(self.n_feat_dim,
                                                                 time_dim) for _ in range(num_layers)])
        elif agg_method == 'mean':
            self.logger.info('Aggregation uses constant mean model')
            self.attn_model_list = torch.nn.ModuleList([MeanPool(self.n_feat_dim) for _ in range(num_layers)])
        else:
        
            raise ValueError('invalid agg_method value, use attn or lstm')
        
        
        if use_time == 'time':
            self.logger.info('Using time encoding')
            self.time_encoder = TimeEncode(expand_dim=time_dim)
        elif use_time == 'pos':
            assert(seq_len is not None)
            self.logger.info('Using positional encoding')
            self.time_encoder = PosEncode(expand_dim=time_dim, seq_len=seq_len)
        elif use_time == 'empty':
            self.logger.info('Using empty encoding')
            self.time_encoder = EmptyEncode(expand_dim=time_dim)
        elif use_time == 'disentangle':
            self.logger.info('Using disentangle time encoding')
            self.time_encoder = DisentangleTimeEncode(args.disencomponents, expand_dim=time_dim)
        else:
            raise ValueError('invalid time option!')
        
        self.affinity_score = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim, 1) #torch.nn.Bilinear(self.feat_dim, self.feat_dim, 1, bias=True)
        
    def forward(self, src_idx_l, target_idx_l, cut_time_l, num_neighbors=20):
        
        src_embed = self.tem_conv(src_idx_l, cut_time_l, self.num_layers, num_neighbors)
        target_embed = self.tem_conv(target_idx_l, cut_time_l, self.num_layers, num_neighbors)
        
 
        score = self.affinity_score(src_embed, target_embed).squeeze(dim=-1)
        
        return score

    def contrast(self, src_idx_l, target_idx_l, background_idx_l, cut_time_l, num_neighbors=20):
        src_embed = self.tem_conv(src_idx_l, cut_time_l, self.num_layers, num_neighbors)
        target_embed = self.tem_conv(target_idx_l, cut_time_l, self.num_layers, num_neighbors)
        background_embed = self.tem_conv(background_idx_l, cut_time_l, self.num_layers, num_neighbors)
        pos_score = self.affinity_score(src_embed, target_embed).squeeze(dim=-1)
        neg_score = self.affinity_score(src_embed, background_embed).squeeze(dim=-1)
        return pos_score.sigmoid(), neg_score.sigmoid()

    def contrast_nosigmoid(self, src_idx_l, target_idx_l, background_idx_l, cut_time_l, num_neighbors=20):
        src_embed = self.tem_conv(src_idx_l, cut_time_l, self.num_layers, num_neighbors)
        target_embed = self.tem_conv(target_idx_l, cut_time_l, self.num_layers, num_neighbors)
        background_embed = self.tem_conv(background_idx_l, cut_time_l, self.num_layers, num_neighbors)
        pos_score = self.affinity_score(src_embed, target_embed).squeeze(dim=-1)
        neg_score = self.affinity_score(src_embed, background_embed).squeeze(dim=-1)
        return pos_score, neg_score


    def time_att_aggregate(self, node_emb, node_time_emb):
        """
        node_emb: [batch_size, node_dim] or [batch_size, L, node_dim]
        node_time_emb: [batch_size, components, time_emb]
        """
        batch_size = node_emb.shape[0]

        #[N, L(optional), node_dim] * [node_dim, time_dim] = [N, L(optional), time_dim]
        node_emb_to_time = torch.tensordot(node_emb, self.time_att_weights, dims=([-1], [0]))


        node_emb_to_time = torch.unsqueeze(node_emb_to_time, dim=-2) #[N, L(optional) 1, time_dim]
        if len(node_emb.shape) == 2:
            node_emb_to_time = torch.unsqueeze(node_emb_to_time, dim=-2) #[N, L(optional), 1, 1, time_dim]

        unnormalized_attentions = torch.sum(node_emb_to_time * node_time_emb, dim=-1) #[N, L(optional), components]

        normalized_attentions = F.softmax(unnormalized_attentions, dim=-1) #[N, L(optional), components]
        normalized_attentions = torch.unsqueeze(normalized_attentions, dim=-1) #[N, L(optional), components, 1]

        weighted_time_emb = torch.sum(normalized_attentions * node_time_emb, dim=-2) #[batch_size, L(optional), time_emb]
        return weighted_time_emb



    def tem_conv(self, src_idx_l, cut_time_l, curr_layers, num_neighbors=20):
        assert(curr_layers >= 0)
        
        device = torch.device('cuda:{}'.format(0))
    
        batch_size = len(src_idx_l)
        
        src_node_batch_th = torch.from_numpy(src_idx_l).long().to(device)
        cut_time_l_th = torch.from_numpy(cut_time_l).float().to(device)
        
        cut_time_l_th = torch.unsqueeze(cut_time_l_th, dim=1)
        # query node always has the start time -> time span == 0
        src_node_t_embed = self.time_encoder(torch.zeros_like(cut_time_l_th))
        src_node_feat = self.node_hist_embed(src_node_batch_th)
        
        if curr_layers == 0:
            return src_node_feat
        else:
            src_node_conv_feat = self.tem_conv(src_idx_l, 
                                           cut_time_l,
                                           curr_layers=curr_layers - 1, 
                                           num_neighbors=num_neighbors)
            if self.use_time == 'disentangle':
                src_node_t_embed = self.time_att_aggregate(src_node_conv_feat, src_node_t_embed)
            
            
            src_ngh_node_batch, src_ngh_eidx_batch, src_ngh_t_batch = self.ngh_finder.get_temporal_neighbor( 
                                                                    src_idx_l, 
                                                                    cut_time_l, 
                                                                    num_neighbors=num_neighbors)

            src_ngh_node_batch_th = torch.from_numpy(src_ngh_node_batch).long().to(device)
            
            src_ngh_t_batch_delta = cut_time_l[:, np.newaxis] - src_ngh_t_batch
            src_ngh_t_batch_th = torch.from_numpy(src_ngh_t_batch_delta).float().to(device)
            
            # get previous layer's node features
            src_ngh_node_batch_flat = src_ngh_node_batch.flatten() #reshape(batch_size, -1)
            src_ngh_t_batch_flat = src_ngh_t_batch.flatten() #reshape(batch_size, -1)  
            src_ngh_node_conv_feat = self.tem_conv(src_ngh_node_batch_flat, 
                                                   src_ngh_t_batch_flat,
                                                   curr_layers=curr_layers - 1, 
                                                   num_neighbors=num_neighbors)
            src_ngh_feat = src_ngh_node_conv_feat.view(batch_size, num_neighbors, -1)
            
            # get edge time features and node features
            src_ngh_t_embed = self.time_encoder(src_ngh_t_batch_th)
            if self.use_time == 'disentangle':
                src_ngh_t_embed = self.time_att_aggregate(src_ngh_feat, src_ngh_t_embed)

            # attention aggregation
            mask = src_ngh_node_batch_th == 0
            attn_m = self.attn_model_list[curr_layers - 1]
                        
            local, weight = attn_m(src_node_conv_feat, 
                                   src_node_t_embed,
                                   src_ngh_feat,
                                   src_ngh_t_embed, 
                                   mask)
            return local
