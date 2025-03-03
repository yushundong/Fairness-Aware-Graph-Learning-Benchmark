import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)

import torch
from torch import Tensor

from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax


import torch.optim as optim
import numpy as np
import time
from datetime import datetime
import argparse
import pickle

from sklearn.metrics import f1_score, roc_auc_score
from torch_geometric.utils import convert
from scipy.sparse.csgraph import laplacian
from torch_geometric.nn.inits import glorot, zeros

from torch_geometric.nn import GINConv, GCNConv, JumpingKnowledge
import numpy as np
import pandas as pd

import sklearn.preprocessing as skpp
from scipy.sparse.csgraph import laplacian
import networkx as nx
import scipy.sparse as sp
import networkx as nx
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score


def avg_err(x_corresponding, x_similarity, x_sorted_scores, y_ranks, top_k):
    the_maxs, _ = torch.max(x_corresponding, 1)
    the_maxs = the_maxs.reshape(the_maxs.shape[0], 1).repeat(1, x_corresponding.shape[1])
    c = 2 * torch.ones_like(x_corresponding)
    x_corresponding = ( c.pow(x_corresponding) - 1) / c.pow(the_maxs)
    the_ones = torch.ones_like(x_corresponding)
    new_x_corresponding = torch.cat((the_ones, 1 - x_corresponding), 1)

    for i in range(x_corresponding.shape[1] - 1):
        x_corresponding = torch.mul(x_corresponding, new_x_corresponding[:, -x_corresponding.shape[1] - 1 - i : -1 - i])
    the_range = torch.arange(0., x_corresponding.shape[1]).repeat(x_corresponding.shape[0], 1) + 1
    score_rank = (1 / the_range[:, 0:]) * x_corresponding[:, 0:]
    final = torch.mean(torch.sum(score_rank, axis=1))
    print("Now Average ERR@k = ", final.item())

    return final.item()


def sgc_precompute(features, adj, degree):
    t = perf_counter()
    for i in range(degree):
        features = torch.spmm(adj, features)
    precompute_time = perf_counter()-t
    return features, precompute_time

def simi(output):  # new_version

    a = output.norm(dim=1)[:, None]
    the_ones = torch.ones_like(a)
    a = torch.where(a==0, the_ones, a)
    a_norm = output / a
    b_norm = output / a

    res = 5 * (torch.mm(a_norm, b_norm.transpose(0, 1)) + 1)

    return res

def avg_ndcg(x_corresponding, x_similarity, x_sorted_scores, y_ranks, top_k):
    c = 2 * torch.ones_like(x_sorted_scores[:, :top_k])
    numerator = c.pow(x_sorted_scores[:, :top_k]) - 1
    denominator = torch.log2(2 + torch.arange(x_sorted_scores[:, :top_k].shape[1], dtype=torch.float)).repeat(x_sorted_scores.shape[0], 1).cuda()
    idcg = torch.sum((numerator / denominator), 1)
    new_score_rank = torch.zeros(y_ranks.shape[0], y_ranks[:, :top_k].shape[1])
    numerator = c.pow(x_corresponding.cuda()[:, :top_k]) - 1
    denominator = torch.log2(2 + torch.arange(new_score_rank[:, :top_k].shape[1], dtype=torch.float)).repeat(x_sorted_scores.shape[0], 1).cuda()
    ndcg_list = torch.sum((numerator / denominator), 1) / idcg
    avg_ndcg = torch.mean(ndcg_list)
    print("Now Average NDCG@k = ", avg_ndcg.item())

    return avg_ndcg.item()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# by rows
def idcg_computation(x_sorted_scores, top_k):
    c = 2 * torch.ones_like(x_sorted_scores)[:top_k]
    numerator = c.pow(x_sorted_scores[:top_k]) - 1
    denominator = torch.log2(2 + torch.arange(x_sorted_scores[:top_k].shape[0], dtype=torch.float)).cuda()
    final = numerator / denominator

    return torch.sum(final)

# by rows
def dcg_computation(score_rank, top_k):
    c = 2 * torch.ones_like(score_rank)[:top_k]
    numerator = c.pow(score_rank[:top_k]) - 1
    denominator = torch.log2(2 + torch.arange(score_rank[:top_k].shape[0], dtype=torch.float))
    final = numerator / denominator

    return torch.sum(final)


def ndcg_exchange_abs(x_corresponding, j, k, idcg, top_k):
    new_score_rank = x_corresponding
    dcg1 = dcg_computation(new_score_rank, top_k)
    the_index = np.arange(new_score_rank.shape[0])
    temp = the_index[j]
    the_index[j] = the_index[k]
    the_index[k] = temp
    new_score_rank = new_score_rank[the_index]
    dcg2 = dcg_computation(new_score_rank, top_k)

    return torch.abs((dcg1 - dcg2) / idcg)


def err_computation(score_rank, top_k):
    the_maxs = torch.max(score_rank).repeat(1, score_rank.shape[0])
    c = 2 * torch.ones_like(score_rank)
    score_rank = (( c.pow(score_rank) - 1) / c.pow(the_maxs))[0]
    the_ones = torch.ones_like(score_rank)
    new_score_rank = torch.cat((the_ones, 1 - score_rank))

    for i in range(score_rank.shape[0] - 1):
        score_rank = torch.mul(score_rank, new_score_rank[-score_rank.shape[0] - 1 - i : -1 - i])
    the_range = torch.arange(0., score_rank.shape[0]) + 1

    final = (1 / the_range[0:]) * score_rank[0:]

    return torch.sum(final)



def err_exchange_abs(x_corresponding, j, k, top_k):
    new_score_rank = x_corresponding
    err1 = err_computation(new_score_rank, top_k)
    the_index = np.arange(new_score_rank.shape[0])
    temp = the_index[j]
    the_index[j] = the_index[k]
    the_index[k] = temp
    new_score_rank = new_score_rank[the_index]
    err2 = err_computation(new_score_rank, top_k)

    return torch.abs(err1 - err2)




def lambdas_computation(x_similarity, y_similarity, top_k, k_para, sigma_1):
    max_num = 2000000
    x_similarity[range(x_similarity.shape[0]), range(x_similarity.shape[0])] = max_num * torch.ones_like(x_similarity[0, :])
    y_similarity[range(y_similarity.shape[0]), range(y_similarity.shape[0])] = max_num * torch.ones_like(y_similarity[0, :])

    # ***************************** ranking ******************************
    (x_sorted_scores, x_sorted_idxs) = x_similarity.sort(dim=1, descending=True)
    (y_sorted_scores, y_sorted_idxs) = y_similarity.sort(dim=1, descending=True)
    y_ranks = torch.zeros(y_similarity.shape[0], y_similarity.shape[0])
    the_row = torch.arange(y_similarity.shape[0]).view(y_similarity.shape[0], 1).repeat(1, y_similarity.shape[0])
    y_ranks[the_row, y_sorted_idxs] = 1 + torch.arange(y_similarity.shape[1]).repeat(y_similarity.shape[0], 1).float()

    # ***************************** pairwise delta ******************************
    sigma_tuned = sigma_1
    length_of_k = k_para * top_k
    y_sorted_scores = y_sorted_scores[:, 1 :(length_of_k + 1)]
    y_sorted_idxs = y_sorted_idxs[:, 1 :(length_of_k + 1)]
    x_sorted_scores = x_sorted_scores[:, 1 :(length_of_k + 1)]
    pairs_delta = torch.zeros(y_sorted_scores.shape[1], y_sorted_scores.shape[1], y_sorted_scores.shape[0])

    for i in range(y_sorted_scores.shape[0]):
        pairs_delta[:, :, i] = y_sorted_scores[i, :].view(y_sorted_scores.shape[1], 1) - y_sorted_scores[i, :].float()

    fraction_1 = - sigma_tuned / (1 + (pairs_delta * sigma_tuned).exp())
    x_delta = torch.zeros(y_sorted_scores.shape[1], y_sorted_scores.shape[1], y_sorted_scores.shape[0])
    x_corresponding = torch.zeros(x_similarity.shape[0], length_of_k)

    for i in range(x_corresponding.shape[0]):
        x_corresponding[i, :] = x_similarity[i, y_sorted_idxs[i, :]]

    for i in range(x_corresponding.shape[0]):
        # print(i / x_corresponding.shape[0])
        x_delta[:, :, i] = x_corresponding[i, :].view(x_corresponding.shape[1], 1) - x_corresponding[i, :].float()

    S_x = torch.sign(x_delta)
    zero = torch.zeros_like(S_x)
    S_x = torch.where(S_x < 0, zero, S_x)

    # ***************************** NDCG delta from ranking ******************************

    ndcg_delta = torch.zeros(x_corresponding.shape[1], x_corresponding.shape[1], x_corresponding.shape[0])
    for i in range(y_similarity.shape[0]):
        if i >= 0.6 * y_similarity.shape[0]:
            break
        idcg = idcg_computation(x_sorted_scores[i, :], top_k)
        for j in range(x_corresponding.shape[1]):
            for k in range(x_corresponding.shape[1]):
                if S_x[j, k, i] == 0:
                    continue
                if j < k:
                    the_delta = ndcg_exchange_abs(x_corresponding[i, :], j, k, idcg, top_k)
                    # print(the_delta)
                    ndcg_delta[j, k, i] = the_delta
                    ndcg_delta[k, j, i] = the_delta

    without_zero = S_x * fraction_1 * ndcg_delta
    lambdas = torch.zeros(x_corresponding.shape[0], x_corresponding.shape[1])
    for i in range(lambdas.shape[0]):
        for j in range(lambdas.shape[1]):
            lambdas[i, j] = torch.sum(without_zero[j, :, i]) - torch.sum(without_zero[:, j, i])   # 本来是 -

    mid = torch.zeros_like(x_similarity)
    the_x = torch.arange(x_similarity.shape[0]).repeat(length_of_k, 1).transpose(0, 1).reshape(length_of_k * x_similarity.shape[0], 1).squeeze()
    the_y = y_sorted_idxs.reshape(length_of_k * x_similarity.shape[0], 1).squeeze()
    the_data = lambdas.reshape(length_of_k * x_similarity.shape[0], 1).squeeze()
    mid.index_put_((the_x, the_y.long()), the_data.cuda())

    return mid, x_sorted_scores, y_sorted_idxs, x_corresponding


def lambdas_computation_only_review(x_similarity, y_similarity, top_k, k_para):
    max_num = 2000000
    x_similarity[range(x_similarity.shape[0]), range(x_similarity.shape[0])] = max_num * torch.ones_like(x_similarity[0, :])
    y_similarity[range(y_similarity.shape[0]), range(y_similarity.shape[0])] = max_num * torch.ones_like(y_similarity[0, :])

    # ***************************** ranking ******************************
    (x_sorted_scores, x_sorted_idxs) = x_similarity.sort(dim=1, descending=True)
    (y_sorted_scores, y_sorted_idxs) = y_similarity.sort(dim=1, descending=True)
    y_ranks = torch.zeros(y_similarity.shape[0], y_similarity.shape[0])
    the_row = torch.arange(y_similarity.shape[0]).view(y_similarity.shape[0], 1).repeat(1, y_similarity.shape[0])
    y_ranks[the_row, y_sorted_idxs] = 1 + torch.arange(y_similarity.shape[1]).repeat(y_similarity.shape[0], 1).float()
    length_of_k = k_para * top_k - 1
    y_sorted_idxs = y_sorted_idxs[:, 1 :(length_of_k + 1)]
    x_sorted_scores = x_sorted_scores[:, 1 :(length_of_k + 1)]
    x_corresponding = torch.zeros(x_similarity.shape[0], length_of_k)

    for i in range(x_corresponding.shape[0]):
        x_corresponding[i, :] = x_similarity[i, y_sorted_idxs[i, :]]

    return x_sorted_scores, y_sorted_idxs, x_corresponding
class JK(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout): 
        super(JK, self).__init__()
        self.body = JK_Body(nfeat, nhid, dropout)
        self.fc = nn.Linear(nhid, nclass)
        
        for m in self.modules():
            self.weights_init(m)


    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index): 
        x = self.body(x, edge_index)
        x = self.fc(x)
        return x

class JK_Body(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(JK_Body, self).__init__()
        self.conv1 = GCNConv(nfeat, nhid)
        self.convx= GCNConv(nhid, nhid)
        self.jk = JumpingKnowledge(mode='max')
        self.transition = nn.Sequential(
            nn.ReLU(),
        )

    def forward(self, x, edge_index):
        xs = []
        x = self.conv1(x, edge_index)
        x = self.transition(x)
        xs.append(x)
        for _ in range(1): 
            x = self.convx(x, edge_index)
            x = self.transition(x)
            xs.append(x)
        x = self.jk(xs)
        return x

class GIN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout): 
        super(GIN, self).__init__()


        self.body = GIN_Body(nfeat,nhid,dropout)
        self.fc = nn.Linear(nhid, nclass)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
        
    def forward(self, x, edge_index): 
        x = self.body(x, edge_index)
        x = self.fc(x)
        return x

class GIN_Body(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GIN_Body, self).__init__()
        
        self.mlp1 = nn.Sequential(
            nn.Linear(nfeat, nhid), 
            nn.ReLU(),
            nn.BatchNorm1d(nhid),
            nn.Linear(nhid, nhid), 
        )
        self.gc1 = GINConv(self.mlp1)

    def forward(self, x, edge_index):
        x = self.gc1(x, edge_index)
        return x

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.body = GCN_Body(nfeat,nhid,dropout)
        self.fc = nn.Linear(nhid, nclass)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        x = self.body(x, edge_index)
        x = self.fc(x)
        return x


class GCN_Body(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN_Body, self).__init__()
        self.gc1 = GCNConv(nfeat, nhid)

    def forward(self, x, edge_index):
        x = self.gc1(x, edge_index)
        return x    






class SimAttConv(MessagePassing):
    r"""
    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0,
                 add_self_loops: bool = False, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(SimAttConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=False)
            self.lin_r = self.lin_l
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels, False)
            self.lin_r = Linear(in_channels[1], heads * out_channels, False)

        self.att_l = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()


    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        glorot(self.att_l)
        glorot(self.att_r)
        zeros(self.bias)



    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_weight: OptTensor=None,
                size: Size = None, return_attention_weights=None):
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = x_r = self.lin_l(x).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            alpha_r = (x_r * self.att_r).sum(dim=-1)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = self.lin_l(x_l).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)
                alpha_r = (x_r * self.att_r).sum(dim=-1)

        assert x_l is not None
        assert alpha_l is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                if size is not None:
                    num_nodes = min(size[0], size[1])
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)

        out = self.propagate(edge_index, x=(x_l, x_r),
                             alpha=(alpha_l, alpha_r), edge_weight=edge_weight, size=size)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out


    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor, edge_weight: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = alpha * edge_weight.view(-1, 1) # multiply by the similarity
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j  * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)



def jaccard_similarity(mat):
    """
    get jaccard similarity matrix
    :param mat: scipy.sparse.csc_matrix
    :return: similarity matrix of nodes
    """
    # make it a binary matrix
    mat_bin = mat.copy()
    mat_bin.data[:] = 1

    col_sum = mat_bin.getnnz(axis=0)
    ab = mat_bin.dot(mat_bin.T)
    aa = np.repeat(col_sum, ab.getnnz(axis=0))
    bb = col_sum[ab.indices]
    sim = ab.copy()
    sim.data /= (aa + bb - ab.data)
    return sim


def cosine_similarity(mat):
    """
    get cosine similarity matrix
    :param mat: scipy.sparse.csc_matrix
    :return: similarity matrix of nodes
    """
    mat_row_norm = skpp.normalize(mat, axis=1)
    sim = mat_row_norm.dot(mat_row_norm.T)
    return sim


def get_similarity_matrix(mat, metric=None):
    """
    get similarity matrix of nodes in specified metric
    :param mat: scipy.sparse matrix (csc, csr or coo)
    :param metric: similarity metric
    :return: similarity matrix of nodes
    """
    if metric == 'jaccard':
        return jaccard_similarity(mat.tocsc())
    elif metric == 'cosine':
        return cosine_similarity(mat.tocsc())
    else:
        raise ValueError('Please specify the type of similarity metric.')



def filter_similarity_matrix(sim, sigma):
    """
    filter value by threshold = mean(sim) + sigma * std(sim)
    :param sim: similarity matrix
    :param sigma: hyperparameter for filtering values
    :return: filtered similarity matrix
    """
    sim_mean = np.mean(sim.data)
    sim_std = np.std(sim.data)
    threshold = sim_mean + sigma * sim_std
    sim.data *= sim.data >= threshold  # filter values by threshold
    sim.eliminate_zeros()
    return sim


def symmetric_normalize(mat):
    """
    symmetrically normalize a matrix
    :param mat: scipy.sparse matrix (csc, csr or coo)
    :return: symmetrically normalized matrix
    """
    degrees = np.asarray(mat.sum(axis=0).flatten())
    degrees = np.divide(1, degrees, out=np.zeros_like(degrees), where=degrees != 0)
    degrees = np.diags(np.asarray(degrees)[0, :])
    degrees.data = np.sqrt(degrees.data)
    return degrees @ mat @ degrees


def calculate_similarity_matrix(adj, features, metric=None, filterSigma = None, normalize = None, largestComponent=False):
    if metric in ['cosine', 'jaccard']:
        # build similarity matrix
        if largestComponent:
            graph = nx.from_scipy_sparse_matrix(adj)
            lcc = max(nx.connected_components(graph), key=len)  # take largest connected components
            adj = nx.to_scipy_sparse_matrix(graph, nodelist=lcc, dtype='float', format='csc')
        sim = get_similarity_matrix(adj, metric=metric)
        if filterSigma:
            sim = filter_similarity_matrix(sim, sigma=filterSigma)
        if normalize:
            sim = symmetric_normalize(sim)
    return sim



def calculate_group_lap(sim, sens):
    unique_sens = [int(x) for x in sens.unique(sorted=True).tolist()]
    num_unique_sens = sens.unique().shape[0]
    sens = [int(x) for x in sens.tolist()]
    m_list = [0]*num_unique_sens
    avgSimD_list = [[] for i in range(num_unique_sens)]
    sim_list = [sim.copy() for i in range(num_unique_sens)]

    for row, col in zip(*sim.nonzero()):
        sensRow = unique_sens[sens[row]]
        sensCol = unique_sens[sens[col]]
        if sensRow == sensCol:
            sim_list[sensRow][row,col] = 2*sim_list[sensRow][row,col]
            sim_to_zero_list = [x for x in unique_sens if x != sensRow]
            for sim_to_zero in sim_to_zero_list:
                sim_list[sim_to_zero][row,col] = 0
            m_list[sensRow] += 1
        else:
            m_list[sensRow] += 0.5
            m_list[sensRow] += 0.5

    lap = laplacian(sim)
    lap = lap.tocsr()
    for i in range(lap.shape[0]):
        sen_label = sens[i]
        avgSimD_list[sen_label].append(lap[i,i])
    avgSimD_list = [np.mean(l) for l in avgSimD_list]

    lap_list = [laplacian(sim) for sim in sim_list]

    return lap_list, m_list, avgSimD_list


def convert_sparse_matrix_to_sparse_tensor(X):
    X = X.tocoo()

    X = torch.sparse_coo_tensor(torch.tensor([X.row.tolist(), X.col.tolist()]),
                              torch.tensor(X.data.astype(np.float32)))
    return X


class GUIDE(nn.Module):
    def __init__(self,
                 num_layers=1,
                 nfeat=16,
                 nhid=16,
                 nclass=1,
                 heads=1,
                 negative_slope=0.2,
                 concat=False,
                 dropout=0):
        super(GUIDE, self).__init__()
        self.body = guideEncoder_body(num_layers, nfeat, nhid, heads, negative_slope, concat, dropout)
        self.fc = nn.Linear(nhid, nclass)
        self.activation = nn.ReLU()
        self.bn = nn.LayerNorm(nhid)
        self.num_class = nclass

    def forward(self, x, edge_index, edge_weight, return_attention_weights=None):
        if isinstance(return_attention_weights, bool) and return_attention_weights==True:
            logits, attention_weights = self.body(x, edge_index, edge_weight, return_attention_weights=return_attention_weights)
            logits = self.activation(self.bn(logits))
            logits = self.fc(logits)
            return logits, attention_weights
        else:
            logits = self.body(x, edge_index, edge_weight)
            logits = self.activation(self.bn(logits))
            logits = self.fc(logits)
            return logits



    def fit(self, dataset_name, adj, features, idx_train, idx_val, idx_test, labels, sens, hidden_num=16, dropout=0, lr=0.001 /2,
            weight_decay=1e-5, initialize_training_epochs=1000, epochs=1000, alpha=5e-6 *0.01, beta=1*0.001, gnn_name='gcn', device='cuda'):
        self.sens=sens
        self.idx_val=idx_val


        self.model = GCN(nfeat=features.shape[1],
                    nhid=hidden_num,
                    nclass=self.num_class,
                    dropout=dropout)
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)


        if gnn_name == 'gin':
            self.model = GIN(nfeat=features.shape[1],
                        nhid=hidden_num,
                        nclass=self.num_class,
                        dropout=dropout)
            optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)


        elif gnn_name == 'jk':
            self.model = JK(nfeat=features.shape[1],
                        nhid=hidden_num,
                        nclass=self.num_class,
                        dropout=dropout)
            optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)


        

        ifgOptimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        row=adj._indices()[0].cpu().numpy()
        col=adj._indices ()[1].cpu().numpy()
        data=adj._values().cpu().numpy()
        shape=adj.size()
        adj=sp.csr_matrix((data,(row, col)), shape=shape)

        print(f"Getting similarity matrix...")
        edge_index = convert.from_scipy_sparse_matrix(adj)[0]
        

        sim = calculate_similarity_matrix(adj, features, metric='cosine')
        sim_edge_index, sim_edge_weight = convert.from_scipy_sparse_matrix(sim)
        sim_edge_weight = sim_edge_weight.type(torch.FloatTensor)
        lap = laplacian(sim)
        print(f"Similarity matrix nonzero entries: {torch.count_nonzero(sim_edge_weight)}")


        try:
            with open("laplacians_{}".format(dataset_name) + '.pickle', 'rb') as f:
                loadLaplacians = pickle.load(f)
            lap_list, m_list, avgSimD_list = loadLaplacians['lap_list'], loadLaplacians['m_list'], loadLaplacians[
                'avgSimD_list']
            print("Laplacians loaded from previous runs")
        except:
            print("Calculating laplacians...(this may take a while)")
            lap_list, m_list, avgSimD_list = calculate_group_lap(sim, sens)
            saveLaplacians = {}
            saveLaplacians['lap_list'] = lap_list
            saveLaplacians['m_list'] = m_list
            saveLaplacians['avgSimD_list'] = avgSimD_list
            with open("laplacians_{}".format(dataset_name) + '.pickle', 'wb') as f:
                pickle.dump(saveLaplacians, f, protocol=pickle.HIGHEST_PROTOCOL)
            print("Laplacians calculated and stored.")


        #print("Get laplacians for IFG calculations...")
        #print("Calculating laplacians...(this may take a while for pokec_n)")
        #lap_list, m_list, avgSimD_list = calculate_group_lap(sim, sens)
        #saveLaplacians = {}
        #saveLaplacians['lap_list'] = lap_list
        #saveLaplacians['m_list'] = m_list
        #saveLaplacians['avgSimD_list'] = avgSimD_list

        #with open("laplacians" + '.pickle', 'wb') as f:
        #    pickle.dump(saveLaplacians, f, protocol=pickle.HIGHEST_PROTOCOL)
        #print("Laplacians calculated and stored.")

        #with open("laplacians-1" + '.pickle', 'rb') as f:
        #    loadLaplacians = pickle.load(f)
        #lap_list, m_list, avgSimD_list = loadLaplacians['lap_list'], loadLaplacians['m_list'], loadLaplacians['avgSimD_list']
        #print("Laplacians loaded from previous runs")

        

                
                
        lap = convert_sparse_matrix_to_sparse_tensor(lap)
        lap_list = [convert_sparse_matrix_to_sparse_tensor(X) for X in lap_list]
        lap_1 = lap_list[0]
        lap_2 = lap_list[1]
        m_u1 = m_list[0]
        m_u2 = m_list[1]




        best_total_loss_val = np.inf
        self = self.to(device)
        self.model = self.model.to(device)
        features = features.to(device)
        edge_index = edge_index.to(device)
        labels = labels.to(device)
        sim_edge_index = sim_edge_index.to(device)
        sim_edge_weight = sim_edge_weight.to(device)
        lap = lap.to(device)
        lap_1 = lap_1.to(device)
        lap_2 = lap_2.to(device)






        # preserve variables
        self.edge_index = edge_index
        self.features = features
        self.sim_edge_index = sim_edge_index
        self.sim_edge_weight = sim_edge_weight
        self.device = device
        self.labels = labels
        self.lap = lap
        self.lap_1 = lap_1
        self.lap_2 = lap_2
        self.m_u1 = m_u1
        self.m_u2 = m_u2

        self.idx_test = idx_test




        for epoch in range(initialize_training_epochs+1):
            t = time.time()
            self.model.train()
            optimizer.zero_grad()
            output = self.model(features, edge_index)

            loss_label_init_train = F.binary_cross_entropy_with_logits(output[idx_train], labels[idx_train].unsqueeze(1).float().to(device))

            auc_roc_init_train = roc_auc_score(labels.cpu().numpy()[idx_train.cpu().numpy()], output.detach().cpu().numpy()[idx_train.cpu().numpy()])
            individual_unfairness_vanilla = torch.trace(torch.mm(output.t(), torch.sparse.mm(lap,output))).item()
            f_u1 = torch.trace(torch.mm(output.t(), torch.sparse.mm(lap_1, output)))/m_u1
            f_u1 = f_u1.item()
            f_u2 = torch.trace(torch.mm(output.t(), torch.sparse.mm(lap_2, output)))/m_u2
            f_u2 = f_u2.item()
            GDIF_vanilla = max(f_u1/f_u2, f_u2/f_u1)

            loss_label_init_train.backward()
            optimizer.step()
            ################################Logging################################
            if epoch % 100 == 0:
                print(f"----------------------------")
                print(f"[Train] Epoch {epoch}: ")
                print(f"---Embedding Initialize---")
                print(f"Embedding Initialize: loss_label_train: {loss_label_init_train.item():.4f}, auc_roc_train: {auc_roc_init_train:.4f}, individual_unfairness_vanilla: {individual_unfairness_vanilla:.4f}, GDIF_vanilla {GDIF_vanilla:.4f}")



                
        print(f"--------------------Training GUIDE--------------------------")

        for epoch in range(epochs+1):
            t = time.time()
            ################################Training################################
            
            self.train()
            ifgOptimizer.zero_grad()
            with torch.no_grad():
                output = self.model.body(features, edge_index)
            ifgOutput = self.forward(output, sim_edge_index, sim_edge_weight)

            loss_label_guide_train = F.binary_cross_entropy_with_logits(ifgOutput[idx_train], labels[idx_train].unsqueeze(1).float().to(device))
            auc_roc_guide_train = roc_auc_score(labels.cpu().numpy()[idx_train.cpu().numpy()], ifgOutput.detach().cpu().numpy()[idx_train.cpu().numpy()])
            ifair_loss = torch.trace( torch.mm( ifgOutput.t(), torch.sparse.mm(lap, ifgOutput) ) )
            f_u1 = torch.trace(torch.mm(ifgOutput.t(), torch.sparse.mm(lap_1, ifgOutput)))/m_u1
            f_u2 = torch.trace(torch.mm(ifgOutput.t(), torch.sparse.mm(lap_2, ifgOutput)))/m_u2
            GDIF = max(f_u1/f_u2, f_u2/f_u1)
            ifg_loss = (f_u1/f_u2-1)**2 + (f_u2/f_u1-1)**2

            loss_guide_train = loss_label_guide_train + alpha * ifair_loss + beta * ifg_loss
            loss_guide_train.backward()
            ifgOptimizer.step()

            ################################Validation################################
            # Evaluate validation set performance separately
            self.eval()
            ifgOutput = self.forward(output, sim_edge_index, sim_edge_weight)


            # Get validation losses for guide encoder    
            # Label loss
            loss_label_guide_val = F.binary_cross_entropy_with_logits(ifgOutput[idx_val], labels[idx_val].unsqueeze(1).float().to(device))
            # Individual unfairness loss
            individual_unfairness = torch.trace(torch.mm(ifgOutput.t(), torch.sparse.mm(lap, ifgOutput))).item()
            # IF Group loss
            f_u1 = torch.trace(torch.mm(ifgOutput.t(), torch.sparse.mm(lap_1, ifgOutput)))/m_u1
            f_u1 = f_u1.item()
            f_u2 = torch.trace(torch.mm(ifgOutput.t(), torch.sparse.mm(lap_2, ifgOutput)))/m_u2
            f_u2 = f_u2.item()
            GDIF = max(f_u1/f_u2, f_u2/f_u1)
            ifg_loss = (f_u1/f_u2-1)**2 + (f_u2/f_u1-1)**2

            # Reporting metrics
            preds_guide = (ifgOutput.squeeze()>0).type_as(labels)
            auc_roc_guide_val = roc_auc_score(labels.cpu().numpy()[idx_val.cpu().numpy()], ifgOutput.detach().cpu().numpy()[idx_val.cpu().numpy()])
            if_reduction = (individual_unfairness_vanilla-individual_unfairness)/individual_unfairness_vanilla

            perf_val = auc_roc_guide_val + if_reduction + (GDIF_vanilla-GDIF)/(GDIF_vanilla-1)
            total_loss_val = loss_label_guide_val + alpha * individual_unfairness + beta * ifg_loss
            if (total_loss_val < best_total_loss_val) and (epoch > 500):
                best_total_loss_val = total_loss_val
            #     torch.save(ifgModel.state_dict(), guideEncoderWeightsName)
            ################################Logging################################
            if epoch % 100 == 0:
                print(f"----------------------------")
                print(f"[Train] Epoch {epoch}: ")
                #print(f"output {output}")
                #print(f"ifgOutput {ifgOutput}")
                print(f"---Training All objectives---")
                print(f"loss_label train {loss_label_guide_train.item():.4f}, auc_roc_train: {auc_roc_guide_train.item():.4f}")
                print(f"---Validation---")
                print(f"individual_unfairness_vanilla {individual_unfairness_vanilla:.4f}, GDIF_vanilla {GDIF_vanilla:.4f}")
                print(f"loss_total_val: {total_loss_val:.4f}, loss_label_val: {loss_label_guide_val.item():.4f}, loss_ifair: {alpha * individual_unfairness:.4f}, loss_ifg: {beta*ifg_loss:.4f}, auc_roc_val: {auc_roc_guide_val:.4f}, Individual Fairness: {individual_unfairness:.4f}, if_reduction: {'{:.2%}'.format(if_reduction)}, GDIF: {GDIF:.4f}, Perf_val: {perf_val:.4f}")


    def predict(self):

        self.model.eval()
        output = self.model.body(self.features, self.edge_index)

        self.eval()
        output, attention_weights = self.forward(output, self.sim_edge_index.to(self.device), self.sim_edge_weight.to(self.device), return_attention_weights=True )
        print(output.shape)

        #attention_weights = torch.sparse_coo_tensor(attention_weights[0], attention_weights[1])
        #attention_weights = attention_weights.detach()
        #attention_weights = attention_weights.coalesce()


        # Report
        output_preds = (output.squeeze()>0).type_as(self.labels)


        #F1 = f1_score(self.labels.cpu().numpy()[self.idx_test.cpu().numpy()], output_preds.detach().cpu().numpy()[self.idx_test.cpu().numpy()], average='micro')
        #ACC=accuracy_score(self.labels.detach().cpu().numpy()[self.idx_test.cpu().numpy()], output_preds.detach().cpu().numpy()[self.idx_test.cpu().numpy()],)
        #AUCROC=roc_auc_score(self.labels.cpu().numpy()[self.idx_test.cpu().numpy()], output_preds.detach().cpu().numpy()[self.idx_test.cpu().numpy()])

        output=output
        IF = torch.trace(torch.mm(output.t(), torch.sparse.mm(self.lap.cuda(), output))).item()
        f_u1 = torch.trace(torch.mm(output.t(), torch.sparse.mm(self.lap_1, output))) / self.m_u1
        f_u1 = f_u1.item()
        f_u2 = torch.trace(torch.mm(output.t(), torch.sparse.mm(self.lap_2, output))) / self.m_u2
        f_u2 = f_u2.item()
        if_group_pct_diff = np.abs(f_u1 - f_u2) / min(f_u1, f_u2)
        GDIF = max(f_u2 / (f_u1 + 1e-9), f_u1 / (f_u2 + 1e-9))


        x_inverse=1-output[self.idx_test].sigmoid()
        x_inverse=torch.log(x_inverse/(1-x_inverse))
        y_similarity = simi( torch.concat([output[self.idx_test], x_inverse],-1))
        x_similarity = simi(self.features[self.idx_test])
        x_sorted_scores, y_sorted_idxs, x_corresponding = lambdas_computation_only_review(x_similarity, y_similarity,
                                                                                          top_k=10, k_para=1)
        ndcg_value = avg_ndcg(x_corresponding, x_similarity, x_sorted_scores, y_sorted_idxs, top_k=10)
        print('ndcg', ndcg_value)
        print("\n")

        # print report
        print(f'IF: {IF}')
        # print(f'Individual Unfairness for Group 1: {f_u1}')
        # print(f'Individual Unfairness for Group 2: {f_u2}')
        print(f'GDIF: {GDIF}')

        idx_test = self.idx_test.cpu().numpy()
        self.labels = self.labels.cpu().numpy()

        pred = output_preds[idx_test].cpu().numpy()
        F1 = f1_score(self.labels[idx_test], pred, average='micro')
        ACC = accuracy_score(self.labels[idx_test], pred, )

        if self.labels.max() > 1:
            AUCROC = 0
        else:
            AUCROC = roc_auc_score(self.labels[idx_test], pred)

        ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1, F1_sens1 = self.predict_sens_group(pred, idx_test)

        SP, EO = self.fair_metric(np.array(pred), self.labels[idx_test], self.sens[idx_test].cpu().numpy())

        pred = output[self.idx_val].detach().cpu().numpy()
        loss_fn = torch.nn.BCELoss()
        self.val_loss = loss_fn(torch.FloatTensor(pred).sigmoid().squeeze(),
                                torch.tensor(self.labels[self.idx_val.detach().cpu().numpy()]).squeeze().float()).item()


        return ACC, AUCROC, F1, ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1, F1_sens1, SP, EO, IF, GDIF, ndcg_value


    def predict_val(self):

        self.model.eval()
        output = self.model.body(self.features, self.edge_index)

        self.eval()
        output, attention_weights = self.forward(output, self.sim_edge_index.to(self.device), self.sim_edge_weight.to(self.device), return_attention_weights=True )
        print(output.shape)

        # Report
        output_preds = (output.squeeze()>0).type_as(torch.tensor(self.labels))

        output=output
        IF = torch.trace(torch.mm(output.t(), torch.sparse.mm(self.lap.cuda(), output))).item()
        f_u1 = torch.trace(torch.mm(output.t(), torch.sparse.mm(self.lap_1, output))) / self.m_u1
        f_u1 = f_u1.item()
        f_u2 = torch.trace(torch.mm(output.t(), torch.sparse.mm(self.lap_2, output))) / self.m_u2
        f_u2 = f_u2.item()
        if_group_pct_diff = np.abs(f_u1 - f_u2) / min(f_u1, f_u2)
        GDIF = max(f_u2 / (f_u1 + 1e-9), f_u1 / (f_u2 + 1e-9))


        x_inverse=1-output[self.idx_val].sigmoid()
        x_inverse=torch.log(x_inverse/(1-x_inverse))
        y_similarity = simi( torch.concat([output[self.idx_val], x_inverse],-1))
        x_similarity = simi(self.features[self.idx_val])
        x_sorted_scores, y_sorted_idxs, x_corresponding = lambdas_computation_only_review(x_similarity, y_similarity,
                                                                                          top_k=10, k_para=1)
        ndcg_value = avg_ndcg(x_corresponding, x_similarity, x_sorted_scores, y_sorted_idxs, top_k=10)
        print('ndcg', ndcg_value)
        print("\n")

        # print report
        print(f'IF: {IF}')
        # print(f'Individual Unfairness for Group 1: {f_u1}')
        # print(f'Individual Unfairness for Group 2: {f_u2}')
        print(f'GDIF: {GDIF}')

        idx_test = self.idx_val.cpu().numpy()


        pred = output_preds[idx_test].cpu().numpy()
        F1 = f1_score(self.labels[idx_test], pred, average='micro')
        ACC = accuracy_score(self.labels[idx_test], pred, )

        if self.labels.max() > 1:
            AUCROC = 0
        else:
            AUCROC = roc_auc_score(self.labels[idx_test], pred)

        ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1, F1_sens1 = self.predict_sens_group(pred, idx_test)

        SP, EO = self.fair_metric(np.array(pred), self.labels[idx_test], self.sens[idx_test].cpu().numpy())

        pred = output[self.idx_val].detach().cpu().numpy()
        loss_fn = torch.nn.BCELoss()
        self.val_loss = loss_fn(torch.FloatTensor(pred).sigmoid().squeeze(),
                                torch.tensor(self.labels[self.idx_val.detach().cpu().numpy()]).squeeze().float()).item()


        return ACC, AUCROC, F1, ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1, F1_sens1, SP, EO, IF, GDIF, ndcg_value



    def fair_metric(self, pred, labels, sens):
        idx_s0 = sens == 0
        idx_s1 = sens == 1
        idx_s0_y1 = np.bitwise_and(idx_s0, labels == 1)
        idx_s1_y1 = np.bitwise_and(idx_s1, labels == 1)
        parity = abs(sum(pred[idx_s0]) / sum(idx_s0) -
                     sum(pred[idx_s1]) / sum(idx_s1))

        equality = abs(sum(pred[idx_s0_y1]) / sum(idx_s0_y1) -
                       sum(pred[idx_s1_y1]) / sum(idx_s1_y1))

        return parity.item(), equality.item()

    def predict_sens_group(self, pred, idx_test):

        result = []
        for sens in [0, 1]:
            F1 = f1_score(self.labels[idx_test][self.sens[idx_test] == sens], pred[self.sens[idx_test] == sens],
                          average='micro')
            ACC = accuracy_score(self.labels[idx_test][self.sens[idx_test] == sens],
                                 pred[self.sens[idx_test] == sens], )
            if self.labels.max() > 1:
                AUCROC = 0
            else:
                AUCROC = roc_auc_score(self.labels[idx_test][self.sens[idx_test] == sens],
                                       pred[self.sens[idx_test] == sens])
            result.extend([ACC, AUCROC, F1])

        return result


class guideEncoder_body(nn.Module):
    def __init__(self,
                 num_layers,
                 nfeat,
                 nhid,
                 heads,
                 negative_slope,
                 concat,
                 dropout):
        super(guideEncoder_body, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        
        self.layers.append(SimAttConv(
            nfeat, nhid, heads, self.concat, self.negative_slope, self.dropout))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the nfeat = nhid * num_heads
            if self.concat == True:
                self.layers.append(SimAttConv(
                nhid * heads, nhid, heads, self.concat, self.negative_slope))
            else:
                self.layers.append(SimAttConv(
                nhid, nhid, heads, self.concat, self.negative_slope))

        
    def forward(self, x, edge_index, edge_weight, return_attention_weights=None):
        h = x
        for l in range(self.num_layers-1):
            h = self.layers[l](h, edge_index, edge_weight).flatten(1)

        if isinstance(return_attention_weights, bool) and return_attention_weights==True:
            logits, attention_weights = self.layers[-1](h, edge_index, edge_weight, return_attention_weights=return_attention_weights)
            return logits, attention_weights
        else:
            logits = self.layers[-1](h, edge_index, edge_weight)
            return logits



