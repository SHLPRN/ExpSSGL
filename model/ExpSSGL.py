from base.graph_recommender import GraphRecommender
from base.torch_interface import TorchGraphInterface
from data.augmentor import GraphAugmentor
from util.conf import OptionConf
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
from util.sampler import next_batch_pairwise, exp_next_batch_pairwise
import math
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpSSGL(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(ExpSSGL, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['ExpSSGL'])
        """
        # structure A/C
        self.cl_rate1 = float(args['-lambda1'])
        self.cl_rate2 = float(args['-lambda2'])
        # structure B
        self.cl_rate = float(args['-lambda'])
        """
        # structure D/E
        self.cl_rate = float(args['-lambda1'])
        self.gl_rate = float(args['-lambda2'])
        self.drop_rate = float(args['-droprate'])
        self.keep_rate = float(args['-keeprate'])
        self.eps = float(args['-eps'])
        self.temp = float(args['-tau'])
        self.n_layers = int(args['-n_layer'])
        # residue_edge: edges to be dropout
        # keep_edge: edges kept by keep_rate
        self.keep_edge_u_idx, self.keep_edge_i_idx, self.residue_edge = self.low_degree_node_keeper()
        self.model = ExpSSGL_Encoder(self.data, self.emb_size, self.n_layers, self.eps)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            """
            # structure A/B/C
            dropped_adj = self.graph_edge_dropout()
            """
            # structure D/E
            dropped_adj, dropped_interaction_mat = self.graph_edge_dropout(need=True)
            """
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
            """
            for n, batch in enumerate(exp_next_batch_pairwise(self.data, self.batch_size, dropped_interaction_mat)):
                """
                # structure A/B/C
                user_idx, pos_idx, neg_idx = batch
                """
                # structure D/E
                user_idx, pos_idx, neg_idx, drop_user_idx, drop_pos_idx, drop_neg_idx = batch
                rec_user_emb, rec_item_emb = model()
                user_emb, pos_item_emb, neg_item_emb = (rec_user_emb[user_idx], rec_item_emb[pos_idx],
                                                        rec_item_emb[neg_idx])
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                """
                # structure A
                cl_loss = (self.cl_rate1 * self.cal_cl_loss1([user_idx, pos_idx]) +
                           self.cl_rate2 * self.cal_cl_loss2([user_idx, pos_idx], rec_user_emb, rec_item_emb,
                                                             dropped_adj))
                # structure B
                cl_loss = self.cl_rate * self.cal_cl_loss3([user_idx, pos_idx], dropped_adj)
                # structure C
                cl_loss = (self.cl_rate1 * self.cal_cl_loss1([user_idx, pos_idx]) +
                           self.cl_rate2 * self.cal_cl_loss4([user_idx, pos_idx]))
                # structure D
                cl_loss = self.cl_rate * self.cal_cl_loss1([user_idx, pos_idx])
                # structure D_1
                gl_loss = self.gl_rate * self.cal_gl_loss1(dropped_adj, drop_user_idx, drop_pos_idx, drop_neg_idx)
                # structure D_2
                gl_loss = self.gl_rate * self.cal_gl_loss2(rec_user_emb, rec_item_emb, dropped_adj, drop_user_idx,
                                                           drop_pos_idx, drop_neg_idx)
                ssl_loss = cl_loss + gl_loss
                """
                # structure E
                cl_loss, gl_loss = self.cal_ssl_loss(dropped_adj, drop_user_idx, drop_pos_idx, drop_neg_idx)
                ssl_loss = cl_loss + gl_loss
                """
                # structure A/B/C
                batch_loss = rec_loss + cl_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb)
                """
                # structure D/E
                batch_loss = rec_loss + ssl_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb)
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100 == 0 and n > 0:
                    """
                    # structure A/B/C
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss:', cl_loss.item())
                    """
                    # structure D/E
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'ssl_loss:', ssl_loss.item())
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            self.fast_evaluation(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def low_degree_node_keeper(self):
        u_cnt = self.data.interaction_mat.get_shape()[0]
        i_cnt = self.data.interaction_mat.get_shape()[1]
        val = np.ones(u_cnt, dtype=np.float32)
        row_idx = np.zeros(u_cnt, dtype=np.float32)
        i_degree = (sp.csr_matrix((val, (row_idx, np.arange(u_cnt))), shape=(1, u_cnt)) *
                    self.data.interaction_mat).toarray().reshape(i_cnt)
        low_degree_i = np.argsort(i_degree)[:math.floor(i_cnt * self.keep_rate)]
        val = np.ones(i_cnt, dtype=np.float32)
        col_idx = np.zeros(i_cnt, dtype=np.float32)
        u_degree = (self.data.interaction_mat * sp.csr_matrix((val, (np.arange(i_cnt), col_idx)),
                                                              shape=(i_cnt, 1))).toarray().reshape(u_cnt)
        low_degree_u = np.argsort(u_degree)[:math.floor(u_cnt * self.keep_rate)]
        keep_edge_u_idx = []
        keep_edge_i_idx = []
        residue_edge = np.arange(self.data.interaction_mat.count_nonzero())
        row_idx, col_idx = self.data.interaction_mat.nonzero()
        for i in range(row_idx.size):
            mid_row_idx = row_idx[i]
            mid_col_idx = col_idx[i]
            if mid_row_idx in low_degree_u or mid_col_idx in low_degree_i:
                keep_edge_u_idx.append(mid_row_idx)
                keep_edge_i_idx.append(mid_col_idx)
                np.delete(residue_edge, i)
        return keep_edge_u_idx, keep_edge_i_idx, residue_edge

    def graph_edge_dropout(self, need=False):
        dropped_mat = GraphAugmentor.exp_edge_dropout(self.data.interaction_mat, self.drop_rate,
                                                      self.keep_edge_u_idx, self.keep_edge_i_idx, self.residue_edge)
        dropped_interaction_mat = None
        if need:
            dropped_interaction_mat = TorchGraphInterface.convert_sparse_mat_to_tensor(dropped_mat).cuda()
        dropped_mat = self.data.convert_to_laplacian_mat(dropped_mat)
        if need:
            return TorchGraphInterface.convert_sparse_mat_to_tensor(dropped_mat).cuda(), dropped_interaction_mat
        else:
            return TorchGraphInterface.convert_sparse_mat_to_tensor(dropped_mat).cuda()

    def cal_cl_loss1(self, idx):
        """CL: noise & noise"""
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_view_1, item_view_1 = self.model(perturbed=True)
        user_view_2, item_view_2 = self.model(perturbed=True)
        user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], self.temp)
        item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], self.temp)
        return user_cl_loss + item_cl_loss

    def cal_cl_loss2(self, idx, user_view_2, item_view_2, perturbed_mat):
        """CL: dropout & main"""
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_view_1, item_view_1 = self.model(perturbed_adj=perturbed_mat)
        view1 = torch.cat((user_view_1[u_idx], item_view_1[i_idx]), 0)
        view2 = torch.cat((user_view_2[u_idx], item_view_2[i_idx]), 0)
        return InfoNCE(view1, view2, self.temp)

    def cal_cl_loss3(self, idx, perturbed_mat):
        """CL: dropout & noise"""
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_view_1, item_view_1 = self.model(perturbed_adj=perturbed_mat)
        user_view_2, item_view_2 = self.model(perturbed=True)
        user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], self.temp)
        item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], self.temp)
        return user_cl_loss + item_cl_loss

    def cal_cl_loss4(self, idx):
        # CL: dropout & dropout
        perturbed_mat1 = self.graph_edge_dropout()
        perturbed_mat2 = self.graph_edge_dropout()
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_view_1, item_view_1 = self.model(perturbed_adj=perturbed_mat1)
        user_view_2, item_view_2 = self.model(perturbed_adj=perturbed_mat2)
        user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], self.temp)
        item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], self.temp)
        return user_cl_loss + item_cl_loss

    def cal_gl_loss1(self, perturbed_mat, drop_user_idx, drop_pos_idx, drop_neg_idx):
        """GL: base on the raw embeddings"""
        perturbed_user_emb, perturbed_item_emb = self.model(perturbed_adj=perturbed_mat)
        user_emb, pos_item_emb, neg_item_emb = (perturbed_user_emb[drop_user_idx], perturbed_item_emb[drop_pos_idx],
                                                perturbed_item_emb[drop_neg_idx])
        return bpr_loss(user_emb, pos_item_emb, neg_item_emb)

    def cal_gl_loss2(self, rec_user_emb, rec_item_emb, perturbed_mat, drop_user_idx, drop_pos_idx, drop_neg_idx):
        """GL: base on the embeddings after GCN process"""
        perturbed_user_emb, perturbed_item_emb = self.model(rec_user_emb=rec_user_emb, rec_item_emb=rec_item_emb,
                                                            perturbed_adj=perturbed_mat)
        user_emb, pos_item_emb, neg_item_emb = (perturbed_user_emb[drop_user_idx], perturbed_item_emb[drop_pos_idx],
                                                perturbed_item_emb[drop_neg_idx])
        return bpr_loss(user_emb, pos_item_emb, neg_item_emb)

    def cal_ssl_loss(self, perturbed_mat, drop_user_idx, drop_pos_idx, drop_neg_idx):
        """SSL: CL-dropout&noise + GL-base on the raw embeddings"""
        perturbed_user_emb, perturbed_item_emb = self.model(perturbed_adj=perturbed_mat)
        user_emb, pos_item_emb, neg_item_emb = (perturbed_user_emb[drop_user_idx], perturbed_item_emb[drop_pos_idx],
                                                perturbed_item_emb[drop_neg_idx])
        gl_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_view_2, item_view_2 = self.model(perturbed=True)
        user_cl_loss = InfoNCE(perturbed_user_emb[u_idx], user_view_2[u_idx], self.temp)
        item_cl_loss = InfoNCE(perturbed_item_emb[i_idx], item_view_2[i_idx], self.temp)
        cl_loss = user_cl_loss + item_cl_loss
        return cl_loss, gl_loss

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class ExpSSGL_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers, eps):
        super(ExpSSGL_Encoder, self).__init__()
        self.data = data
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.eps = eps
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict

    def forward(self, rec_user_emb=None, rec_item_emb=None, perturbed_adj=None, perturbed=False):
        if rec_user_emb is None:
            ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        else:
            ego_embeddings = torch.cat([rec_user_emb, rec_item_emb], 0)
        all_embeddings = []
        for k in range(self.n_layers):
            if perturbed_adj is not None:
                ego_embeddings = torch.sparse.mm(perturbed_adj, ego_embeddings)
            else:
                ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
                if perturbed:
                    random_noise = torch.rand_like(ego_embeddings).cuda()
                    ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = (
            torch.split(all_embeddings, [self.data.user_num, self.data.item_num]))
        return user_all_embeddings, item_all_embeddings
