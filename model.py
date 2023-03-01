import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import aggr, MLP


class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4, non_linear=True, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(dim1 + dim2, dim3)
        self.fc2 = nn.Linear(dim3, dim4)
        self.act = nn.ReLU()
        self.dropout = dropout

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

        self.non_linear = non_linear
        if not non_linear:
            assert (dim1 == dim2)
            self.fc = nn.Linear(dim1, 1)
            nn.init.xavier_normal_(self.fc1.weight)

    def forward(self, x1, x2):
        z_walk = None
        if self.non_linear:
            x = torch.cat([x1, x2], dim=-1)
            h = self.act(self.fc1(x))
            h = F.dropout(h, p=self.dropout, training=self.training)
            z = self.fc2(h)
        else:
            # x1, x2 shape: [B, M, F]
            x = torch.cat([x1, x2], dim=-2)  # x shape: [B, 2M, F]
            z_walk = self.fc(x).squeeze(-1)  # z_walk shape: [B, 2M]
            z = z_walk.sum(dim=-1, keepdim=True)  # z shape [B, 1]
        return z, z_walk

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)


class Net(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, out_dim, x_dim=0, dropout=0.5, use_feature=False,
                 aggrs='attn'):
        super(Net, self).__init__()
        self.use_feature = use_feature
        self.dropout = dropout
        self.x_dim = x_dim
        self.aggrfunc = aggrs

        self.pe_embedding = nn.Sequential(nn.Linear(in_features=input_dim, out_features=hidden_dim),
                                          nn.ReLU(), nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
        self.feature_embedding = nn.Sequential(nn.Linear(in_features=x_dim, out_features=hidden_dim),
                                               nn.ReLU(), nn.Linear(in_features=hidden_dim, out_features=hidden_dim))

        if aggrs == 'attn':
            gate_nn = MLP([hidden_dim, 1], act='relu')
            fnn = MLP([hidden_dim, hidden_dim], act='relu')
            self.aggr = aggr.AttentionalAggregation(gate_nn, fnn)
        elif aggrs == 'lstm':
            self.aggr = aggr.LSTMAggregation(
                in_channels=hidden_dim, out_channels=hidden_dim)
        else:
            self.aggr = aggr.MeanAggregation()

        if use_feature:
            self.affinity_score = MergeLayer(2 * hidden_dim, 2 * hidden_dim, hidden_dim, out_dim, non_linear=True,
                                             dropout=dropout)
        else:
            self.affinity_score = MergeLayer(hidden_dim, hidden_dim, hidden_dim, out_dim, non_linear=True,
                                             dropout=dropout)

    def forward(self, x, ptr, feature=None, debug=None):
        # out shape [2 (u,v), batch*num_walk, 2 (l,r), pos_dim]
        x = self.pe_embedding(x).sum(dim=-2)

        if self.aggrfunc != 'lstm':
            xl, xr = self.aggr(x, ptr=ptr).view(2, -1, x.shape[-1])
        else:
            xl, xr = self.aggr(x, index=ptr).view(2, -1, x.shape[-1])

        if self.use_feature:
            f_i, f_j = self.feature_embedding(feature)
            xl, xr = torch.cat([xl, f_i], dim=-1), torch.cat([xr, f_j], dim=-1)

        score, _ = self.affinity_score(xl, xr)
        return score.squeeze(1)

    def reset_parameters(self):
        for layer in self.pe_embedding:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
                nn.init.xavier_normal_(layer.weight)
        if self.use_feature:
            for layer in self.feature_embedding:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
                    nn.init.xavier_normal_(layer.weight)
        if self.aggrfunc != 'mean':
            self.aggr.reset_parameters()
        self.affinity_score.reset_parameters()
