import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_scatter import scatter_mean


class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, non_linear=True, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(dim1 * 4, dim2)
        self.fc2 = nn.Linear(dim2, dim3)
        self.act = nn.ReLU()
        self.dropout = dropout

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

        self.non_linear = non_linear
        if not non_linear:
            assert (dim1 == dim2 == dim3)
            self.fc = nn.Linear(dim1, 1)
            nn.init.xavier_normal_(self.fc1.weight)

    def forward(self, x1, x2, x3, x4):
        z_walk = None
        if self.non_linear:
            x = torch.cat([x1, x2, x3, x4], dim=-1)
            h = self.act(self.fc1(x))
            h = F.dropout(h, p=self.dropout, training=self.training)
            z = self.fc2(h)
        else:
            x = torch.cat([x1, x2, x3, x4], dim=-2)  # x shape: [B, 2M, F]
            z_walk = self.fc(x).squeeze(-1)  # z_walk shape: [B, 2M]
            z = z_walk.sum(dim=-1, keepdim=True)  # z shape [B, 1]
        return z, z_walk

    def reset_parameter(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()


class HONet(torch.nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, out_dim, x_dim=0, dropout=0.5):
        super(HONet, self).__init__()
        self.dropout = dropout
        self.x_dim = x_dim
        self.enc = 'LP'

        self.pe_embedding = nn.Sequential(nn.Linear(in_features=input_dim, out_features=hidden_dim),
                                          nn.ReLU(), nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
        self.affinity_score = MergeLayer(
            hidden_dim, hidden_dim, out_dim, non_linear=True, dropout=dropout)
        self.concat_norm = nn.LayerNorm(hidden_dim * 2)

    def forward(self, x, ind, feature=None, debug=None):
        x = self.pe_embedding(x).sum(dim=-2)
        xu, xwu, xv, xwv = scatter_mean(x, ind, dim=0).view(4, -1, x.shape[-1])
        score, _ = self.affinity_score(xu, xwu, xv, xwv)
        return score.squeeze(1)

    def reset_parameters(self):
        for layer in self.pe_embedding:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
                nn.init.xavier_normal_(layer.weight)
        self.affinity_score.reset_parameter()
