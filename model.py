import torch

from torch_geometric.nn import PNAConv
from torch_geometric.nn import global_mean_pool, BatchNorm

from ogb.graphproppred.mol_encoder import AtomEncoder
import torch.nn.functional as F


class PNA_Net(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, deg):
        super(PNA_Net, self).__init__()

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']
        
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for _ in self.num_layers:
            self.layers.append(PNAConv(in_channels=hidden_channels, out_channels=hidden_channels,aggregators=aggregators,scalers=scalers,deg=deg, post_layers=1))
            self.batch_norms.append(BatchNorm(hidden_channels))

        self.mlp = torch.nn.Linear(80, 1)
        self.embedding_h = AtomEncoder(emb_dim=hidden_channels)

    def forward(self,x,edge_index, batch):
        x = self.embedding_h(x)
        
        for i in range(len(self.layers)):
            x_h = x
            x = self.layers[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = x_h + x
            x = F.dropout(x, 0.3, training=self.training)
            
        x = global_mean_pool(x, batch)
        x = self.mlp(x)

        return x