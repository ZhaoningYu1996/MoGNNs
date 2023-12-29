import torch
import torch.nn as nn
from torch_geometric.nn import PNAConv, GCNConv, GraphConv, GINConv, GINEConv, GATConv, GATv2Conv, TransformerConv, ARMAConv, TAGConv, SAGEConv
from torch_geometric.nn import global_mean_pool, BatchNorm

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
import torch.nn.functional as F

class PNA_Net(torch.nn.Module):
    def __init__(self, hid_channels, out_channels, num_layers, deg):
        super(PNA_Net, self).__init__()

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']
        
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(PNAConv(in_channels=hid_channels, out_channels=hid_channels,aggregators=aggregators,scalers=scalers,deg=deg, post_layers=1))
            self.batch_norms.append(BatchNorm(hid_channels))

        self.mlp = torch.nn.Linear(hid_channels, out_channels)
        self.embedding_h = AtomEncoder(emb_dim=hid_channels)

    def forward(self, x, edge_index, batch):
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
    
class GCN_Net(torch.nn.Module):
    def __init__(self, hid_channels, out_channels, num_layers, dropout):
        super(GCN_Net, self).__init__()
        
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.dropout = dropout
        for _ in range(self.num_layers):
            self.layers.append(GCNConv(in_channels=hid_channels, out_channels=hid_channels, add_self_loops=True))
            self.batch_norms.append(BatchNorm(hid_channels))
        
        self.mlp = torch.nn.Linear(hid_channels, out_channels)
        self.embedding_h = AtomEncoder(emb_dim=hid_channels)
        
    def forward(self, x, edge_index, batch):
        x = self.embedding_h(x)
        
        for i in range(len(self.layers)):
            x_h = x
            x = self.layers[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = x_h + x
            x = F.dropout(x, self.dropout, training=self.training)
            
        x = global_mean_pool(x, batch)
        x = self.mlp(x)

        return x
    
class GCNv2_Net(torch.nn.Module):
    def __init__(self, hid_channels, out_channels, num_layers, dropout):
        super(GCNv2_Net, self).__init__()
        
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.dropout = dropout
        for _ in range(self.num_layers):
            self.layers.append(GraphConv(in_channels=hid_channels, out_channels=hid_channels, add_self_loops=True))
            self.batch_norms.append(BatchNorm(hid_channels))
        
        self.mlp = torch.nn.Linear(hid_channels, out_channels)
        self.embedding_h = AtomEncoder(emb_dim=hid_channels)
        
    def forward(self, x, edge_index, batch):
        x = self.embedding_h(x)
        
        for i in range(len(self.layers)):
            x_h = x
            x = self.layers[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = x_h + x
            x = F.dropout(x, self.dropout, training=self.training)
            
        x = global_mean_pool(x, batch)
        x = self.mlp(x)

        return x
    
class SAGE_Net(torch.nn.Module):
    def __init__(self, hid_channels, out_channels, num_layers, dropout):
        super(SAGE_Net, self).__init__()
        
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.dropout = dropout
        for _ in range(self.num_layers):
            self.layers.append(SAGEConv(in_channels=hid_channels, out_channels=hid_channels, add_self_loops=True))
            self.batch_norms.append(BatchNorm(hid_channels))
        
        self.mlp = torch.nn.Linear(hid_channels, out_channels)
        self.embedding_h = AtomEncoder(emb_dim=hid_channels)
        
    def forward(self, x, edge_index, batch):
        x = self.embedding_h(x)
        
        for i in range(len(self.layers)):
            x_h = x
            x = self.layers[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = x_h + x
            x = F.dropout(x, self.dropout, training=self.training)
            
        x = global_mean_pool(x, batch)
        x = self.mlp(x)

        return x
    
class GIN_Net(torch.nn.Module):
    def __init__(self, hid_channels, out_channels, num_layers, dropout):
        super(GIN_Net, self).__init__()
        
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.dropout = dropout
        for _ in range(self.num_layers):
            self.layers.append(GINConv(nn.Sequential(nn.Linear(hid_channels, hid_channels), nn.ReLU(), nn.Linear(hid_channels, hid_channels))))
            self.batch_norms.append(BatchNorm(hid_channels))
        
        self.mlp = torch.nn.Linear(hid_channels, out_channels)
        self.embedding_h = AtomEncoder(emb_dim=hid_channels)
        
    def forward(self, x, edge_index, batch):
        x = self.embedding_h(x)
        
        for i in range(len(self.layers)):
            x_h = x
            x = self.layers[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = x_h + x
            x = F.dropout(x, self.dropout, training=self.training)
            
        x = global_mean_pool(x, batch)
        x = self.mlp(x)

        return x
    
class GINE_Net(torch.nn.Module):
    def __init__(self, hid_channels, out_channels, num_layers, dropout):
        super(GINE_Net, self).__init__()
        
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.dropout = dropout
        for _ in range(self.num_layers):
            self.layers.append(GINEConv(nn.Sequential(nn.Linear(hid_channels, hid_channels), nn.ReLU(), nn.Linear(hid_channels, hid_channels))))
            self.batch_norms.append(BatchNorm(hid_channels))
        
        self.mlp = torch.nn.Linear(hid_channels, out_channels)
        self.embedding_h = AtomEncoder(emb_dim=hid_channels)
        self.embedding_b = BondEncoder(emb_dim=hid_channels)
        
    def forward(self, x, edge_index, edge_attr, batch):
        x = self.embedding_h(x)
        e = self.embedding_b(edge_attr)
        
        for i in range(len(self.layers)):
            x_h = x
            x = self.layers[i](x, edge_index, e)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = x_h + x
            x = F.dropout(x, self.dropout, training=self.training)
            
        x = global_mean_pool(x, batch)
        x = self.mlp(x)

        return x
    
class GAT_Net(torch.nn.Module):
    def __init__(self, hid_channels, out_channels, heads, num_layers, dropout):
        super(GAT_Net, self).__init__()
        
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.dropout = dropout
        for _ in range(self.num_layers):
            self.layers.append(GATConv(in_channels=hid_channels, out_channels=hid_channels, heads=heads, add_self_loops=True))
            self.batch_norms.append(BatchNorm(hid_channels))
        
        self.mlp = torch.nn.Linear(hid_channels, out_channels)
        self.embedding_h = AtomEncoder(emb_dim=hid_channels)
        
    def forward(self, x, edge_index, batch):
        x = self.embedding_h(x)
        
        for i in range(len(self.layers)):
            x_h = x
            x = self.layers[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = x_h + x
            x = F.dropout(x, self.dropout, training=self.training)
            
        x = global_mean_pool(x, batch)
        x = self.mlp(x)

        return x
    
class GATv2_Net(torch.nn.Module):
    def __init__(self, hid_channels, out_channels, heads, num_layers, dropout):
        super(GATv2_Net, self).__init__()
        
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.dropout = dropout
        for _ in range(self.num_layers):
            self.layers.append(GATv2Conv(in_channels=hid_channels, out_channels=hid_channels, heads=heads, add_self_loops=True))
            self.batch_norms.append(BatchNorm(hid_channels))
        
        self.mlp = torch.nn.Linear(hid_channels, out_channels)
        self.embedding_h = AtomEncoder(emb_dim=hid_channels)
        
    def forward(self, x, edge_index, batch):
        x = self.embedding_h(x)
        
        for i in range(len(self.layers)):
            x_h = x
            x = self.layers[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = x_h + x
            x = F.dropout(x, self.dropout, training=self.training)
            
        x = global_mean_pool(x, batch)
        x = self.mlp(x)

        return x
    
class Transformer_Net(torch.nn.Module):
    def __init__(self, hid_channels, out_channels, heads, num_layers, dropout):
        super(Transformer_Net, self).__init__()
        
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.dropout = dropout
        for _ in range(self.num_layers):
            self.layers.append(TransformerConv(in_channels=hid_channels, out_channels=hid_channels, heads=heads, add_self_loops=True))
            self.batch_norms.append(BatchNorm(hid_channels))
        
        self.mlp = torch.nn.Linear(hid_channels, out_channels)
        self.embedding_h = AtomEncoder(emb_dim=hid_channels)
        
    def forward(self, x, edge_index, batch):
        x = self.embedding_h(x)
        
        for i in range(len(self.layers)):
            x_h = x
            x = self.layers[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = x_h + x
            x = F.dropout(x, self.dropout, training=self.training)
            
        x = global_mean_pool(x, batch)
        x = self.mlp(x)

        return x
    
class ARMA_Net(torch.nn.Module):
    def __init__(self, hid_channels, out_channels, num_layers, dropout):
        super(ARMA_Net, self).__init__()
        
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.dropout = dropout
        for _ in range(self.num_layers):
            self.layers.append(ARMAConv(in_channels=hid_channels, out_channels=hid_channels, add_self_loops=True))
            self.batch_norms.append(BatchNorm(hid_channels))
        
        self.mlp = torch.nn.Linear(hid_channels, out_channels)
        self.embedding_h = AtomEncoder(emb_dim=hid_channels)
        
    def forward(self, x, edge_index, batch):
        x = self.embedding_h(x)
        
        for i in range(len(self.layers)):
            x_h = x
            x = self.layers[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = x_h + x
            x = F.dropout(x, self.dropout, training=self.training)
            
        x = global_mean_pool(x, batch)
        x = self.mlp(x)

        return x
    
class TAG_Net(torch.nn.Module):
    def __init__(self, hid_channels, out_channels, num_layers, dropout):
        super(TAG_Net, self).__init__()
        
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.dropout = dropout
        for _ in range(self.num_layers):
            self.layers.append(TAGConv(in_channels=hid_channels, out_channels=hid_channels, add_self_loops=True))
            self.batch_norms.append(BatchNorm(hid_channels))
        
        self.mlp = torch.nn.Linear(hid_channels, out_channels)
        self.embedding_h = AtomEncoder(emb_dim=hid_channels)
        
    def forward(self, x, edge_index, batch):
        x = self.embedding_h(x)
        
        for i in range(len(self.layers)):
            x_h = x
            x = self.layers[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = x_h + x
            x = F.dropout(x, self.dropout, training=self.training)
            
        x = global_mean_pool(x, batch)
        x = self.mlp(x)

        return x

class MoGNNs(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, deg):
        super(MoGNNs, self).__init__()
        
        self.pna_net = PNA_Net(hidden_channels, num_layers, deg)