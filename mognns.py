import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
from torch_geometric.nn import BatchNorm
from torch_geometric.nn import PNAConv, GCNConv, GraphConv, GINConv, GINEConv, GATConv, GATv2Conv, TransformerConv, ARMAConv, TAGConv, SAGEConv
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.graphproppred import Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from model import MoGNNs
from tqdm import tqdm

def train():
    model.train()
    for data in train_loader:
        data.to(device)
        x = []
        for embedding_h in embedding_h_list:
            embedding = embedding_h(data.x)
            x.append(embedding)
        edge_attr = embedding_b_list[0](data.edge_attr)
        out = model(x, data.edge_index, edge_attr, data.batch)
        # out = model(data.x, data.edge_index, data.batch)
        is_labeled = data.y == data.y
        loss = criterion(out[is_labeled], data.y.float()[is_labeled])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(loder):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data in loder:
            data.to(device)
            x = []
            for embedding_h in embedding_h_list:
                embedding = embedding_h(data.x)
                x.append(embedding)
            edge_attr = embedding_b_list[0](data.edge_attr)
            out = model(x, data.edge_index, edge_attr, data.batch)
            # out = model(data.x, data.edge_index, data.batch)
            y_true.append(data.y)
            y_pred.append(out)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_pred = torch.cat(y_pred, dim=0).cpu().detach().numpy()
    input_dict = {'y_true': y_true, 'y_pred': y_pred}

    return evaluator.eval(input_dict)

dataset = PygGraphPropPredDataset(name='ogbg-molhiv', root='./tmp/dataset')

evaluator = Evaluator(name='ogbg-molhiv')

loop_n = 1
list_train_rocauc = [[] for i in range(loop_n)]
list_valid_rocauc = [[] for i in range(loop_n)]
list_test_rocauc = [[] for i in range(loop_n)]
max_train_rocauc = []
max_valid_rocauc = []
max_test_rocauc = []

for loop in range(loop_n):

    split_idx = dataset.get_idx_split()
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    
    # Compute the maximum in-degree in the training data.
    max_degree = -1
    for data in dataset[split_idx["train"]]:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))

    # Compute the in-degree histogram tensor
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in dataset[split_idx["train"]]:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    
    num_models = 5
    heads = 4
    hid_channels = [70, 300, 75, 300, 300]
    model_names = ['PNA_Net', 'GINE_Net', 'GATv2_Net', 'ARMA_Net', 'GCNv2_Net']
    model_list = [torch.nn.ModuleList() for _ in range(num_models)]
    norm_list = [torch.nn.ModuleList() for _ in range(num_models)]
    embedding_h_list = []
    embedding_b_list = []
    num_layers_list = [4, 5, 5, 5, 5]
    
    # PNA Net
    saved_state_dict = torch.load('models/PNA.pth')
    
    aggregators = ['mean', 'min', 'max', 'std']
    scalers = ['identity', 'amplification', 'attenuation']
    for i in range(4):
        layer_params = {k[9:]: v for k, v in saved_state_dict.items() if 'layers.' + str(i) + '.' in k}
        new_layer = PNAConv(in_channels=hid_channels[0], out_channels=hid_channels[0],aggregators=aggregators,scalers=scalers,deg=deg, post_layers=1).to(device)
        new_layer_state_dict = new_layer.state_dict()
        new_layer_state_dict.update(layer_params)
        new_layer.load_state_dict(new_layer_state_dict)
        model_list[0].append(new_layer)
        
        norm_params = {k[14:]: v for k, v in saved_state_dict.items() if 'batch_norms.' + str(i) + '.' in k}
        new_batch_norm = BatchNorm(hid_channels[0]).to(device)
        new_batch_norm_state_dict = new_batch_norm.state_dict()
        new_batch_norm_state_dict.update(norm_params)
        new_batch_norm.load_state_dict(new_batch_norm_state_dict)
        norm_list[0].append(new_batch_norm)
        
    embedding_params = {k[12:]: v for k, v in saved_state_dict.items() if 'embedding_h.' in k}
    new_embedding_h = AtomEncoder(emb_dim=hid_channels[0]).to(device)
    new_embedding_h_state_dict = new_embedding_h.state_dict()
    new_embedding_h_state_dict.update(embedding_params)
    new_embedding_h.load_state_dict(new_embedding_h_state_dict)
    embedding_h_list.append(new_embedding_h)
        
    # GINE Net
    saved_state_dict = torch.load('models/GINE.pth')
    
    for i in range(5):
        layer_params = {k[9:]: v for k, v in saved_state_dict.items() if 'layers.' + str(i) + '.' in k}
        # print(layer_params["nn.2.weight"].size())
        new_layer = GINEConv(nn.Sequential(nn.Linear(hid_channels[1], hid_channels[1]), nn.ReLU(), nn.Linear(hid_channels[1], hid_channels[1]))).to(device)
        new_layer_state_dict = new_layer.state_dict()
        new_layer_state_dict.update(layer_params)
        new_layer.load_state_dict(new_layer_state_dict)
        model_list[1].append(new_layer)
        
        norm_params = {k[14:]: v for k, v in saved_state_dict.items() if 'batch_norms.' + str(i) + '.' in k}
        new_batch_norm = BatchNorm(hid_channels[1]).to(device)
        new_batch_norm_state_dict = new_batch_norm.state_dict()
        new_batch_norm_state_dict.update(norm_params)
        new_batch_norm.load_state_dict(new_batch_norm_state_dict)
        norm_list[1].append(new_batch_norm)
        
    embedding_params = {k[12:]: v for k, v in saved_state_dict.items() if 'embedding_h.' in k}
    new_embedding_h = AtomEncoder(emb_dim=hid_channels[1]).to(device)
    new_embedding_h_state_dict = new_embedding_h.state_dict()
    new_embedding_h_state_dict.update(embedding_params)
    new_embedding_h.load_state_dict(new_embedding_h_state_dict)
    embedding_h_list.append(new_embedding_h)
    
    embedding_params = {k[12:]: v for k, v in saved_state_dict.items() if 'embedding_b.' in k}
    new_embedding_b = BondEncoder(emb_dim=hid_channels[1]).to(device)
    new_embedding_b_state_dict = new_embedding_b.state_dict()
    new_embedding_b_state_dict.update(embedding_params)
    new_embedding_b.load_state_dict(new_embedding_b_state_dict)
    embedding_b_list.append(new_embedding_b)

    # GATv2 Net
    saved_state_dict = torch.load('models/GATv2.pth')

    for i in range(5):
        layer_params = {k[9:]: v for k, v in saved_state_dict.items() if 'layers.' + str(i) + '.' in k}
        new_layer = GATv2Conv(in_channels=hid_channels[2], out_channels=hid_channels[2], heads=heads, concat=False, add_self_loops=True).to(device)
        new_layer_state_dict = new_layer.state_dict()
        new_layer_state_dict.update(layer_params)
        new_layer.load_state_dict(new_layer_state_dict)
        model_list[2].append(new_layer)
        
        norm_params = {k[14:]: v for k, v in saved_state_dict.items() if 'batch_norms.' + str(i) + '.' in k}
        new_batch_norm = BatchNorm(hid_channels[2]).to(device)
        new_batch_norm_state_dict = new_batch_norm.state_dict()
        new_batch_norm_state_dict.update(norm_params)
        new_batch_norm.load_state_dict(new_batch_norm_state_dict)
        norm_list[2].append(new_batch_norm)
        
    embedding_params = {k[12:]: v for k, v in saved_state_dict.items() if 'embedding_h.' in k}
    new_embedding_h = AtomEncoder(emb_dim=hid_channels[2]).to(device)
    new_embedding_h_state_dict = new_embedding_h.state_dict()
    new_embedding_h_state_dict.update(embedding_params)
    new_embedding_h.load_state_dict(new_embedding_h_state_dict)
    embedding_h_list.append(new_embedding_h)
        
    # ARMA Net
    saved_state_dict = torch.load('models/ARMA.pth')

    for i in range(5):
        layer_params = {k[9:]: v for k, v in saved_state_dict.items() if 'layers.' + str(i) + '.' in k}
        new_layer = ARMAConv(in_channels=hid_channels[3], out_channels=hid_channels[3], add_self_loops=True).to(device)
        new_layer_state_dict = new_layer.state_dict()
        new_layer_state_dict.update(layer_params)
        new_layer.load_state_dict(new_layer_state_dict)
        model_list[3].append(new_layer)
        
        norm_params = {k[14:]: v for k, v in saved_state_dict.items() if 'batch_norms.' + str(i) + '.' in k}
        new_batch_norm = BatchNorm(hid_channels[3]).to(device)
        new_batch_norm_state_dict = new_batch_norm.state_dict()
        new_batch_norm_state_dict.update(norm_params)
        new_batch_norm.load_state_dict(new_batch_norm_state_dict)
        norm_list[3].append(new_batch_norm)
        
    embedding_params = {k[12:]: v for k, v in saved_state_dict.items() if 'embedding_h.' in k}
    new_embedding_h = AtomEncoder(emb_dim=hid_channels[3]).to(device)
    new_embedding_h_state_dict = new_embedding_h.state_dict()
    new_embedding_h_state_dict.update(embedding_params)
    new_embedding_h.load_state_dict(new_embedding_h_state_dict)
    embedding_h_list.append(new_embedding_h)
        
    # GCNv2 Net
    saved_state_dict = torch.load('models/GCNv2.pth')

    for i in range(5):
        layer_params = {k[9:]: v for k, v in saved_state_dict.items() if 'layers.' + str(i) + '.' in k}
        new_layer = GraphConv(in_channels=hid_channels[4], out_channels=hid_channels[4], add_self_loops=True).to(device)
        new_layer_state_dict = new_layer.state_dict()
        new_layer_state_dict.update(layer_params)
        new_layer.load_state_dict(new_layer_state_dict)
        model_list[4].append(new_layer)
        
        norm_params = {k[14:]: v for k, v in saved_state_dict.items() if 'batch_norms.' + str(i) + '.' in k}
        new_batch_norm = BatchNorm(hid_channels[4]).to(device)
        new_batch_norm_state_dict = new_batch_norm.state_dict()
        new_batch_norm_state_dict.update(norm_params)
        new_batch_norm.load_state_dict(new_batch_norm_state_dict)
        norm_list[4].append(new_batch_norm)
        
    embedding_params = {k[12:]: v for k, v in saved_state_dict.items() if 'embedding_h.' in k}
    new_embedding_h = AtomEncoder(emb_dim=hid_channels[4]).to(device)
    new_embedding_h_state_dict = new_embedding_h.state_dict()
    new_embedding_h_state_dict.update(embedding_params)
    new_embedding_h.load_state_dict(new_embedding_h_state_dict)
    embedding_h_list.append(new_embedding_h)
    
    # # TAG Net
    # saved_state_dict = torch.load('models/TAG.pth')

    # for i in range(5):
    #     layer_params = {k[9:]: v for k, v in saved_state_dict.items() if 'layers.' + str(i) + '.' in k}
    #     new_layer = TAGConv(in_channels=hid_channels[5], out_channels=hid_channels[5], add_self_loops=True).to(device)
    #     new_layer_state_dict = new_layer.state_dict()
    #     new_layer_state_dict.update(layer_params)
    #     new_layer.load_state_dict(new_layer_state_dict)
    #     model_list[5].append(new_layer)
        
    #     norm_params = {k[14:]: v for k, v in saved_state_dict.items() if 'batch_norms.' + str(i) + '.' in k}
    #     new_batch_norm = BatchNorm(hid_channels[5]).to(device)
    #     new_batch_norm_state_dict = new_batch_norm.state_dict()
    #     new_batch_norm_state_dict.update(norm_params)
    #     new_batch_norm.load_state_dict(new_batch_norm_state_dict)
    #     norm_list[5].append(new_batch_norm)
        
    # embedding_params = {k[12:]: v for k, v in saved_state_dict.items() if 'embedding_h.' in k}
    # new_embedding_h = AtomEncoder(emb_dim=hid_channels[5]).to(device)
    # new_embedding_h_state_dict = new_embedding_h.state_dict()
    # new_embedding_h_state_dict.update(embedding_params)
    # new_embedding_h.load_state_dict(new_embedding_h_state_dict)
    # embedding_h_list.append(new_embedding_h)
    
    dropout = [0.3, 0.5, 0.5, 0.3, 0.5]
    atten_dropout = 0.2
    model = MoGNNs(hid_channels=hid_channels, out_channels=1, model_names=model_names, model_list=model_list, norm_list=norm_list, num_layers_list=num_layers_list, dropout=dropout, atten_dropout=atten_dropout,).to(device)
    model_params = model.models.parameters()
    batch_norm_params = model.batch_norms.parameters()
    Q_params = model.Q_weight.parameters()
    K_params = model.K_weight.parameters()
    mlp_params = model.mlp.parameters()
    
    ori_lr = 0.00001
    cur_lr = 0.001
    
    all_params = [
        {'params': model_params, 'lr': ori_lr},
        {'params': batch_norm_params, 'lr': ori_lr},
        {'params': Q_params, 'lr': cur_lr},
        {'params': K_params, 'lr': cur_lr},
        {'params': mlp_params, 'lr': cur_lr}
    ]
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    optimizer = torch.optim.Adam(all_params)

    criterion = torch.nn.BCEWithLogitsLoss()
    max_valid = 0.0
    for epoch in tqdm(range(1,101)):
        train()
        train_acc = test(train_loader)
        valid_acc = test(valid_loader)
        test_acc = test(test_loader)
        list_train_rocauc[loop].append(list(train_acc.values()))
        list_valid_rocauc[loop].append(list(valid_acc.values()))
        list_test_rocauc[loop].append(list(test_acc.values()))

        print(
            f'Epoch: {epoch:03d}, Train AUC: {train_acc}, Valid AUC :{valid_acc}, Test AUC: {test_acc}')

    max_valid_rocauc_result = max(list_valid_rocauc[loop])
    index_test = list_valid_rocauc[loop].index(max_valid_rocauc_result)
    max_train_rocauc.append(list_train_rocauc[loop][index_test])
    max_valid_rocauc.append(list_valid_rocauc[loop][index_test])
    max_test_rocauc.append(list_test_rocauc[loop][index_test])

for i in range(loop_n):
    print('max_train_rocauc -> max_valid_rocauc -> max_test_rocauc:', max_train_rocauc[i], ' -> ', max_valid_rocauc[i],
          ' -> ', max_test_rocauc[i])


