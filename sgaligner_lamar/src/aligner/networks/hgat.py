import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv

class HGN_Conv(nn.Module):
    def __init__(self, n_units=[17, 128, 100], dim_hyperedge_emb=64, n_heads=[2, 2], dropout=0.0, attn_mode="node"):
        super(HGN_Conv, self).__init__()
        self.num_layers = len(n_units) - 1
        self.dropout = dropout
        layer_stack = []

        # in_channels, out_channels, heads
        for i in range(self.num_layers):
            in_channels = n_units[i]#* n_heads[i-1] if i else n_units[i]
            layer = HypergraphConv(in_channels=in_channels, out_channels=n_units[i+1], use_attention=False, heads=n_heads[i])
            if layer.bias is not None:
                layer.bias = nn.Parameter(torch.zeros(n_units[i+1]))
            layer_stack.append(layer)
        
        self.layer_stack = nn.ModuleList(layer_stack)

    def forward(self, x, hyperedges, hyperedge_attrs):
        for idx, hg_layer in enumerate(self.layer_stack):
            x = F.dropout(x, self.dropout, training=self.training)
            x = hg_layer(x, hyperedges)
            if idx+1 < self.num_layers:
                x = F.relu(x)
        
        return x
    
class HGN_Attn(nn.Module):
    def __init__(self, n_units=[17, 128, 100], dim_hyperedge_emb=64, n_heads=[2, 2], dropout=0.0, attn_mode="node"):
        super(HGN_Attn, self).__init__()
        self.num_layers = len(n_units) - 1
        self.dropout = dropout
        self.change_first_dim = not dim_hyperedge_emb==n_units[0] 
        layer_stack = []
        hyperedge_attr_stack = [] 

        if self.change_first_dim:
            hyperedge_attr_stack.append(nn.Linear(dim_hyperedge_emb, n_units[0])) 

        

        # in_channels, out_channels, heads
        for i in range(self.num_layers):
            in_channels = n_units[i]* n_heads[i-1] if i else n_units[i]
            layer = HypergraphConv(in_channels=in_channels, out_channels=n_units[i+1], use_attention=True, attention_mode=attn_mode, heads=n_heads[i])
            if layer.bias is not None:
                layer.bias = nn.Parameter(torch.zeros(n_heads[i]*n_units[i+1]))
            layer_stack.append(layer)

            # if the dimensions of X change, we have to change those of the hyperedge attributes accordingly, since they need to be the same for HyperGraphConv
            if in_channels!=n_heads[i]*n_units[i+1]:
                hyperedge_attr_stack.append(nn.Linear(in_channels, n_heads[i]*n_units[i+1]))
        
        self.layer_stack = nn.ModuleList(layer_stack)
        self.hyperedge_attr_stack = nn.ModuleList(hyperedge_attr_stack)

    def forward(self, x, hyperedges, hyperedge_attr):
       # hyperedges: sparse incidence matrix e.g. hyperedge_index = torch.tensor([[0, 1, 2, 1, 2, 3], [0, 0, 0, 1, 1, 1],]) for E {{0, 1, 2}, {1, 2, 3}} 
       # hyperedge_attr: hyperedge feature matrix of shape nr(hyperedges) x dim(attr)
        id_change_dim = 0
        for idx, hg_layer in enumerate(self.layer_stack):
            x = F.dropout(x, self.dropout, training=self.training)
            #if dimensions of embeddings of x and hyperedges don't align, pass through linear layer (needed for HyperGraphCov)
            if x.shape[1] != hyperedge_attr.shape[1]:
                hyperedge_attr = self.hyperedge_attr_stack[id_change_dim](hyperedge_attr) 
                id_change_dim += 1
            x = hg_layer(x, hyperedges, hyperedge_attr=hyperedge_attr)
            if idx+1 < self.num_layers:
                x = F.relu(x)
        
        return x
    

if __name__ == '__main__':
    
    hiddenUnits=[17, 128, 128]
    heads = [2, 2]
    
    # hid = 8
    # in_head = 8
    # out_head = 1
    numFeatures = 3
    
    x = torch.randn((19, numFeatures))
    edges = torch.Tensor([[1, 2], [2, 3], [1, 3]])
    edges = torch.transpose(edges, 0, 1).to(torch.int64)

    # model = MultiGAT(n_units=hiddenUnits, n_heads=heads)
    # out = model(x, edges)
    # print(out.size())

    print(x.shape, edges.shape)
    model = MultiGCN(n_units=[3, 256, 256])
    out = model(x, edges)
    print(out.size())
    # summary(model, [(3, 10), (10, 10)])