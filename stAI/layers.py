import torch
from torch import nn
from torch_geometric.nn import GATConv
import torch.nn.functional as F



class MLPEncoder(nn.Module):
    def __init__(self, d_input, d_hidden, d_latent, dropout=0.0):
        super(MLPEncoder, self).__init__()
        self.d_input = d_input
        self.d_latent = d_latent
        self.d_hidden = d_hidden
        self.model = nn.Sequential(nn.Linear(d_input, d_hidden),
                                   nn.ReLU(),
                                   nn.Dropout(dropout),
                                   nn.Linear(d_hidden, d_latent))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0, std=0.1)
                torch.nn.init.normal_(module.bias, mean=0, std=0.1)

    def forward(self, x):
        z = self.model(x)
        return z


class MLPDecoder(nn.Module):
    def __init__(self, d_input, d_hidden, d_latent):
        super(MLPDecoder, self).__init__()
        self.d_input = d_input
        self.d_latent = d_latent
        self.d_hidden = d_hidden
        self.model = nn.Sequential(nn.Linear(d_latent, d_hidden),
                                   nn.ReLU(),
                                   nn.Linear(d_hidden, d_input))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0, std=0.1)
                torch.nn.init.normal_(module.bias, mean=0, std=0.1)
        
    def forward(self, z):
        x = self.model(z)
        return x


class GATLayer(nn.Module):
    def __init__(self, d_input, num_heads=4, n_layers=1, dropout=0.0, residual=True):
        super(GATLayer, self).__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.conv_layers = nn.ModuleList()
        self.n_layers = n_layers
        
        for i in range(self.n_layers):
            self.conv_layers.append(GATConv(d_input, d_input, heads=num_heads, concat=False))
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        if self.residual:
            resid = x
        for layer in self.conv_layers:
            x = layer(x, edge_index)
            x = self.dropout(F.leaky_relu(x))
        if self.residual:
            x += resid
        return x


class GATEncoder(nn.Module):
    def __init__(self, d_input, d_hidden, d_latent, num_heads=1, n_layers=1, dropout=0.0, residual=True):
        super().__init__()
        
        self.pre_fc = nn.Sequential(nn.Linear(d_input, d_hidden, bias=True),
                                    nn.ReLU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(d_hidden, d_latent, bias=True))
        self.post_fc = nn.Linear(d_latent, d_latent, bias=True)
        self.gat_layer = GATLayer(d_input=d_latent, num_heads=num_heads, n_layers=n_layers, dropout=dropout, residual=residual)
    
    def forward(self, x, e):
        z = self.pre_fc(x)
        z = self.gat_layer(z, e)
        z = self.post_fc(z)
        
        return z

class Classifier(nn.Module):
    
    def __init__(self, d_input, d_hidden, n_classes):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(d_input, d_hidden),
                                   nn.LeakyReLU(),
                                   nn.Linear(d_hidden, n_classes))
    
    def forward(self, x):
        return self.model(x)

class LinearClassifier(nn.Module):
    
    def __init__(self, d_input, n_classes):
        super().__init__()
        self.model = nn.Linear(d_input, n_classes)
    
    def forward(self, x):
        return self.model(x)


class ScaledDotProductAttention(nn.Module):
    
    def __init__(self, d_input):
        super().__init__()
        self.query = nn.Linear(d_input, d_input)
        self.key = nn.Linear(d_input, d_input)
        # self.value = nn.Linear(input_dim, input_dim)
        self.value = nn.Identity()
        
    def forward(self, query, key, value):
        Q = self.query(query) # (batch_size, output_dim)
        K = self.key(key).transpose(1, 0) # (batch_size, output_dim) -> (output_dim, batch_size)
        sim = torch.matmul(Q, K) # (batch_size, batch_size)
        attn_weights = F.softmax(sim, dim=-1) # (batch_size, batch_size)
        V = self.value(value)        
        attn_output = torch.matmul(attn_weights, V) # (batch_size, output_dim)
        return attn_output, attn_weights




class DistanceScore(nn.Module):
    
    def __init__(self, topk=None, method='softmax'):
        super().__init__()
        self.topk = topk
        assert method in ['softmax', 'linear']
        self.method = method
    
    def forward(self, X1, X2):
        neg_dist_matrix = -torch.cdist(X1, X2, p=2)
        out = torch.zeros_like(neg_dist_matrix)
        
        
        if self.method == 'softmax':
            if self.topk is None:
                out = F.softmax(neg_dist_matrix, dim=1)
            else:
                val, idx = torch.topk(neg_dist_matrix, k=self.topk, dim=1)
                score = F.softmax(val, dim=1)
                out = out.scatter_(dim=1, index=idx, src=score)
        
        elif self.method == 'linear':
            if self.topk is None:
                out = (1-neg_dist_matrix / neg_dist_matrix.sum(axis=1, keepdim=True))/(neg_dist_matrix.shape[1]-1)
            else:
                val, idx = torch.topk(neg_dist_matrix, k=self.topk, dim=1)
                score = (1-neg_dist_matrix / neg_dist_matrix.sum(axis=1, keepdim=True))/(neg_dist_matrix.shape[1]-1)
                out = out.scatter_(dim=1, index=idx, src=score)
        
        return out, score




class EuclideanAttention(nn.Module):
    
    def __init__(self, topk=None, method='softmax'):
        super().__init__()
        self.attn_score = DistanceScore(topk=topk, method=method)
        
    def forward(self, query, key, value):
        attn_weights, score = self.attn_score(query, key)
        out = torch.matmul(attn_weights, value)
        return out, score


