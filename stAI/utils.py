import random
import numpy as np
import pandas as pd
import torch
import pynndescent
from scipy.sparse import csr_matrix
from scipy import sparse
import torch_geometric.utils
import scanpy as sc
import yaml
import itertools
from sklearn.metrics import accuracy_score, f1_score, classification_report

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def pairwise_corr(x, y, keepdim=False):
    # calculate pairwise correlation for each features
    # to evaluate the impute result
    # x: n_sample x n_feature
    # y: n_sample x n_feature
    x_mean = np.mean(x, axis=0)
    y_mean = np.mean(y, axis=0)
    x_std = np.std(x, axis=0)
    y_std = np.std(y, axis=0)
    r = np.mean((x - x_mean) * (y - y_mean), axis=0) / (x_std * y_std+1e-6)
    if keepdim:
        return r
    else:
        return r.squeeze()

def get_knn(location, n_neighbors):
    index = pynndescent.NNDescent(location, n_neighbors=n_neighbors)
    neighbor_idx, neighbor_dist = index.neighbor_graph
    return neighbor_idx, neighbor_dist

def knn_to_adj(knn_idx, knn_dist):
    
    n_samples, n_neighbors = knn_idx.shape

    row = np.repeat(np.arange(n_samples), n_neighbors)
    col = knn_idx.flatten()
    data = np.ones(n_samples * n_neighbors)

    adj_matrix = csr_matrix((data, (row, col)), shape=(n_samples, n_samples))
    return adj_matrix

def location_to_edge(location, n_neighbors):
    idx, dist = get_knn(location, n_neighbors)
    adj = knn_to_adj(idx, dist)
    edge = torch_geometric.utils.from_scipy_sparse_matrix(adj)[0]
    return edge

def cross_dist(data1, data2, method='correlation'):
    """n * k1, n * k2 --> k1 * k2

    Parameters
    ----------
    method : str, optional
        ['correlation', 'cosine']
    """
    if method == 'correlation':
        data1_demean = (data1 - data1.mean(dim=0).unsqueeze(0))
        data2_demean = (data2 - data2.mean(dim=0).unsqueeze(0))
        
        
        cov = torch.einsum('ij, ik -> jk', data1_demean, data2_demean) / (data1_demean.shape[0]-1)
        var1 = torch.einsum('ij -> j', data1_demean**2) / (data1_demean.shape[0]-1)
        var2 = torch.einsum('ij -> j', data2_demean**2) / (data2_demean.shape[0]-1)
        out = cov / ((var1.unsqueeze(-1)* var2.unsqueeze(0)+1e-6)**.5)
    if method == 'cosine':
        numerator = torch.einsum('ij, ik -> jk', data1, data2)
        denominator = (torch.einsum('ij -> j', data1**2).unsqueeze(-1) * torch.einsum('ij -> j', data2**2).unsqueeze(0))**.5
        out = numerator / denominator
    
    return out

def process_celltype(adata_spatial, adata_rna):
    if 'celltype' in adata_spatial.obs.columns:
        all_celltype = list(adata_rna.obs.celltype.unique()) + [ct for ct in adata_spatial.obs.celltype.unique() if ct not in adata_rna.obs.celltype.unique()]
    else:
        all_celltype = adata_rna.obs.celltype.unique()
    celltype2int = {celltype:i for i, celltype in enumerate(all_celltype)}
    int2celltype = {i: celltype for celltype, i in celltype2int.items()}

    adata_rna.obs['intlabel'] = adata_rna.obs.celltype.map(celltype2int)
    if 'celltype' in adata_spatial.obs.columns:
        adata_spatial.obs['intlabel'] = adata_spatial.obs.celltype.map(celltype2int)
    return celltype2int, int2celltype

def normalize(adata_rna):
    adata_rna = adata_rna.copy()
    sc.pp.normalize_total(adata_rna, target_sum=1e4)

def dict_to_string(dictionary):
    result = ''
    for key, value in dictionary.items():
        if key not in ['device_id', 'eval_step']:
            result += f'{key}_{value}__'
    return result[:-2]

def generate_parameter_list(parameter_dict):
    keys = parameter_dict.keys()
    values = parameter_dict.values()
    parameter_combinations = itertools.product(*values)
    parameter_dict_list = []
    for combination in parameter_combinations:
        parameter_dict = dict(zip(keys, combination))
        parameter_dict_list.append(parameter_dict)

    return parameter_dict_list

def generate_parameter_list_all(all_parameter_grid):
    model_parameter_list = generate_parameter_list(all_parameter_grid['model_parameters'])
    training_parameter_list = generate_parameter_list(all_parameter_grid['training_parameters'])
    return [{'model_parameters':model_parameters, 'training_parameters':training_parameters} \
        for model_parameters in model_parameter_list for training_parameters in training_parameter_list]

def parse_yaml(config):
    with open(config, 'r') as rf:
        all_parameters = yaml.load(rf, Loader=yaml.FullLoader)
    return generate_parameter_list_all(all_parameters)

def evaluate_impute(adata_spatial, impute_result):
    # evaluate the ensembled result by the gene-wise correlation 
    # impute_result: indexed with ['test_gene', 'submodel', 'epoch']
    if isinstance(adata_spatial.X, np.ndarray):
        spatial_df = pd.DataFrame(adata_spatial.X, columns=adata_spatial.var.index).T
    else:
        spatial_df = pd.DataFrame(adata_spatial.X.toarray(), columns=adata_spatial.var.index).T
    result_ensembled = impute_result.groupby(['test_gene', 'epoch']).mean()
    all_genes = impute_result.index.get_level_values('test_gene').unique()
    ensembled_res = []
    for gene in all_genes:
        temp = pd.DataFrame()
        temp['epoch'] = result_ensembled.loc[gene].index
        out = result_ensembled.loc[gene].apply(lambda z:np.corrcoef(spatial_df.loc[gene].values, z.values.astype(float))[0,1], axis=1)
        temp['score'] = out.values
        temp.set_index('epoch', inplace=True)
        temp.columns = [gene]
        ensembled_res.append(temp)
    all_score = pd.concat(ensembled_res, axis=1)
    return all_score

def evaluate_impute_single(adata_spatial, impute_result, genes):
    # evaluate the ensembled result by the gene-wise correlation 
    # impute_result: indexed with ['submodel', 'epoch']
    
    spatial_df = pd.DataFrame(adata_spatial.X.toarray(), columns=adata_spatial.var.index).T
    result_ensembled = impute_result.groupby(['epoch']).mean()
    
    temp = pd.DataFrame()
    temp['epoch'] = result_ensembled.index
    
    out = result_ensembled.apply(lambda z:np.corrcoef(spatial_df.loc[genes].values, z.values.astype(float))[0,1], axis=1)
    temp['score'] = out.values
    temp.set_index('epoch', inplace=True)
    temp.columns = [gene]
        
    return temp

def evaluate_annotate(adata_spatial, annotate_result):
    # evaluate the ensembled result by the gene-wise correlation 
    groundtruth = adata_spatial.obs['intlabel'].values
    result_ensembled = annotate_result.groupby(['epoch', 'cell']).mean()
    accuracy = result_ensembled.groupby('epoch').apply(lambda z:(accuracy_score(groundtruth, z.values.argmax(axis=1)))).rename('accuracy')
    f1 = result_ensembled.groupby('epoch').apply(lambda z:(f1_score(groundtruth, z.values.argmax(axis=1), average='macro'))).rename('f1score')
    report = result_ensembled.groupby('epoch').apply(lambda z:(classification_report(groundtruth, z.values.argmax(axis=1), zero_division=0)))
    for index in report.index:
        print(f'epoch {index}')
        print(report[index])
    return pd.concat([accuracy, f1], axis=1)

def convert_to_dense(adata):
    adata = adata.copy()
    if isinstance(adata.X, sparse.spmatrix):
        adata.X = adata.X.toarray()
        return adata
    elif isinstance(adata.X, np.ndarray):
        return adata
    else:
        raise TypeError(f"Unsupported data type {type(adata.X)}")