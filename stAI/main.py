import os
import scanpy as sc
import numpy as np
import pandas as pd
import json
import yaml

from sklearn.model_selection import KFold
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix

from .datasets import spatialDataset, rnaDataset, spatialCollate, rnaCollate
from .utils import (
    pairwise_corr,
    set_seed,
    process_celltype,
    convert_to_dense
)
from .stai import stAI
from .logging import logger

def eval_and_impute(model, spatial_dataset, rna_dataset, ST_gt, SC_ref, spatial_knn, device='cuda'):
    model.eval()
    device = list(model.parameters())[0].device if device == 'cuda' else 'cpu'
    model.to(device)
    with torch.no_grad():
        
        ST_fit, ST_calibration, ST_edge = spatialCollate(device=device, knn=spatial_knn)(spatial_dataset)
        SC_fit, SC_calibration, SC_label, SC_genegraph = rnaCollate(device=device)(rna_dataset)
        
        # import pdb
        # pdb.set_trace()
        if SC_ref is not None:
            ST_test_impute = model.impute_attn(ST_fit, ST_edge, SC_fit, torch.Tensor(SC_ref).to(device)).detach().cpu().numpy()
        else:
            ST_test_impute = None
        if ST_gt is not None:
            test_impute_score = pairwise_corr(ST_test_impute, ST_gt)
        else:
            test_impute_score = np.nan
        return test_impute_score, ST_test_impute


def train_and_impute(adata_spatial, adata_rna, ST_gt, SC_ref, gene_fit, gene_calibration, model_parameters, training_parameters, random_seed, save_dir):
    # adata_spatial and adata_rna are already normalized
    gene_map = {
        'training_genes': list(gene_fit),
        'calibration_genes': list(gene_calibration)
    }
    json.dump(gene_map, open(os.path.join(save_dir, 'gene_map.json'), 'w'))
    os.makedirs(f"{save_dir}/state_dict", exist_ok=True)
    os.makedirs(f"{save_dir}/impute_res", exist_ok=True)
    set_seed(random_seed)
    device = torch.device(f"cuda:{training_parameters['device_id']}")
    ST_fit = adata_spatial[:,gene_fit].X.astype('float32')
    ST_calibration = adata_spatial[:,gene_calibration].X.astype('float32')
    ST_location = adata_spatial.obsm['spatial']
    
    SC_fit = adata_rna[:,gene_fit].X.astype('float32')
    SC_calibration = adata_rna[:,gene_calibration].X.astype('float32')
    SC_label = adata_rna.obs['intlabel'].values
    
    spatial_dataset = spatialDataset(ST_fit, ST_calibration, ST_location)
    rna_dataset = rnaDataset(SC_fit, SC_calibration, SC_label)
    
    spatial_batchsize = training_parameters['spatial_batchsize'] if training_parameters['spatial_batchsize'] != -1 else len(spatial_dataset)
    rna_batchsize = training_parameters['rna_batchsize'] if training_parameters['rna_batchsize'] != -1 else len(rna_dataset)
    spatial_loader = DataLoader(spatial_dataset, batch_size=spatial_batchsize, collate_fn=spatialCollate(device=device, knn=training_parameters['spatial_knn']), drop_last=True, shuffle=True)
    rna_loader = DataLoader(rna_dataset, batch_size=rna_batchsize, collate_fn=rnaCollate(device=device), drop_last=True, shuffle=True)
    
    model = stAI(d_input=len(gene_fit), n_classes=len(np.unique(SC_label)), **model_parameters).to(device)
    optimizer = optim.Adam(model.parameters(), lr=training_parameters['lr'])
    num_epoch = training_parameters['num_epoch']
    for i in range(num_epoch):
        model.train()
        model.to(device)
        for spatial_batch, rna_batch in zip(spatial_loader, rna_loader):
            ST_fit_batch, ST_calibration_batch, ST_edge_batch = spatial_batch
            SC_fit_batch, SC_calibration_batch, SC_label_batch, SC_genegraph_batch = rna_batch
            
            optimizer.zero_grad()
            loss, loss_recon, loss_mmd, loss_cos, loss_clf, loss_impute, loss_genegraph \
                            = model(ST_fit_batch, ST_calibration_batch, ST_edge_batch, \
                                SC_fit_batch, SC_calibration_batch, SC_label_batch, SC_genegraph_batch)
            loss.backward()
            optimizer.step()
        if (i+1) % training_parameters['eval_step'] == 0:
            logger.info('Epoch [{}/{}], Loss: {:.4f}, Loss_recon: {:.4f}, Loss_mmd: {:.4f}, Loss_cos: {:.4f}, Loss_clf: {:.4f}, Loss_impute: {:.4f}, Loss_genegraph: {:.4f}'\
                        .format(i+1, num_epoch, loss.item(), loss_recon.item(), loss_mmd.item(), \
                            loss_cos.item(), loss_clf.item(), loss_impute.item(), loss_genegraph.item()))
    torch.save(model.state_dict(), os.path.join(save_dir, 'state_dict', f'epoch_{num_epoch}.pth'))
    logger.info(f"Save model state_dict of epoch {num_epoch} at {os.path.join(save_dir, 'state_dict', f'epoch_{num_epoch}.pth')}")
    test_impute_score, ST_test_impute = eval_and_impute(model, spatial_dataset, rna_dataset, ST_gt, SC_ref, training_parameters['spatial_knn'])
    logger.info('Test Score: {:.4f}'.format(test_impute_score))
    if ST_test_impute is not None:
        np.save(os.path.join(save_dir, 'impute_res', f'epoch_{num_epoch}.npy'), ST_test_impute)

def run_impute(model_parameters, training_parameters, adata_spatial, adata_rna, save_dir, genes_to_impute=None, random_seed=8848):
    # accept adata before processed

    adata_spatial = adata_spatial.copy()
    adata_rna = adata_rna.copy()
    celltype2int, int2celltype = process_celltype(adata_spatial, adata_rna)
    
    sc.pp.normalize_total(adata_rna, target_sum=1e4)
    sc.pp.log1p(adata_rna)
    
    shared_genes = [g for g in adata_spatial.var.index if g in adata_rna.var.index]
    kfold_randomstate = np.random.RandomState(random_seed)
    
    
    logger.info(f"Starting training on {len(shared_genes)} genes")
    
    gene_train = np.array(shared_genes)
        
    adata_rna_train = adata_rna[:, gene_train].copy()
    adata_spatial_train = adata_spatial[:, gene_train].copy()
        
        
    # WARNING may cause an exception TODO
    if genes_to_impute is not None:
        SC_ref = convert_to_dense(adata_rna[:,genes_to_impute]).X
    else:
        SC_ref = None
    ST_gt = None
        
    sc.pp.scale(adata_rna_train, max_value=10)
    sc.pp.scale(adata_spatial_train, max_value=10)
    if isinstance(adata_rna_train.X, csr_matrix):
        adata_rna_train.X = adata_rna_train.X.todense()
    if isinstance(adata_spatial_train.X, csr_matrix):
        adata_spatial_train.X = adata_spatial_train.X.todense()

    kfold = KFold(n_splits = training_parameters['k_folds'], shuffle=True, random_state=kfold_randomstate)
    temp_impute_res = {}
    for i_submodel, (gene_fit_idx, gene_calibration_idx) in enumerate(kfold.split(gene_train)):
        gene_fit = gene_train[gene_fit_idx]
        gene_calibration = gene_train[gene_calibration_idx]
        logger.info(f"Training submodel {i_submodel+1}/{training_parameters['k_folds']}...")      
        logger.info(f"Training genes: {gene_fit}, Calibration genes: {gene_calibration}")
        temp_save_dir = f"{save_dir}/submodel_{i_submodel}"
        os.makedirs(temp_save_dir, exist_ok=True)
        train_and_impute(adata_spatial_train, adata_rna_train, ST_gt, SC_ref, gene_fit, gene_calibration, model_parameters, training_parameters, random_seed, temp_save_dir)


# modifying
def run_impute_loo(model_parameters, training_parameters, adata_spatial, adata_rna, save_dir, random_seed=8848):
    # accept adata before processed

    adata_spatial = adata_spatial.copy()
    adata_rna = adata_rna.copy()
    celltype2int, int2celltype = process_celltype(adata_spatial, adata_rna)
    
    sc.pp.normalize_total(adata_rna, target_sum=1e4)
    sc.pp.log1p(adata_rna)
    
    shared_genes = [g for g in adata_spatial.var.index if g in adata_rna.var.index]
    kfold_randomstate = np.random.RandomState(random_seed)
    logger.info(f"Starting perform leave-one-out cross-validation on {len(shared_genes)} genes")
    for gene_test in shared_genes:
        logger.info(f'Predicting gene {gene_test}')
        gene_train = np.array([g for g in shared_genes if g != gene_test])
        
        adata_rna_train = adata_rna[:, gene_train].copy()
        adata_spatial_train = adata_spatial[:, gene_train].copy()
        
        
        # WARNING may cause an exception TODO
        ST_gt = adata_spatial[:, gene_test].X.toarray()
        SC_ref = adata_rna[:,gene_test].X.toarray()
        
        sc.pp.scale(adata_rna_train, max_value=10)
        sc.pp.scale(adata_spatial_train, max_value=10)
        if isinstance(adata_rna_train.X, csr_matrix):
            adata_rna_train.X = adata_rna_train.X.todense()
        if isinstance(adata_spatial_train.X, csr_matrix):
            adata_spatial_train.X = adata_spatial_train.X.todense()

        kfold = KFold(n_splits = training_parameters['k_folds'], shuffle=True, random_state=kfold_randomstate)
        temp_impute_res = {}
        for i_submodel, (gene_fit_idx, gene_calibration_idx) in enumerate(kfold.split(gene_train)):
            gene_fit = gene_train[gene_fit_idx]
            gene_calibration = gene_train[gene_calibration_idx]
            logger.info(f"Training submodel {i_submodel+1}/{training_parameters['k_folds']}...")      
            logger.info(f"Training genes: {gene_fit}, Calibration genes: {gene_calibration}")
            temp_save_dir = f"{save_dir}/{gene_test}/submodel_{i_submodel}"
            os.makedirs(temp_save_dir, exist_ok=True)
            train_and_impute(adata_spatial_train, adata_rna_train, ST_gt, SC_ref, gene_fit, gene_calibration, model_parameters, training_parameters, random_seed, temp_save_dir)


def impute_unmeasured_genes(genes_to_impute, model_dir, epoch=None, agg=True, device='cpu'):
    cfg = yaml.safe_load(open(os.path.join(model_dir, 'train_config.yaml')))
    adata_spatial = sc.read_h5ad(cfg['data_paths']['spatial_data'])
    adata_rna = sc.read_h5ad(cfg['data_paths']['rna_data'])
    training_parameters = cfg['training_parameters']
    model_parameters = cfg['model_parameters']
    celltype2int, int2celltype = process_celltype(adata_spatial, adata_rna)
    
    sc.pp.normalize_total(adata_rna, target_sum=1e4)
    sc.pp.log1p(adata_rna)
    
    SC_ref = adata_rna[:, genes_to_impute].X.toarray()
    sc.pp.scale(adata_rna, max_value=10)
    sc.pp.scale(adata_spatial, max_value=10)
    
    imputed_results = []
    for i_submodel in range(cfg['training_parameters']['k_folds']):
        temp_dir = f"{model_dir}/submodel_{i_submodel}"
        gene_map = json.load(open(os.path.join(temp_dir, 'gene_map.json')))
        training_genes = gene_map['training_genes']
        calibration_genes = gene_map['calibration_genes']
        
        device = torch.device(f"cuda:{training_parameters['device_id']}") if device != 'cpu' else 'cpu'
        ST_fit = adata_spatial[:,training_genes].X.astype('float32')
        ST_calibration = adata_spatial[:,calibration_genes].X.astype('float32')
        ST_location = adata_spatial.obsm['spatial']
        
        SC_fit = adata_rna[:,training_genes].X.astype('float32')
        SC_calibration = adata_rna[:,calibration_genes].X.astype('float32')
        SC_label = adata_rna.obs['intlabel'].values
        
        spatial_dataset = spatialDataset(ST_fit, ST_calibration, ST_location)
        rna_dataset = rnaDataset(SC_fit, SC_calibration, SC_label)
        
        model = stAI(d_input=len(training_genes), n_classes=len(np.unique(SC_label)), **model_parameters).to(device)
        epochs = [int(f.split('_')[-1].split('.')[0]) for f in os.listdir(f"{temp_dir}/state_dict") if f.endswith('.pth')]
        if epoch is None:
            epoch = max(epochs)
        model.load_state_dict(torch.load(f"{temp_dir}/state_dict/epoch_{epoch}.pth"))
        with torch.no_grad():
        
            ST_fit, ST_calibration, ST_edge = spatialCollate(device=device, knn=training_parameters['spatial_knn'])(spatial_dataset)
            SC_fit, SC_calibration, SC_label, SC_genegraph = rnaCollate(device=device)(rna_dataset)
            
            submodel_impute_res = model.impute_attn(ST_fit, ST_edge, SC_fit, torch.Tensor(SC_ref).to(device)).detach().cpu().numpy()
        imputed_results.append(submodel_impute_res)
    
    imputed_results = np.stack(imputed_results, axis=0)
    if agg:
        imputed_results = np.mean(imputed_results, axis=0)
    return imputed_results
        
        
    
    