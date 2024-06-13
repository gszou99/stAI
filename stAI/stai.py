import torch
import torch.nn.functional as F
from torch import optim, nn
from .layers import (
    GATEncoder, 
    MLPEncoder, 
    MLPDecoder, 
    Classifier, 
    LinearClassifier, 
    EuclideanAttention, 
    ScaledDotProductAttention
)
from .losses import MMDLoss, PcorrLoss, CosineLoss
from .utils import cross_dist

class stAI(nn.Module):
    
    def __init__(self, 
                 d_input,
                 d_hidden,
                 d_latent,
                 n_classes,
                 lam_recon,
                 lam_mmd,
                 lam_cos,
                 lam_clf,
                 lam_impute,
                 lam_genegraph,
                 dropout=0.0,
                 class_weight=None,
                 attn_type='euclidean',
                 topk=50,
                 linear_clf=False):
        super().__init__()
        
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.d_latent = d_latent
        
        self.ST_encoder = GATEncoder(d_input=d_input, d_hidden=d_hidden, d_latent=d_latent, 
                                     num_heads=1, n_layers=1, dropout=dropout, residual=True)
        self.ST_decoder = MLPDecoder(d_latent=d_latent, d_hidden=d_hidden, d_input=d_input)
        
        
        self.SC_encoder = MLPEncoder(d_input=d_input, d_hidden=d_hidden, d_latent=d_latent, dropout=dropout)
        self.SC_decoder = MLPDecoder(d_latent=d_latent, d_hidden=d_hidden, d_input=d_input)
        
        if linear_clf:
            self.classifier = LinearClassifier(d_input=d_latent, n_classes=n_classes)
        else:
            self.classifier = Classifier(d_input=d_latent, d_hidden=d_hidden, n_classes=n_classes)
        
        if attn_type == 'euclidean':
            self.attn = EuclideanAttention(topk=topk, method='softmax')
        elif attn_type == 'attention':
            self.attn = ScaledDotProductAttention(d_input=d_latent)
        else:
            raise ValueError('Invalid attention type')
            
        
        self.mmd_loss = MMDLoss()
        self.mse_loss = nn.MSELoss()
        self.impute_loss = PcorrLoss()
        self.clf_loss = nn.CrossEntropyLoss(weight=class_weight)
        
        self.genegraph_loss = CosineLoss()
        
        self.lam_recon = lam_recon
        self.lam_mmd = lam_mmd
        self.lam_cos = lam_cos
        self.lam_clf = lam_clf
        self.lam_impute = lam_impute
        self.lam_genegraph = lam_genegraph
        
        
    
    def forward(self, ST_fit, ST_supervision, ST_edge, SC_fit, SC_supervision, SC_label, SC_genegraph):
        
        ST_latent = self.ST_encoder(ST_fit, ST_edge)
        SC_latent = self.SC_encoder(SC_fit)
        
        
        ST_recon = self.ST_decoder(ST_latent)
        SC_recon = self.SC_decoder(SC_latent)
        
        SC_pred_label = self.classifier(SC_latent)

        ST_impute_pred, _ = self.attn(ST_latent, SC_latent, SC_supervision)
        
        ST_genegraph = cross_dist(ST_fit, ST_impute_pred)
        
        loss_genegraph = self.genegraph_loss(ST_genegraph, SC_genegraph)
        
        
        loss_clf = self.clf_loss(SC_pred_label, SC_label)
        
        loss_impute = self.impute_loss(ST_impute_pred, ST_supervision)
        
        
        loss_mmd = self.mmd_loss(ST_latent, SC_latent)
        
        loss_recon = self.mse_loss(ST_fit, ST_recon) + self.mse_loss(SC_fit, SC_recon)
        

        
        loss_cos = (1 - torch.sum(F.normalize(self.SC_decoder(ST_latent), p=2) * F.normalize(ST_fit, p=2), 1)).mean()+\
            (1 - torch.sum(F.normalize(self.ST_decoder(SC_latent), p=2) * F.normalize(SC_fit, p=2), 1)).mean()
        
        
        
        loss = self.lam_recon * loss_recon + self.lam_mmd * loss_mmd \
            + self.lam_cos * loss_cos + self.lam_clf * loss_clf + self.lam_impute * loss_impute \
                + self.lam_genegraph * loss_genegraph
        
        
        
        return loss, loss_recon, loss_mmd, loss_cos, loss_clf, loss_impute, loss_genegraph
    
        
    def get_embedding(self, ST_data=None, ST_edge=None, SC_data=None):
        with torch.no_grad():
            if ST_data is None:
                return self.SC_encoder(SC_data)
            elif SC_data is None:
                return self.ST_encoder(ST_data, ST_edge)
            else:
                return self.ST_encoder(ST_data, ST_edge), self.SC_encoder(SC_data)
    
    def impute_attn(self, ST_fit, ST_edge, SC_fit, SC_ref):
        with torch.no_grad():
            ST_latent = self.ST_encoder(ST_fit, ST_edge)
            SC_latent = self.SC_encoder(SC_fit)
            imputed, _ = self.attn(ST_latent, SC_latent, SC_ref)
            return imputed
    
    def annotate(self, ST_fit, ST_edge):
        with torch.no_grad():
            ST_latent = self.ST_encoder(ST_fit, ST_edge)
            ST_pred_label = self.classifier(ST_latent)
            return F.softmax(ST_pred_label, dim=-1)
            

