from pytorch_metric_learning import losses 
import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.weight_norm import weight_norm
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
import os
import torch.nn.functional as F
import itertools

torch.autograd.set_detect_anomaly(True)
       
class DualMSLoss_FineGrained(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, margin=0, max_violation=False):
        super(DualMSLoss_FineGrained, self).__init__()
        self.margin = margin
        self.max_violation = max_violation
        self.thresh = 0.5
        self.margin = 0.7 # 0.1
        self.scale_pos = 2
        self.scale_neg = 40.0

    def ms_sample(self,sim_mat,label):
        pos_exp     = torch.exp(-self.scale_pos*(sim_mat-self.thresh))
        neg_exp     = torch.exp( self.scale_neg*(sim_mat-self.thresh))
        pos_mask    = torch.eq(label.view(-1,1)-label.view(1,-1),0.0).float().cuda()
        neg_mask    = 1 - pos_mask
        P_sim       = torch.where(pos_mask == 1,sim_mat,torch.ones_like(pos_exp)*1e16)
        N_sim       = torch.where(neg_mask == 1,sim_mat,torch.ones_like(neg_exp)*-1e16)
        min_P_sim,_ = torch.min(P_sim,dim=1,keepdim=True)
        max_N_sim,_ = torch.max(N_sim,dim=1,keepdim=True)
        hard_P_sim  = torch.where(P_sim - self.margin < max_N_sim,pos_exp,torch.zeros_like(pos_exp)).sum(dim=-1)
        hard_N_sim  = torch.where(N_sim + self.margin > min_P_sim,neg_exp,torch.zeros_like(neg_exp)).sum(dim=-1)
        pos_loss    = torch.log(1+hard_P_sim).sum()/self.scale_pos
        neg_loss    = torch.log(1+hard_N_sim).sum()/self.scale_neg
        
        return pos_loss + neg_loss

    def ms_sample_cbcb_clcl(self,sim_mat,label):
        pos_exp = torch.exp(-self.scale_pos*(sim_mat-self.thresh))
        neg_exp = torch.exp( self.scale_neg*(sim_mat-self.thresh))
        pos_mask = torch.eq(label.view(-1,1)-label.view(1,-1),0.0).float().cuda()
        
        pos_mask = pos_mask + torch.eye(pos_mask.shape[0]).cuda()
        P_sim = torch.where(pos_mask == 1,sim_mat,torch.ones_like(pos_exp)*1e16)
        N_sim = torch.where(pos_mask == 0,sim_mat,torch.ones_like(neg_exp)*-1e16)
        min_P_sim,_ = torch.min(P_sim,dim=1,keepdim=True)
        max_N_sim,_ = torch.max(N_sim,dim=1,keepdim=True)
        hard_P_sim = torch.where(P_sim - self.margin < max_N_sim,pos_exp,torch.zeros_like(pos_exp)).sum(dim=-1)
        hard_N_sim = torch.where(N_sim + self.margin > min_P_sim,neg_exp,torch.zeros_like(neg_exp)).sum(dim=-1)
        pos_loss = torch.log(1+hard_P_sim).sum()/self.scale_pos
        neg_loss = torch.log(1+hard_N_sim).sum()/self.scale_neg
        
        return pos_loss + neg_loss
    
    def ms_sample_cbcb_clcl_trans(self,sim_mat,label):
        pos_exp = torch.exp(-self.scale_pos*(sim_mat-self.thresh))
        neg_exp = torch.exp( self.scale_neg*(sim_mat-self.thresh))
        pos_mask = torch.eq(label.view(-1,1)-label.view(1,-1),0.0).float().cuda()
        
        n_sha = pos_mask.shape[0]
        mask_pos = torch.ones(n_sha, n_sha, dtype=torch.bool)
        mask_pos = mask_pos.triu(1) | mask_pos.tril(-1)
        pos_mask = torch.transpose(torch.transpose(pos_mask[mask_pos].reshape(n_sha, n_sha-1),0,1),0,1)

        neg_mask = 1-pos_mask
        P_sim = torch.where(pos_mask == 1,sim_mat,torch.ones_like(pos_exp)*1e16)
        N_sim = torch.where(neg_mask == 1,sim_mat,torch.ones_like(neg_exp)*-1e16)
        min_P_sim,_ = torch.min(P_sim,dim=1,keepdim=True)
        max_N_sim,_ = torch.max(N_sim,dim=1,keepdim=True)
        hard_P_sim = torch.where(P_sim - self.margin < max_N_sim,pos_exp,torch.zeros_like(pos_exp)).sum(dim=-1)
        hard_N_sim = torch.where(N_sim + self.margin > min_P_sim,neg_exp,torch.zeros_like(neg_exp)).sum(dim=-1)
        pos_loss = torch.log(1+hard_P_sim).sum()/self.scale_pos
        neg_loss = torch.log(1+hard_N_sim).sum()/self.scale_neg
        
        return pos_loss + neg_loss

    def compute_sharded_cosine_similarity(self, tensor1, tensor2, shard_size):
        B, T, D = tensor1.shape
        average_sim_matrix = torch.zeros((B, B), device=tensor1.device)

        for start_idx in range(0, T, shard_size):
            end_idx = min(start_idx + shard_size, T)

            # Get the shard
            shard_tensor1 = tensor1[:, start_idx:end_idx, :]
            shard_tensor2 = tensor2[:, start_idx:end_idx, :]

            # Reshape and expand
            shard_tensor1_expanded = shard_tensor1.unsqueeze(1).unsqueeze(3)
            shard_tensor2_expanded = shard_tensor2.unsqueeze(0).unsqueeze(2)

            # Compute cosine similarity for the shard
            shard_cos_sim = F.cosine_similarity(shard_tensor1_expanded, shard_tensor2_expanded, dim=-1)

            # Accumulate the sum of cosine similarities
            average_sim_matrix += torch.sum(shard_cos_sim, dim=[2, 3])

        # Normalize by the total number of elements (T*T)
        average_sim_matrix /= (T * T)

        return average_sim_matrix

    def forward(self, x_contactless, x_contactbased, x_cl_tokens, x_cb_tokens, labels, device):
                
        sim_mat_clcl = F.linear(self.l2_norm(x_contactless), self.l2_norm(x_contactless))
        n = sim_mat_clcl.shape[0]
        sim_mat_cbcb = F.linear(self.l2_norm(x_contactbased), self.l2_norm(x_contactbased))
        sim_mat_cbcl = F.linear(self.l2_norm(x_contactbased), self.l2_norm(x_contactless))

        loss2              = self.ms_sample_cbcb_clcl(sim_mat_clcl, labels).cuda() + self.ms_sample_cbcb_clcl(sim_mat_clcl.t(), labels).cuda()
        loss3              = self.ms_sample_cbcb_clcl(sim_mat_cbcb, labels).cuda() + self.ms_sample_cbcb_clcl(sim_mat_cbcb.t(), labels).cuda()

        loss4              = self.ms_sample(sim_mat_cbcl, labels).cuda() + self.ms_sample(sim_mat_cbcl.t(), labels).cuda()
        return loss4 + loss2 + loss3#+ (1.5*loss2) + (1.5*loss3)  # + loss2 + loss3#+ loss5 # 0.1*loss5  + loss3

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-12)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        
        return output
        
class DualMSLoss_FineGrained_domain_agnostic(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, margin=0, max_violation=False):
        super(DualMSLoss_FineGrained_domain_agnostic, self).__init__()
        self.margin = margin
        self.max_violation = max_violation
        self.thresh = 0.5
        self.margin = 0.5 # 0.1
        self.scale_pos = 2
        self.scale_neg = 40.0
        self.criterion = nn.CrossEntropyLoss()

    def ms_sample(self,sim_mat,label):
        pos_exp     = torch.exp(-self.scale_pos*(sim_mat-self.thresh))
        neg_exp     = torch.exp( self.scale_neg*(sim_mat-self.thresh))
        pos_mask    = torch.eq(label.view(-1,1)-label.view(1,-1),0.0).float().cuda()
        neg_mask    = 1 - pos_mask
        P_sim       = torch.where(pos_mask == 1,sim_mat,torch.ones_like(pos_exp)*1e16)
        N_sim       = torch.where(neg_mask == 1,sim_mat,torch.ones_like(neg_exp)*-1e16)
        min_P_sim,_ = torch.min(P_sim,dim=1,keepdim=True)
        max_N_sim,_ = torch.max(N_sim,dim=1,keepdim=True)
        hard_P_sim  = torch.where(P_sim - self.margin < max_N_sim,pos_exp,torch.zeros_like(pos_exp)).sum(dim=-1)
        hard_N_sim  = torch.where(N_sim + self.margin > min_P_sim,neg_exp,torch.zeros_like(neg_exp)).sum(dim=-1)
        pos_loss    = torch.log(1+hard_P_sim).sum()/self.scale_pos
        neg_loss    = torch.log(1+hard_N_sim).sum()/self.scale_neg
        
        return pos_loss + neg_loss

    def ms_sample_cbcb_clcl(self,sim_mat,label):
        pos_exp = torch.exp(-self.scale_pos*(sim_mat-self.thresh))
        neg_exp = torch.exp( self.scale_neg*(sim_mat-self.thresh))
        pos_mask = torch.eq(label.view(-1,1)-label.view(1,-1),0.0).float().cuda()
        
        pos_mask = pos_mask + torch.eye(pos_mask.shape[0]).cuda()
        P_sim = torch.where(pos_mask == 1,sim_mat,torch.ones_like(pos_exp)*1e16)
        N_sim = torch.where(pos_mask == 0,sim_mat,torch.ones_like(neg_exp)*-1e16)
        min_P_sim,_ = torch.min(P_sim,dim=1,keepdim=True)
        max_N_sim,_ = torch.max(N_sim,dim=1,keepdim=True)
        hard_P_sim = torch.where(P_sim - self.margin < max_N_sim,pos_exp,torch.zeros_like(pos_exp)).sum(dim=-1)
        hard_N_sim = torch.where(N_sim + self.margin > min_P_sim,neg_exp,torch.zeros_like(neg_exp)).sum(dim=-1)
        pos_loss = torch.log(1+hard_P_sim).sum()/self.scale_pos
        neg_loss = torch.log(1+hard_N_sim).sum()/self.scale_neg
        
        return pos_loss + neg_loss
    
    def ms_sample_cbcb_clcl_trans(self,sim_mat,label):
        pos_exp = torch.exp(-self.scale_pos*(sim_mat-self.thresh))
        neg_exp = torch.exp( self.scale_neg*(sim_mat-self.thresh))
        pos_mask = torch.eq(label.view(-1,1)-label.view(1,-1),0.0).float().cuda()
        
        n_sha = pos_mask.shape[0]
        mask_pos = torch.ones(n_sha, n_sha, dtype=torch.bool)
        mask_pos = mask_pos.triu(1) | mask_pos.tril(-1)
        pos_mask = torch.transpose(torch.transpose(pos_mask[mask_pos].reshape(n_sha, n_sha-1),0,1),0,1)

        neg_mask = 1-pos_mask
        P_sim = torch.where(pos_mask == 1,sim_mat,torch.ones_like(pos_exp)*1e16)
        N_sim = torch.where(neg_mask == 1,sim_mat,torch.ones_like(neg_exp)*-1e16)
        min_P_sim,_ = torch.min(P_sim,dim=1,keepdim=True)
        max_N_sim,_ = torch.max(N_sim,dim=1,keepdim=True)
        hard_P_sim = torch.where(P_sim - self.margin < max_N_sim,pos_exp,torch.zeros_like(pos_exp)).sum(dim=-1)
        hard_N_sim = torch.where(N_sim + self.margin > min_P_sim,neg_exp,torch.zeros_like(neg_exp)).sum(dim=-1)
        pos_loss = torch.log(1+hard_P_sim).sum()/self.scale_pos
        neg_loss = torch.log(1+hard_N_sim).sum()/self.scale_neg
        
        return pos_loss + neg_loss

    def compute_sharded_cosine_similarity(self, tensor1, tensor2, shard_size):
        B, T, D = tensor1.shape
        average_sim_matrix = torch.zeros((B, B), device=tensor1.device)

        for start_idx in range(0, T, shard_size):
            end_idx = min(start_idx + shard_size, T)

            # Get the shard
            shard_tensor1 = tensor1[:, start_idx:end_idx, :]
            shard_tensor2 = tensor2[:, start_idx:end_idx, :]

            # Reshape and expand
            shard_tensor1_expanded = shard_tensor1.unsqueeze(1).unsqueeze(3)
            shard_tensor2_expanded = shard_tensor2.unsqueeze(0).unsqueeze(2)

            # Compute cosine similarity for the shard
            shard_cos_sim = F.cosine_similarity(shard_tensor1_expanded, shard_tensor2_expanded, dim=-1)

            # Accumulate the sum of cosine similarities
            average_sim_matrix += torch.sum(shard_cos_sim, dim=[2, 3])

        # Normalize by the total number of elements (T*T)
        average_sim_matrix /= (T * T)

        return average_sim_matrix

    def forward(self, x_contactless, x_contactbased, x_cl_tokens, x_cb_tokens, labels, device, domain_class_cl, domain_class_cb, domain_class_cl_gt, domain_class_cb_gt):
                
        sim_mat_clcl = F.linear(self.l2_norm(x_contactless), self.l2_norm(x_contactless))
        n = sim_mat_clcl.shape[0]

        sim_mat_cbcb = F.linear(self.l2_norm(x_contactbased), self.l2_norm(x_contactbased))
        sim_mat_cbcl = F.linear(self.l2_norm(x_contactbased), self.l2_norm(x_contactless))

        loss2                = self.ms_sample_cbcb_clcl(sim_mat_clcl, labels).cuda() + self.ms_sample_cbcb_clcl(sim_mat_clcl.t(), labels).cuda()
        loss3                = self.ms_sample_cbcb_clcl(sim_mat_cbcb, labels).cuda() + self.ms_sample_cbcb_clcl(sim_mat_cbcb.t(), labels).cuda()

        loss4                = self.ms_sample(sim_mat_cbcl, labels).cuda() + self.ms_sample(sim_mat_cbcl.t(), labels).cuda()

        pred = torch.cat([domain_class_cl,    domain_class_cb])
        gt = torch.cat([domain_class_cl_gt,   domain_class_cb_gt])
        
        domain_class_loss = self.criterion(pred,gt)
        return loss4  + loss2 + loss3 + (3*domain_class_loss)

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-12)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        
        return output
    
class DualMSLoss_FineGrained_domain_agnostic_ft(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, margin=0, max_violation=False):
        super(DualMSLoss_FineGrained_domain_agnostic_ft, self).__init__()
        self.margin = margin
        self.max_violation = max_violation
        self.thresh = 0.5
        self.margin = 0.7 # 0.1
        self.scale_pos = 2
        self.scale_neg = 40.0
        self.criterion = nn.CrossEntropyLoss()

    def ms_sample(self,sim_mat,label):
        pos_exp     = torch.exp(-self.scale_pos*(sim_mat-self.thresh))
        neg_exp     = torch.exp( self.scale_neg*(sim_mat-self.thresh))
        pos_mask    = torch.eq(label.view(-1,1)-label.view(1,-1),0.0).float().cuda()
        neg_mask    = 1 - pos_mask
        P_sim       = torch.where(pos_mask == 1,sim_mat,torch.ones_like(pos_exp)*1e16)
        N_sim       = torch.where(neg_mask == 1,sim_mat,torch.ones_like(neg_exp)*-1e16)
        min_P_sim,_ = torch.min(P_sim,dim=1,keepdim=True)
        max_N_sim,_ = torch.max(N_sim,dim=1,keepdim=True)
        hard_P_sim  = torch.where(P_sim - self.margin < max_N_sim,pos_exp,torch.zeros_like(pos_exp)).sum(dim=-1)
        hard_N_sim  = torch.where(N_sim + self.margin > min_P_sim,neg_exp,torch.zeros_like(neg_exp)).sum(dim=-1)
        pos_loss    = torch.log(1+hard_P_sim).sum()/self.scale_pos
        neg_loss    = torch.log(1+hard_N_sim).sum()/self.scale_neg
        
        return pos_loss + neg_loss

    def ms_sample_cbcb_clcl(self,sim_mat,label):
        pos_exp = torch.exp(-self.scale_pos*(sim_mat-self.thresh))
        neg_exp = torch.exp( self.scale_neg*(sim_mat-self.thresh))
        pos_mask = torch.eq(label.view(-1,1)-label.view(1,-1),0.0).float().cuda()
        
        pos_mask = pos_mask + torch.eye(pos_mask.shape[0]).cuda()
        P_sim = torch.where(pos_mask == 1,sim_mat,torch.ones_like(pos_exp)*1e16)
        N_sim = torch.where(pos_mask == 0,sim_mat,torch.ones_like(neg_exp)*-1e16)
        min_P_sim,_ = torch.min(P_sim,dim=1,keepdim=True)
        max_N_sim,_ = torch.max(N_sim,dim=1,keepdim=True)
        hard_P_sim = torch.where(P_sim - self.margin < max_N_sim,pos_exp,torch.zeros_like(pos_exp)).sum(dim=-1)
        hard_N_sim = torch.where(N_sim + self.margin > min_P_sim,neg_exp,torch.zeros_like(neg_exp)).sum(dim=-1)
        pos_loss = torch.log(1+hard_P_sim).sum()/self.scale_pos
        neg_loss = torch.log(1+hard_N_sim).sum()/self.scale_neg
        
        return pos_loss + neg_loss
    
    def ms_sample_cbcb_clcl_trans(self,sim_mat,label):
        pos_exp = torch.exp(-self.scale_pos*(sim_mat-self.thresh))
        neg_exp = torch.exp( self.scale_neg*(sim_mat-self.thresh))
        pos_mask = torch.eq(label.view(-1,1)-label.view(1,-1),0.0).float().cuda()
        
        n_sha = pos_mask.shape[0]
        mask_pos = torch.ones(n_sha, n_sha, dtype=torch.bool)
        mask_pos = mask_pos.triu(1) | mask_pos.tril(-1)
        pos_mask = torch.transpose(torch.transpose(pos_mask[mask_pos].reshape(n_sha, n_sha-1),0,1),0,1)

        neg_mask = 1-pos_mask
        P_sim = torch.where(pos_mask == 1,sim_mat,torch.ones_like(pos_exp)*1e16)
        N_sim = torch.where(neg_mask == 1,sim_mat,torch.ones_like(neg_exp)*-1e16)
        min_P_sim,_ = torch.min(P_sim,dim=1,keepdim=True)
        max_N_sim,_ = torch.max(N_sim,dim=1,keepdim=True)
        hard_P_sim = torch.where(P_sim - self.margin < max_N_sim,pos_exp,torch.zeros_like(pos_exp)).sum(dim=-1)
        hard_N_sim = torch.where(N_sim + self.margin > min_P_sim,neg_exp,torch.zeros_like(neg_exp)).sum(dim=-1)
        pos_loss = torch.log(1+hard_P_sim).sum()/self.scale_pos
        neg_loss = torch.log(1+hard_N_sim).sum()/self.scale_neg
        
        return pos_loss + neg_loss

    def compute_sharded_cosine_similarity(self, tensor1, tensor2, shard_size):
        B, T, D = tensor1.shape
        average_sim_matrix = torch.zeros((B, B), device=tensor1.device)

        for start_idx in range(0, T, shard_size):
            end_idx = min(start_idx + shard_size, T)

            # Get the shard
            shard_tensor1 = tensor1[:, start_idx:end_idx, :]
            shard_tensor2 = tensor2[:, start_idx:end_idx, :]

            # Reshape and expand
            shard_tensor1_expanded = shard_tensor1.unsqueeze(1).unsqueeze(3)
            shard_tensor2_expanded = shard_tensor2.unsqueeze(0).unsqueeze(2)

            # Compute cosine similarity for the shard
            shard_cos_sim = F.cosine_similarity(shard_tensor1_expanded, shard_tensor2_expanded, dim=-1)

            # Accumulate the sum of cosine similarities
            average_sim_matrix += torch.sum(shard_cos_sim, dim=[2, 3])

        # Normalize by the total number of elements (T*T)
        average_sim_matrix /= (T * T)

        return average_sim_matrix

    def forward(self, x_contactless, x_contactbased, x_cl_tokens, x_cb_tokens, labels, device, domain_class_cl, domain_class_cb, domain_class_cl_gt, domain_class_cb_gt):
                
        sim_mat_clcl = F.linear(self.l2_norm(x_contactless), self.l2_norm(x_contactless))
        n = sim_mat_clcl.shape[0]

        sim_mat_cbcb = F.linear(self.l2_norm(x_contactbased), self.l2_norm(x_contactbased))
        sim_mat_cbcl = F.linear(self.l2_norm(x_contactbased), self.l2_norm(x_contactless))

        loss2                = self.ms_sample_cbcb_clcl(sim_mat_clcl, labels).cuda() + self.ms_sample_cbcb_clcl(sim_mat_clcl.t(), labels).cuda()
        loss3                = self.ms_sample_cbcb_clcl(sim_mat_cbcb, labels).cuda() + self.ms_sample_cbcb_clcl(sim_mat_cbcb.t(), labels).cuda()

        loss4                = self.ms_sample(sim_mat_cbcl, labels).cuda() + self.ms_sample(sim_mat_cbcl.t(), labels).cuda()

        return loss4  + loss2 + loss3 

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-12)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        
        return output
