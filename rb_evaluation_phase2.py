import torch
from datasets.rb_loader_cl import RB_loader_cl
from datasets.rb_loader_cb import RB_loader_cb
from utils import Prev_RetMetric, l2_norm, compute_recall_at_k
import numpy as np
from tqdm import tqdm
from model import SwinModel_Fusion as Model
from sklearn.metrics import roc_curve, auc
import json
import torch.nn.functional as F

def get_fused_cross_score_matrix(model, cl_tokens, cb_tokens):
    cl_tokens   = torch.cat(cl_tokens)
    cb_tokens   = torch.cat(cb_tokens)
    
    batch_size_cl = cl_tokens.shape[0]
    batch_size_cb = cb_tokens.shape[0]
    shard_size  = 20
    similarity_matrix = torch.zeros((batch_size_cl, batch_size_cb))
    for i_start in tqdm(range(0, batch_size_cl, shard_size)):
        i_end   = min(i_start + shard_size, batch_size_cl)
        shard_i = cl_tokens[i_start:i_end]
        for j_start in range(0, batch_size_cb, shard_size):
            j_end               = min(j_start + shard_size, batch_size_cb)
            shard_j             = cb_tokens[j_start:j_end]
            batch_i             = shard_i.unsqueeze(1)
            batch_j             = shard_j.unsqueeze(0)

            pairwise_i          = batch_i.expand(-1, shard_j.shape[0], -1, -1)
            pairwise_j          = batch_j.expand(shard_i.shape[0], -1, -1, -1)
            
            similarity_scores, distances = model.combine_features(
                pairwise_i.reshape(-1, 197, shard_i.shape[-1]), 
                pairwise_j.reshape(-1, 197, shard_j.shape[-1])
            )
            scores = similarity_scores - 0.1 * distances  #-0.1
            scores   = scores.reshape(shard_i.shape[0], shard_j.shape[0])
            similarity_matrix[i_start:i_end, j_start:j_end] = scores.cpu().detach()
    return similarity_matrix

device = torch.device('cuda')
data_cl = RB_loader_cl(split="test")
data_cb = RB_loader_cb(split="test")
dataloader_cb = torch.utils.data.DataLoader(data_cb,batch_size = 16, num_workers = 1, pin_memory = True)
dataloader_cl = torch.utils.data.DataLoader(data_cl,batch_size = 16, num_workers = 1, pin_memory = True)
model = Model().to(device)
checkpoint = torch.load("ridgeformer_checkpoints/phase2_scratch.pt",map_location = torch.device('cpu'))
model.load_state_dict(checkpoint,strict=False)

model.eval()
cl_feats, cb_feats, cl_labels, cb_labels, cl_fnames, cb_fnames, cl_feats_unnormed, cb_feats_unnormed = list(),list(),list(),list(),list(),list(),list(),list()
print("Computing Test Recall")

with torch.no_grad():
    for (x_cb, target) in tqdm(dataloader_cb):
        x_cb, label = x_cb.to(device), target.to(device)
        x_cb_token  = model.get_tokens(x_cb,'contactbased')
        label = label.cpu().detach().numpy()
        cb_feats.append(x_cb_token)
        cb_labels.append(label)

with torch.no_grad():
    for (x_cl, target) in tqdm(dataloader_cl):
        x_cl, label = x_cl.to(device), target.to(device)
        x_cl_token  = model.get_tokens(x_cl,'contactless')
        label = label.cpu().detach().numpy()
        cl_feats.append(x_cl_token)
        cl_labels.append(label)

cl_label = torch.from_numpy(np.concatenate(cl_labels))
cb_label = torch.from_numpy(np.concatenate(cb_labels))

# CB2CL <---------------------------------------->
scores_mat = get_fused_cross_score_matrix(model, cl_feats, cb_feats)
scores = scores_mat.cpu().detach().numpy().flatten().tolist()
labels = torch.eq(cl_label.view(-1,1) - cb_label.view(1,-1),0.0).flatten().tolist()
ids_mod = list()
for i in labels:
    if i==True:
        ids_mod.append(1)
    else:
        ids_mod.append(0)
fpr,tpr,thresh = roc_curve(labels,scores,drop_intermediate=True)
lower_fpr_idx = max(i for i, val in enumerate(fpr) if val < 0.01)
upper_fpr_idx = min(i for i, val in enumerate(fpr) if val >= 0.01)
tar_far_102 = tpr[upper_fpr_idx]#(tpr[lower_fpr_idx]+tpr[upper_fpr_idx])/2

lower_fpr_idx = max(i for i, val in enumerate(fpr) if val < 0.001)
upper_fpr_idx = min(i for i, val in enumerate(fpr) if val >= 0.001)
tar_far_103 = (tpr[lower_fpr_idx]+tpr[upper_fpr_idx])/2

lower_fpr_idx = max(i for i, val in enumerate(fpr) if val < 0.0001)
upper_fpr_idx = min(i for i, val in enumerate(fpr) if val >= 0.0001)
tar_far_104 = (tpr[lower_fpr_idx]+tpr[upper_fpr_idx])/2

fnr = 1 - tpr

EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
roc_auc = auc(fpr, tpr)
print(f"ROCAUC for CB2CL: {roc_auc * 100} %")
print(f"EER for CB2CL: {EER * 100} %")
eer_cb2cl = EER * 100
cbcltf102 = tar_far_102 * 100
cbcltf103 = tar_far_103 * 100
cbcltf104 = tar_far_104 * 100
cl_label = cl_label.cpu().detach()
cb_label = cb_label.cpu().detach()
print(f"TAR@FAR=10^-2 for CB2CL: {tar_far_102 * 100} %")
print(f"TAR@FAR=10^-3 for CB2CL: {tar_far_103 * 100} %")
print(f"TAR@FAR=10^-4 for CB2CL: {tar_far_104 * 100} %")

print(f"R@1 for CB2CL: {compute_recall_at_k(scores_mat, cl_label, cb_label, 1) * 100} %")
print(f"R@10 for CB2CL: {compute_recall_at_k(scores_mat, cl_label, cb_label, 10) * 100} %")
print(f"R@50 for CB2CL: {compute_recall_at_k(scores_mat, cl_label, cb_label, 50) * 100} %")
print(f"R@100 for CB2CL: {compute_recall_at_k(scores_mat, cl_label, cb_label, 100) * 100} %")

# CL2CL -------------------------
scores = get_fused_cross_score_matrix(model, cl_feats, cl_feats)
scores_mat = scores
row, col = torch.triu_indices(row=scores.size(0), col=scores.size(1), offset=1)
scores = scores[row, col]
labels = torch.eq(cl_label.view(-1,1) - cl_label.view(1,-1),0.0).float().cuda()
labels = labels[torch.triu(torch.ones(labels.shape),diagonal = 1) == 1]
scores = scores.cpu().detach().numpy().flatten().tolist()
labels = labels.flatten().tolist()
ids_mod = list()
for i in labels:
    if i==True:
        ids_mod.append(1)
    else:
        ids_mod.append(0)
fpr,tpr,thresh = roc_curve(labels,scores,drop_intermediate=True)

lower_fpr_idx = max(i for i, val in enumerate(fpr) if val < 0.01)
upper_fpr_idx = min(i for i, val in enumerate(fpr) if val >= 0.01)
tar_far_102 = tpr[upper_fpr_idx]#(tpr[lower_fpr_idx]+tpr[upper_fpr_idx])/2

lower_fpr_idx = max(i for i, val in enumerate(fpr) if val < 0.001)
upper_fpr_idx = min(i for i, val in enumerate(fpr) if val >= 0.001)
tar_far_103 = (tpr[lower_fpr_idx]+tpr[upper_fpr_idx])/2

lower_fpr_idx = max(i for i, val in enumerate(fpr) if val < 0.0001)
upper_fpr_idx = min(i for i, val in enumerate(fpr) if val >= 0.0001)
tar_far_104 = (tpr[lower_fpr_idx]+tpr[upper_fpr_idx])/2

fnr = 1 - tpr

EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
roc_auc = auc(fpr, tpr)
print(f"ROCAUC for CL2CL: {roc_auc * 100} %")
print(f"EER for CL2CL: {EER * 100} %")
eer_cb2cl = EER * 100
cbcltf102 = tar_far_102 * 100
cbcltf103 = tar_far_103 * 100
cbcltf104 = tar_far_104 * 100
cl_label = cl_label.cpu().detach()
print(f"TAR@FAR=10^-2 for CL2CL: {tar_far_102 * 100} %")
print(f"TAR@FAR=10^-3 for CL2CL: {tar_far_103 * 100} %")
print(f"TAR@FAR=10^-4 for CL2CL: {tar_far_104 * 100} %")

print(f"R@1 for CL2CL: {compute_recall_at_k(scores_mat, cl_label, cl_label, 1) * 100} %")
print(f"R@10 for CL2CL: {compute_recall_at_k(scores_mat, cl_label, cl_label, 10) * 100} %")
print(f"R@50 for CL2CL: {compute_recall_at_k(scores_mat, cl_label, cl_label, 50) * 100} %")
print(f"R@100 for CL2CL: {compute_recall_at_k(scores_mat, cl_label, cl_label, 100) * 100} %")
