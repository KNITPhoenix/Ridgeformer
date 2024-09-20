# script to evaluate HKPolyU testing dataset on finetuned model after phase 2
import torch
from datasets.hkpoly_test import hktest
from utils import Prev_RetMetric, l2_norm, compute_recall_at_k
import numpy as np
from tqdm import tqdm
from model import SwinModel_Fusion as Model
from sklearn.metrics import roc_curve, auc
import json

def calculate_tar_at_far(fpr, tpr, target_fars):
    tar_at_far = {}
    for far in target_fars:
        if far in fpr:
            tar = tpr[np.where(fpr == far)][0]
        else:
            tar = np.interp(far, fpr, tpr)
        tar_at_far[far] = tar
    return tar_at_far

def get_fused_cross_score_matrix(model, cl_tokens, cb_tokens):
    cl_tokens   = torch.cat(cl_tokens)
    cb_tokens   = torch.cat(cb_tokens)
    batch_size  = cl_tokens.shape[0]
    shard_size  = 20
    similarity_matrix = torch.zeros((batch_size, batch_size))
    for i_start in tqdm(range(0, batch_size, shard_size)):
        i_end   = min(i_start + shard_size, batch_size)
        shard_i = cl_tokens[i_start:i_end]
        for j_start in range(0, batch_size, shard_size):
            j_end               = min(j_start + shard_size, batch_size)
            shard_j             = cb_tokens[j_start:j_end]
            batch_i             = shard_i.unsqueeze(1)
            batch_j             = shard_j.unsqueeze(0)
            pairwise_i          = batch_i.expand(-1, shard_size, -1, -1)
            pairwise_j          = batch_j.expand(shard_size, -1, -1, -1)
            similarity_scores, distances   = model.combine_features(pairwise_i.reshape(-1, 197, 1024), pairwise_j.reshape(-1, 197, 1024))
            scores = similarity_scores - 0.1 * distances
            scores   = scores.reshape(shard_size, shard_size)
            similarity_matrix[i_start:i_end, j_start:j_end] = scores.cpu().detach()
    return similarity_matrix

if __name__ == '__main__':
    device = torch.device('cuda')
    data = hktest(split = 'test')
    dataloader = torch.utils.data.DataLoader(data,batch_size = 16, num_workers = 1, pin_memory = True)
    model = Model().to(device)
    checkpoint = torch.load("ridgeformer_checkpoints/phase2_ft_hkpoly.pt",map_location = torch.device('cpu'))
    model.load_state_dict(checkpoint,strict=False)
    model.eval()

    cl_feats, cb_feats, cl_labels, cb_labels, cl_feats_unnormed, cb_feats_unnormed = list(),list(),list(),list(),list(),list()
    with torch.no_grad():
        for (x_cl, x_cb, label) in tqdm(dataloader):
            x_cl, x_cb, label = x_cl.to(device), x_cb.to(device), label.to(device)
            x_cl_token  = model.get_tokens(x_cl,'contactless')
            x_cb_token  = model.get_tokens(x_cb,'contactbased')
            label = label.cpu().detach().numpy()
            cl_feats.append(x_cl_token)
            cb_feats.append(x_cb_token)
            cl_labels.append(label)
            cb_labels.append(label)

    cl_label = torch.from_numpy(np.concatenate(cl_labels))
    cb_label = torch.from_numpy(np.concatenate(cb_labels))

    # CB2CL
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
    tar_far_102 = tpr[upper_fpr_idx]

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

    recall_dict = dict()
    for i in range(1,101):
        recall_dict[i] = compute_recall_at_k(scores_mat, cl_label, cb_label, i)

    print(f"R@1 for CB2CL: {compute_recall_at_k(scores_mat, cl_label, cb_label, 1) * 100} %")
    print(f"R@10 for CB2CL: {compute_recall_at_k(scores_mat, cl_label, cb_label, 10) * 100} %")
    print(f"R@50 for CB2CL: {compute_recall_at_k(scores_mat, cl_label, cb_label, 50) * 100} %")
    print(f"R@100 for CB2CL: {compute_recall_at_k(scores_mat, cl_label, cb_label, 100) * 100} %")
