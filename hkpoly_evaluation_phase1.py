# script to evaluated HKPolyU testing dataset on finetuned model after phase 1
import torch
from datasets.hkpoly_test import hktest
from utils import Prev_RetMetric, l2_norm, compute_recall_at_k
import numpy as np
from tqdm import tqdm
from model import SwinModel_domain_agnostic as Model
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

if __name__ == '__main__':
    device = torch.device('cuda')
    data = hktest(split = 'test')
    dataloader = torch.utils.data.DataLoader(data,batch_size = 16, num_workers = 1, pin_memory = True)
    model = Model().to(device)
    checkpoint = torch.load("ridgeformer_checkpoints/phase1_ft_hkpoly.pt",map_location = torch.device('cpu'))
    model.load_state_dict(checkpoint,strict=False)
    model.eval()

    cl_feats, cb_feats, cl_labels, cb_labels, cl_feats_unnormed, cb_feats_unnormed = list(),list(),list(),list(),list(),list()
    with torch.no_grad():
        for (x_cl, x_cb, label) in tqdm(dataloader):
            x_cl, x_cb, label = x_cl.to(device), x_cb.to(device), label.to(device)
            x_cl_feat, x_cl_token = model.get_embeddings(x_cl,'contactless')
            x_cb_feat,x_cb_token = model.get_embeddings(x_cb,'contactbased')
            cl_feats_unnormed.append(x_cl_feat.cpu().detach().numpy())
            cb_feats_unnormed.append(x_cb_feat.cpu().detach().numpy())
            x_cl_feat = l2_norm(x_cl_feat).cpu().detach().numpy()
            x_cb_feat = l2_norm(x_cb_feat).cpu().detach().numpy()
            label = label.cpu().detach().numpy()
            cl_feats.append(x_cl_feat)
            cb_feats.append(x_cb_feat)
            cl_labels.append(label)
            cb_labels.append(label)

    cl_feats = np.concatenate(cl_feats)
    cb_feats = np.concatenate(cb_feats)
    cl_feats_unnormed = np.concatenate(cl_feats_unnormed)
    cb_feats_unnormed = np.concatenate(cb_feats_unnormed)
    cl_label = torch.from_numpy(np.concatenate(cl_labels))
    cb_label = torch.from_numpy(np.concatenate(cb_labels))

    # CB2CL
    squared_diff = np.sum(np.square(cl_feats_unnormed[:, np.newaxis] - cb_feats_unnormed), axis=2)
    distance     = -1 * np.sqrt(squared_diff)
    similarities = np.dot(cl_feats,np.transpose(cb_feats))
    scores_mat = similarities + 0.1 * distance

    scores = scores_mat.flatten().tolist()
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
    print(tpr[lower_fpr_idx], lower_fpr_idx, fpr[lower_fpr_idx], thresh[lower_fpr_idx])
    print(tpr[upper_fpr_idx], upper_fpr_idx, fpr[upper_fpr_idx], thresh[upper_fpr_idx])

    lower_fpr_idx = max(i for i, val in enumerate(fpr) if val < 0.001)
    upper_fpr_idx = min(i for i, val in enumerate(fpr) if val >= 0.001)
    tar_far_103 = (tpr[lower_fpr_idx]+tpr[upper_fpr_idx])/2
    print(tpr[lower_fpr_idx], lower_fpr_idx, fpr[lower_fpr_idx])
    print(tpr[upper_fpr_idx], upper_fpr_idx, fpr[upper_fpr_idx])

    lower_fpr_idx = max(i for i, val in enumerate(fpr) if val < 0.0001)
    upper_fpr_idx = min(i for i, val in enumerate(fpr) if val >= 0.0001)
    tar_far_104 = (tpr[lower_fpr_idx]+tpr[upper_fpr_idx])/2
    print(tpr[lower_fpr_idx], lower_fpr_idx, fpr[lower_fpr_idx])
    print(tpr[upper_fpr_idx], upper_fpr_idx, fpr[upper_fpr_idx])

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
    
    print(f"R@1 for CB2CL: {compute_recall_at_k(torch.from_numpy(scores_mat), cl_label, cb_label, 1) * 100} %")
    print(f"R@10 for CB2CL: {compute_recall_at_k(torch.from_numpy(scores_mat), cl_label, cb_label, 10) * 100} %")
    print(f"R@50 for CB2CL: {compute_recall_at_k(torch.from_numpy(scores_mat), cl_label, cb_label, 50) * 100} %")
    print(f"R@100 for CB2CL: {compute_recall_at_k(torch.from_numpy(scores_mat), cl_label, cb_label, 100) * 100} %")