import torch
from datasets.rb_loader import RB_loader
from utils import Prev_RetMetric, l2_norm, compute_recall_at_k
import numpy as np
from tqdm import tqdm
from model import SwinModel_domain_agnostic as Model
from sklearn.metrics import roc_curve, auc
import json
import torch.nn.functional as F

if __name__ == '__main__':
    device = torch.device('cuda')
    data = RB_loader(split = 'test')
    dataloader = torch.utils.data.DataLoader(data,batch_size = 16, num_workers = 1, pin_memory = True)
    model = Model().to(device)
    checkpoint = torch.load("ridgeformer_checkpoints/phase1_scratch.pt",map_location = torch.device('cpu'))
    model.load_state_dict(checkpoint,strict=False)

    model.eval()
    cl_feats, cb_feats, cl_labels, cb_labels, cl_fnames, cb_fnames, cl_feats_unnormed, cb_feats_unnormed = list(),list(),list(),list(),list(),list(),list(),list()
    print("Computing Test Recall")
    with torch.no_grad():
        for (x_cl, x_cb, target, cl_fname, cb_fname) in tqdm(dataloader):
            x_cl, x_cb, target = x_cl.to(device), x_cb.to(device), target.to(device)
            x_cl, _ = model.get_embeddings(x_cl, ftype="contactless")
            x_cb, _ = model.get_embeddings(x_cb, ftype="contactbased")
            cl_feats_unnormed.append(x_cl.cpu().detach().numpy())
            cb_feats_unnormed.append(x_cb.cpu().detach().numpy())
            x_cl = l2_norm(x_cl).cpu().detach().numpy()
            x_cb = l2_norm(x_cb).cpu().detach().numpy()
            target = target.cpu().detach().numpy()
            cl_feats.append(x_cl)
            cb_feats.append(x_cb)
            cl_labels.append(target)
            cb_labels.append(target)
            cl_fnames.extend(cl_fname)
            cb_fnames.extend(cb_fname)

    cl_feats = torch.from_numpy(np.concatenate(cl_feats))
    cb_feats = torch.from_numpy(np.concatenate(cb_feats))
    cl_labels = torch.from_numpy(np.concatenate(cl_labels))
    cb_labels = torch.from_numpy(np.concatenate(cb_labels))
    cl_feats_unnormed = torch.from_numpy(np.concatenate(cl_feats_unnormed))
    cb_feats_unnormed = torch.from_numpy(np.concatenate(cb_feats_unnormed))

    unique_labels, indices  = torch.unique(cb_labels, return_inverse=True)
    unique_feats            = torch.stack([cb_feats[indices == i].mean(dim=0) for i in range(len(unique_labels))])
    cb_feats                = unique_feats
    unique_labels, indices  = torch.unique(cb_labels, return_inverse=True)
    unique_feats            = torch.stack([cb_feats_unnormed[indices == i].mean(dim=0) for i in range(len(unique_labels))])
    cb_labels               = unique_labels
    cb_feats_unnormed       = unique_feats

    # CL2CB <---------------------------------------->
    cl_feats  = cl_feats.numpy()
    cb_feats  = cb_feats.numpy()
    cb_feats_unnormed = cb_feats_unnormed.numpy()
    cl_feats_unnormed = cl_feats_unnormed.numpy()

    squared_diff = np.sum(np.square(cl_feats_unnormed[:, np.newaxis] - cb_feats_unnormed), axis=2)
    distance     = -1 * np.sqrt(squared_diff)
    similarities = np.dot(cl_feats,np.transpose(cb_feats))
    scores_mat = similarities + 0.1 * distance
    scores = scores_mat.flatten().tolist()

    ids = torch.eq(cl_labels.view(-1,1)-cb_labels.view(1,-1),0.0).flatten().tolist()
    ids_mod = list()
    for x in ids:
        if x==True:
            ids_mod.append(1)
        else:
            ids_mod.append(0)
    fpr,tpr,thresh = roc_curve(ids_mod,scores,drop_intermediate=True)
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
    cl_labels = cl_labels.cpu().detach()
    cb_labels = cb_labels.cpu().detach()
    print(f"TAR@FAR=10^-2 for CB2CL: {tar_far_102 * 100} %")
    print(f"TAR@FAR=10^-3 for CB2CL: {tar_far_103 * 100} %")
    print(f"TAR@FAR=10^-4 for CB2CL: {tar_far_104 * 100} %")
    print(f"R@1 for CB2CL: {compute_recall_at_k(torch.from_numpy(scores_mat), cl_labels, cb_labels, 1) * 100} %")
    print(f"R@10 for CB2CL: {compute_recall_at_k(torch.from_numpy(scores_mat), cl_labels, cb_labels, 10) * 100} %")
    print(f"R@50 for CB2CL: {compute_recall_at_k(torch.from_numpy(scores_mat), cl_labels, cb_labels, 50) * 100} %")
    print(f"R@100 for CB2CL: {compute_recall_at_k(torch.from_numpy(scores_mat), cl_labels, cb_labels, 100) * 100} %")

    ################################################################################

    # CL2CL
    scores = torch.from_numpy(np.dot(cl_feats,np.transpose(cl_feats)))
    row, col = torch.triu_indices(row=scores.size(0), col=scores.size(1), offset=1)
    scores = scores[row, col]
    scores = scores.numpy().flatten().tolist()
    labels = torch.eq(cl_labels.view(-1,1) - cl_labels.view(1,-1),0.0).float().cuda()
    labels = labels[torch.triu(torch.ones(labels.shape),diagonal = 1) == 1].tolist()
    fpr,tpr,_ = roc_curve(labels,scores)
    lower_fpr_idx = max(i for i, val in enumerate(fpr) if val < 0.01)
    upper_fpr_idx = min(i for i, val in enumerate(fpr) if val >= 0.01)
    tar_far_102 = (tpr[lower_fpr_idx]+tpr[upper_fpr_idx])/2
    lower_fpr_idx = max(i for i, val in enumerate(fpr) if val < 0.001)
    upper_fpr_idx = min(i for i, val in enumerate(fpr) if val >= 0.001)
    tar_far_103 = (tpr[lower_fpr_idx]+tpr[upper_fpr_idx])/2
    lower_fpr_idx = max(i for i, val in enumerate(fpr) if val < 0.0001)
    upper_fpr_idx = min(i for i, val in enumerate(fpr) if val >= 0.0001)
    tar_far_104 = (tpr[lower_fpr_idx]+tpr[upper_fpr_idx])/2
    clcltf102 = tar_far_102 * 100
    clcltf103 = tar_far_103 * 100
    clcltf104 = tar_far_104 * 100
    fnr = 1 - tpr
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    roc_auc = auc(fpr, tpr)
    print(f"ROCAUC for CL2CL: {roc_auc * 100} %")
    print(f"EER for CL2CL: {EER * 100} %")
    eer_cl2cl = EER * 100
    print(f"TAR@FAR=10^-2 for CL2CL: {tar_far_102 * 100} %")
    print(f"TAR@FAR=10^-3 for CL2CL: {tar_far_103 * 100} %")
    print(f"TAR@FAR=10^-4 for CL2CL: {tar_far_104 * 100} %")
    cl_labels = cl_labels.cpu().detach().numpy()
    recall_score = Prev_RetMetric([cl_feats,cl_feats],[cl_labels,cl_labels],cl2cl = True)
    cl2clk1 = recall_score.recall_k(k=1) * 100
    print(f"R@1 for CL2CL: {recall_score.recall_k(k=1) * 100} %")
    print(f"R@10 for CL2CL: {recall_score.recall_k(k=10) * 100} %")
    print(f"R@50 for CL2CL: {recall_score.recall_k(k=50) * 100} %")
    print(f"R@100 for CL2CL: {recall_score.recall_k(k=100) * 100} %")