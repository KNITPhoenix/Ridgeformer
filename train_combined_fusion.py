from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from datasets.hkpoly_test import hktest
from datasets.original_combined_train import Combined_original
from datasets.rb_loader import RB_loader
from loss import DualMSLoss_FineGrained_domain_agnostic_ft, DualMSLoss_FineGrained, DualMSLoss_FineGrained_domain_agnostic
import timm
from utils import Prev_RetMetric, RetMetric, compute_recall_at_k, l2_norm, compute_sharded_cosine_similarity, count_parameters
from pprint import pprint
import numpy as np
from tqdm import tqdm
from combined_sampler import BalancedSampler
from torch.utils.data.sampler import BatchSampler
from torch.nn.parallel import DataParallel
from model import SwinModel_Fusion as Model
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import json
from torch.utils.tensorboard import SummaryWriter

def train(args, model, device, train_loader, test_loader, optimizers, epoch, loss_func, pl_arg, stepping, log_writer, checkpoint_save_path):
    model.train()
    steploss = list()
    for batch_idx, (x_cl, x_cb, target,_,_) in enumerate(pbar := tqdm(train_loader)):
        x_cl, x_cb, target = x_cl.to(device), x_cb.to(device), target.to(device)
        for optimizer in optimizers:
            optimizer.zero_grad()
        x_cl_tokens, x_cb_tokens = model(x_cl, x_cb)

        N, M, D = x_cl_tokens.shape

        index_i = torch.arange(N).unsqueeze(1)  # Shape: (100, 1)
        index_j = torch.arange(N).unsqueeze(0)  # Shape: (1, 100)

        x = x_cl_tokens[index_i]  # Shape: (100, 100, 197, 1024)
        y = x_cb_tokens[index_j]  # Shape: (100, 100, 197, 1024)

        x = x.expand(N, N, M, D).reshape(N * N, M, D)  # Shape: (10000, 197, 1024)
        y = y.expand(N, N, M, D).reshape(N * N, M, D)  # Shape: (10000, 197, 1024)
        sim_matrix,_ = model.combine_features(x, y)
        sim_matrix = sim_matrix.view(N, N).to(device)

        loss = loss_func.ms_sample(sim_matrix, target).cuda() + loss_func.ms_sample(sim_matrix.t(), target.t()).cuda()
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
        if batch_idx % args.log_interval == 0:
            if args.dry_run:
                break
        pbar.set_description(f"Loss {loss}")
        steploss.append(loss)
        if (batch_idx + 1)%50 == 0:
            cl2clk1,cl2cbk1,eer_cb2cl,eer_cl2cl,cbcltf102,cbcltf103,cbcltf104,clcltf102,clcltf103,clcltf104 = hkpoly_test_fn(model, device, test_loader, epoch, pl_arg)
            log_writer.add_scalars('recall@1/step',{'CL2CL':cl2clk1,'CB2CL':cl2cbk1},stepping)
            log_writer.add_scalars('EER/step',{'CL2CL':eer_cl2cl,'CB2CL':eer_cb2cl},stepping)
            log_writer.add_scalars('TARFAR10^-2/step',{'CL2CL':clcltf102,'CB2CL':cbcltf102},stepping)
            log_writer.add_scalars('TARFAR10^-4/step',{'CL2CL':clcltf104,'CB2CL':cbcltf104},stepping)
            stepping+=1
            
    return sum(steploss)/len(steploss), stepping

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output

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
            scores = scores.reshape(shard_size, shard_size)
            similarity_matrix[i_start:i_end, j_start:j_end] = scores.cpu().detach()
    return similarity_matrix

def hkpoly_test_fn(model,device,test_loader,epoch,plot_argument):
    model.eval()
    cl_feats, cb_feats, cl_labels, cb_labels = list(),list(),list(),list()
    with torch.no_grad():
        for (x_cl, x_cb, label) in tqdm(test_loader):
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
    np.save("combined_models_scores/task1_cb2cl_score_matrix_"+str(epoch)+"_"+plot_argument[0]+"_"+plot_argument[1]+"_"+plot_argument[2]+".npy", scores_mat)
    scores = scores_mat.cpu().detach().numpy().flatten().tolist()
    labels = torch.eq(cb_label.view(-1,1) - cl_label.view(1,-1),0.0).flatten().tolist()
    ids_mod = list()
    for i in labels:
        if i==True:
            ids_mod.append(1)
        else:
            ids_mod.append(0)
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
    fnr = 1 - tpr
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve CB2CL task1')
    plt.legend(loc="lower right")
    plt.savefig("combined_models_scores/roc_curve_cb2cl_task1_"+"_"+plot_argument[0]+"_"+plot_argument[1]+"_"+plot_argument[2]+str(epoch)+".png", dpi=300, bbox_inches='tight')
    print(f"ROCAUC for CB2CL: {roc_auc * 100} %")   
    print(f"EER for CB2CL: {EER * 100} %")
    eer_cb2cl = EER * 100
    print(f"TAR@FAR=10^-2 for CB2CL: {tar_far_102 * 100} %")
    print(f"TAR@FAR=10^-3 for CB2CL: {tar_far_103 * 100} %")
    print(f"TAR@FAR=10^-4 for CB2CL: {tar_far_104 * 100} %")
    cbcltf102 = tar_far_102 * 100
    cbcltf103 = tar_far_103 * 100
    cbcltf104 = tar_far_104 * 100
    cl2cbk1   = compute_recall_at_k(scores_mat, cl_label, cb_label, 1) * 100
    print(f"R@1 for CB2CL: {compute_recall_at_k(scores_mat, cl_label, cb_label, 1) * 100} %")
    print(f"R@10 for CB2CL: {compute_recall_at_k(scores_mat, cl_label, cb_label, 10) * 100} %")
    print(f"R@50 for CB2CL: {compute_recall_at_k(scores_mat, cl_label, cb_label, 50) * 100} %")
    print(f"R@100 for CB2CL: {compute_recall_at_k(scores_mat, cl_label, cb_label, 100) * 100} %")
    torch.cuda.empty_cache()

    return cl2cbk1,eer_cb2cl,cbcltf102,cbcltf103,cbcltf104

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--manifest-list', type=list, default=mani_lst,
                        help='list of manifest files')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr_fusion', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.9, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--warmup', type=int, default=2, metavar='N',
                        help='warm up rate for feature extractor')
    parser.add_argument('--model-name', type=str, default="swinmodel",
                        help='Name of the model for checkpointing')
    args = parser.parse_args()

    device = torch.device("cuda")
    model = Model().to(device)
    ckpt_combined_phase1_ft = "ridgeformer_checkpoints/combined_models_check/phase1_ft_hkpoly.pt"
    ckpt_combined_phase2    = "ridgeformer_checkpoints/phase2_scratch.pt"

    model.load_pretrained_models(ckpt_combined_phase1_ft, ckpt_combined_phase2)
    model.freeze_backbone()
    checkpoint_save_path = "ridgeformer_checkpoints/"
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    if not os.path.exists("experiment_logs/"+args.model_name):
        os.mkdir("experiment_logs/"+args.model_name)

    log_writer = SummaryWriter("experiment_logs/"+args.model_name+"/",comment = str(args.batch_size)+str(args.lr_fusion))

    torch.manual_seed(args.seed)

    print("loading Normal RGB images -----------------------------")
    train_dataset    = Combined_original(args.manifest_list,split="train")
    val_dataset      = hktest(split="test")
    
    balanced_sampler = BalancedSampler(train_dataset, batch_size = args.batch_size, images_per_class = 2)
    batch_sampler    = BatchSampler(balanced_sampler, batch_size = args.batch_size, drop_last = True)
    
    train_kwargs     = {'batch_sampler': batch_sampler}
    test_kwargs      = {'batch_size':    args.test_batch_size}
    
    if use_cuda:
        cuda_kwargs = {
                       'num_workers': 1,
                       'pin_memory': True
                       }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(val_dataset, **test_kwargs)
    
    print("Number of Trainable Parameters: - ", count_parameters(model))

    loss_func           = DualMSLoss_FineGrained()
    optimizer_fusion    = optim.AdamW(
        [
            {"params": model.output_logit_mlp.parameters(), "lr":args.lr_fusion},
            {"params": model.fusion.parameters(),           "lr":args.lr_fusion},
            {"params": model.sep_token,                     "lr":args.lr_fusion},
            {"params": model.encoder_layer.parameters(),    "lr":args.lr_fusion},

         ],
        weight_decay=0.000001,
        lr=args.lr_fusion)

    scheduler = MultiStepLR(optimizer_fusion, milestones = [3,6,9,14], gamma=0.5)

    cl2cl_lst,cb2cl_lst,eer_cl2cl_lst,eer_cb2cl_lst,cbcltf102_lst,cbcltf103_lst,cbcltf104_lst,clcltf102_lst,clcltf103_lst,clcltf104_lst = list(),list(),list(),list(),list(),list(),list(),list(),list(),list()
    stepping = 1
    for epoch in range(1, args.epochs + 1):
        print(f"running epoch------ {epoch}")            
        avg_step_loss,stepping = train(args, model, device, train_loader, test_loader, [optimizer_fusion], epoch, loss_func, [args.model_name,str(args.batch_size),str(args.lr_fusion)],stepping,log_writer, checkpoint_save_path)
        
        print(f"Learning Rate for {epoch} for linear = {scheduler.get_last_lr()}")
        print(f"Learning Rate for {epoch} for swin = {scheduler.get_last_lr()}")
        
        log_writer.add_scalar('Liner_LR/epoch',scheduler.get_last_lr()[0],epoch)
        log_writer.add_scalar('Swin_LR/epoch',scheduler.get_last_lr()[0],epoch)

        scheduler.step()
        
        cl2clk1,cl2cbk1,eer_cb2cl,eer_cl2cl,cbcltf102,cbcltf103,cbcltf104,clcltf102,clcltf103,clcltf104 = hkpoly_test_fn(model, device, test_loader, epoch, [args.model_name,str(args.batch_size),str(args.lr_fusion)])
        cl2cl_lst.append(cl2clk1)
        cb2cl_lst.append(cl2cbk1)
        eer_cl2cl_lst.append(eer_cl2cl)
        eer_cb2cl_lst.append(eer_cb2cl)
        cbcltf102_lst.append(cbcltf102)
        cbcltf103_lst.append(cbcltf103)
        cbcltf104_lst.append(cbcltf104)
        clcltf102_lst.append(clcltf102)
        clcltf103_lst.append(clcltf103)
        clcltf104_lst.append(clcltf104)

        log_writer.add_scalars('recall@1/epoch',{'CL2CL':cl2clk1,'CB2CL':cl2cbk1},epoch)
        log_writer.add_scalars('EER/epoch',{'CL2CL':eer_cl2cl,'CB2CL':eer_cb2cl},epoch)
        log_writer.add_scalars('TARFAR10^-2/epoch',{'CL2CL':clcltf102,'CB2CL':cbcltf102},epoch)
        log_writer.add_scalars('TARFAR10^-4/epoch',{'CL2CL':clcltf104,'CB2CL':cbcltf104},epoch)
        log_writer.add_scalar('AvgLoss/epoch',avg_step_loss,epoch)

        torch.save(model.state_dict(), checkpoint_save_path + "combinedtrained_hkpolytest_" + args.model_name + "_" + str(args.lr_fusion) + "_" + str(args.batch_size) + str(epoch) + "_" + str(cl2clk1)+ "_" + str(cl2cbk1) + ".pt")
    log_writer.close()

    print(f"Maximum recall@1 for CL2CL: {max(cl2cl_lst)} at epoch {cl2cl_lst.index(max(cl2cl_lst))+1}")
    print(f"Maximum recall@1 for CB2CL: {max(cb2cl_lst)} at epoch {cb2cl_lst.index(max(cb2cl_lst))+1}")
    print(f"Minimum EER for CL2CL: {min(eer_cl2cl_lst)} at epoch {eer_cl2cl_lst.index(min(eer_cl2cl_lst))+1}")
    print(f"Minimum EER for CB2CL: {min(eer_cb2cl_lst)} at epoch {eer_cb2cl_lst.index(min(eer_cb2cl_lst))+1}")
    print(f"Maximum TAR@FAR=10^-2 for CB2CL: {max(cbcltf102_lst)} at epoch {cbcltf102_lst.index(max(cbcltf102_lst))+1}")
    print(f"Maximum TAR@FAR=10^-3 for CB2CL: {max(cbcltf103_lst)} at epoch {cbcltf103_lst.index(max(cbcltf103_lst))+1}")
    print(f"Maximum TAR@FAR=10^-4 for CB2CL: {max(cbcltf104_lst)} at epoch {cbcltf104_lst.index(max(cbcltf104_lst))+1}")
    print(f"Maximum TAR@FAR=10^-2 for CL2CL: {max(clcltf102_lst)} at epoch {clcltf102_lst.index(max(clcltf102_lst))+1}")
    print(f"Maximum TAR@FAR=10^-3 for CL2CL: {max(clcltf103_lst)} at epoch {clcltf103_lst.index(max(clcltf103_lst))+1}")
    print(f"Maximum TAR@FAR=10^-4 for CL2CL: {max(clcltf104_lst)} at epoch {clcltf104_lst.index(max(clcltf104_lst))+1}")

if __name__ == '__main__':
    main()
