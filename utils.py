import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

class RetMetric(object):
    def __init__(self, sim_mat, labels):
        self.gallery_labels, self.query_labels = labels
        self.sim_mat = sim_mat
        self.is_equal_query = False

    def recall_k(self, k=1):
        m = len(self.sim_mat)

        match_counter = 0

        for i in range(m):
            pos_sim = self.sim_mat[i][self.gallery_labels == self.query_labels[i]]
            neg_sim = self.sim_mat[i][self.gallery_labels != self.query_labels[i]]
            
            thresh = np.sort(pos_sim)[-2] if self.is_equal_query and len(pos_sim) > 1 else np.max(pos_sim)

            if np.sum(neg_sim > thresh) < k:
                match_counter += 1
        return float(match_counter) / m

class Prev_RetMetric(object):
    def __init__(self, feats, labels, cl2cl=True):

        if len(feats) == 2 and type(feats) == list:
            """
            feats = [gallery_feats, query_feats]
            labels = [gallery_labels, query_labels]
            """
            self.is_equal_query = False

            self.gallery_feats, self.query_feats = feats
            self.gallery_labels, self.query_labels = labels

        else:
            self.is_equal_query = True
            self.gallery_feats = self.query_feats = feats
            self.gallery_labels = self.query_labels = labels

        self.sim_mat = np.matmul(self.query_feats, np.transpose(self.gallery_feats))
        if cl2cl:
            self.sim_mat = self.sim_mat * (1 - np.identity(self.sim_mat.shape[0]))
        
    def recall_k(self, k=1):
        m = len(self.sim_mat)

        match_counter = 0

        for i in range(m):
            pos_sim = self.sim_mat[i][self.gallery_labels == self.query_labels[i]]
            neg_sim = self.sim_mat[i][self.gallery_labels != self.query_labels[i]]

            thresh = np.sort(pos_sim)[-2] if self.is_equal_query else np.max(pos_sim)

            if np.sum(neg_sim > thresh) < k:
                match_counter += 1
        return float(match_counter) / m
    
def compute_recall_at_k(similarity_matrix, p_labels, g_labels, k):
    num_probes = p_labels.size(0)
    recall_at_k = 0.0
    for i in range(num_probes):
        probe_label = p_labels[i]
        sim_scores = similarity_matrix[i]
        sorted_indices = torch.argsort(sim_scores, descending=True)
        top_k_indices = sorted_indices[:k]
        correct_in_top_k = any(g_labels[idx] == probe_label for idx in top_k_indices)
        recall_at_k += correct_in_top_k
    recall_at_k /= num_probes
    return recall_at_k

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    
    return output

def compute_sharded_cosine_similarity(tensor1, tensor2, shard_size):
    B, T, D = tensor1.shape
    average_sim_matrix = torch.zeros((B, B), device=tensor1.device)

    for start_idx1 in tqdm(range(0, B, shard_size)):
        end_idx1 = min(start_idx1 + shard_size, B)

        for start_idx2 in range(0, B, shard_size):
            end_idx2 = min(start_idx2 + shard_size, B)

            # Get the shard
            shard_tensor1 = tensor1[start_idx1:end_idx1]
            shard_tensor2 = tensor2[start_idx2:end_idx2]

            # Reshape and expand
            shard_tensor1_expanded = shard_tensor1.unsqueeze(1).unsqueeze(3)
            shard_tensor2_expanded = shard_tensor2.unsqueeze(0).unsqueeze(2)

            # Compute cosine similarity for the shard
            shard_cos_sim = F.cosine_similarity(shard_tensor1_expanded, shard_tensor2_expanded, dim=-1)

            # Sum up the cosine similarities
            average_sim_matrix[start_idx1:end_idx1, start_idx2:end_idx2] += torch.sum(shard_cos_sim, dim=[2, 3])

    # Normalize by the total number of elements (T*T)
    average_sim_matrix /= (T * T)

    return average_sim_matrix