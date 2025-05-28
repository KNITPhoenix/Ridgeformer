from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torchvision.models as models
import timm
from pprint import pprint
import numpy as np
from tqdm import tqdm
from torch.utils.data.sampler import BatchSampler
from gradient_reversal.module import GradientReversal

class SwinModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.swin_cl = timm.create_model('vit_large_patch16_224_in21k', pretrained=True, num_classes=0)
        self.swin_cb = self.swin_cl
        
        self.linear_cl = nn.Sequential(nn.Linear(1024, 1024),
                                       nn.ReLU(),
                                       nn.Linear(1024, 1024))
        self.linear_cb = nn.Linear(1024, 1024)

    def freeze_encoder(self):
        for param in self.swin_cl.parameters():
            param.requires_grad = False
        for param in self.swin_cb.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.swin_cl.parameters():
            param.requires_grad = True
        for param in self.swin_cb.parameters():
            param.requires_grad = True

    def get_embeddings(self, image, ftype):
        linear = self.linear_cl if ftype == "contactless" else self.linear_cl
        swin   = self.swin_cl   if ftype == "contactless" else self.swin_cb
        
        tokens = swin(image)
        emb_mean = tokens.mean(dim=1)
        feat = linear(emb_mean)
        tokens_transformed = linear(tokens)
        return feat, tokens

    def forward(self, x_cl, x_cb):
        x_cl_tokens = self.swin_cl(x_cl)
        x_cb_tokens = self.swin_cb(x_cb)

        x_cl_mean = x_cl_tokens.mean(dim=1)
        x_cb_mean = x_cb_tokens.mean(dim=1)

        x_cl = self.linear_cl(x_cl_mean)
        x_cl_tokens_transformed = self.linear_cl(x_cl_tokens)

        x_cb = self.linear_cl(x_cb_mean)
        x_cb_tokens_transformed = self.linear_cl(x_cb_tokens)

        return x_cl, x_cb, x_cl_tokens, x_cb_tokens

class SwinModel_domain_agnostic(nn.Module):
    def __init__(self):
        super().__init__()
        self.swin_cl = timm.create_model('vit_large_patch16_224_in21k', pretrained=True, num_classes=0)
        self.swin_cb = self.swin_cl #timm.create_model('vit_large_patch16_224_in21k', pretrained=True, num_classes=0)
        
        self.linear_cl = nn.Sequential(nn.Linear(1024, 1024),
                                       nn.ReLU(),
                                       nn.Linear(1024, 1024))
        self.linear_cb = nn.Linear(1024, 1024)
        self.classify = nn.Sequential(GradientReversal(alpha=0.6),  # original 0.8
                                      nn.Linear(1024,512),
                                      nn.ReLU(),
                                      nn.Linear(512,8))

    def freeze_encoder(self):
        for param in self.swin_cl.parameters():
            param.requires_grad = False
        for param in self.swin_cb.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.swin_cl.parameters():
            param.requires_grad = True
        for param in self.swin_cb.parameters():
            param.requires_grad = True

    def get_embeddings(self, image, ftype):
        linear = self.linear_cl if ftype == "contactless" else self.linear_cl
        swin   = self.swin_cl   if ftype == "contactless" else self.swin_cb
        
        tokens = swin(image)
        emb_mean = tokens.mean(dim=1)
        feat = linear(emb_mean)
        tokens_transformed = linear(tokens)
        return feat, tokens

    def forward(self, x_cl, x_cb):
        x_cl_tokens = self.swin_cl(x_cl)
        x_cb_tokens = self.swin_cb(x_cb)

        x_cl_mean = x_cl_tokens.mean(dim=1)
        x_cb_mean = x_cb_tokens.mean(dim=1)

        x_cl = self.linear_cl(x_cl_mean)
        x_cl_tokens_transformed = self.linear_cl(x_cl_tokens)

        x_cb = self.linear_cl(x_cb_mean)
        x_cb_tokens_transformed = self.linear_cl(x_cb_tokens)

        domain_class_cl = self.classify(x_cl_mean)
        domain_class_cb = self.classify(x_cb_mean)

        return x_cl, x_cb, x_cl_tokens, x_cb_tokens, domain_class_cl, domain_class_cb

class SwinModel_Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_dim        = 1024
        self.swin_cl            = timm.create_model('vit_large_patch16_224_in21k', pretrained=True, num_classes=0)  
        self.encoder_layer      = nn.TransformerEncoderLayer(d_model=self.feature_dim, nhead=4, dropout=0.5, batch_first=True, norm_first=True, activation="gelu")
        self.fusion             = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.sep_token          = nn.Parameter(torch.randn(1, 1, self.feature_dim))
        self.output_logit_mlp   = nn.Sequential(nn.Linear(1024, 512),
                                       nn.ReLU(),
                                       nn.Dropout(),
                                       nn.Linear(512, 1))
        self.linear_cl          = nn.Sequential(nn.Linear(1024, 1024),
                                       nn.ReLU(),
                                       nn.Linear(1024, 1024))
        
    def load_pretrained_models(self, swin_cl_path, fusion_ckpt_path):
        swin_cl_state_dict = torch.load(swin_cl_path)
        new_dict = {}
        for key in swin_cl_state_dict.keys():
            if "swin_cl" in key:
                new_dict[key.replace("swin_cl.","")] = swin_cl_state_dict[key] 
        self.swin_cl.load_state_dict(new_dict)

        fusion_params = torch.load(fusion_ckpt_path)
        new_dict = {}
        for key in fusion_params.keys():
            if "encoder_layer" in key:
                new_dict[key.replace("encoder_layer.","")] = fusion_params[key] 
        self.encoder_layer.load_state_dict(new_dict)
        
        new_dict = {}
        for key in fusion_params.keys():
            if "fusion" in key:
                new_dict[key.replace("fusion.","")] = fusion_params[key] 
        self.fusion.load_state_dict(new_dict)
        
        self.sep_token = nn.Parameter(fusion_params["sep_token"])   

        new_dict = {}
        for key in fusion_params.keys():
            if "output_logit_mlp" in key:
                new_dict[key.replace("output_logit_mlp.","")] = fusion_params[key] 
        self.output_logit_mlp.load_state_dict(new_dict)

    def l2_norm(self,input):
        input_size = input.shape[0]
        buffer     = torch.pow(input, 2)
        normp      = torch.sum(buffer, 1).add_(1e-12)
        norm       = torch.sqrt(normp)
        _output    = torch.div(input, norm.view(-1, 1).expand_as(input))
        return _output

    def combine_features(self, fingerprint_1_tokens, fingerprint_2_tokens):
        # This function takes a pair of embeddings [B, 49, 1024], [B, 49, 1024] and returns a B logit scores [B]
        # fingerprint_1_tokens        = self.linear_cl(fingerprint_1_tokens)
        # fingerprint_2_tokens        = self.linear_cl(fingerprint_2_tokens)
        batch_size                  = fingerprint_1_tokens.shape[0]
        sep_token                   = self.sep_token.repeat(batch_size, 1, 1)
        combine_features            = torch.cat((fingerprint_1_tokens, sep_token, fingerprint_2_tokens), dim=1)            
        fused_match_representation  = self.fusion(combine_features) 
        fingerprint_1 = fused_match_representation[:,:197,:].mean(dim=1)
        fingerprint_2 = fused_match_representation[:,198:,:].mean(dim=1)

        fingerprint_1_norm = self.l2_norm(fingerprint_1)
        fingerprint_2_norm = self.l2_norm(fingerprint_2)
        
        similarities = torch.sum(fingerprint_1_norm * fingerprint_2_norm, axis=1)

        differences  = fingerprint_1 - fingerprint_2
        squared_differences = differences ** 2
        sum_squared_differences = torch.sum(squared_differences, axis=1)
        distances = torch.sqrt(sum_squared_differences)
        return similarities, distances
    
    def get_tokens(self, image, ftype):
        swin   = self.swin_cl
        tokens = swin(image)
        return tokens

    def freeze_backbone(self):
        for param in self.swin_cl.parameters():
            param.requires_grad = False

    def forward(self, x_cl, x_cb):
        x_cl_tokens = self.swin_cl(x_cl)
        x_cb_tokens = self.swin_cl(x_cb)
        return x_cl_tokens, x_cb_tokens