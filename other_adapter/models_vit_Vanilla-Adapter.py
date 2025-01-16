# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
####models_vit_vanilla Adapter####
from functools import partial

import torch
import torch.nn as nn
from torch.autograd import Function

import timm.models.vision_transformer
from einops import rearrange
import math
from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd, **kwargs: None):
        ctx.lambd = lambd
        return x.view_as(x)

    def backward(ctx, grad_output):
        return grad_output * -ctx.lambd, None
    
class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    
class Vanilla_Adapter(nn.Module):
    #https://arxiv.org/pdf/1902.00751
    def __init__(self,
                 embded_dim = 1024,
                 hidden_dim = 128):
        super().__init__()

        self.linear1 = nn.Linear(embded_dim ,hidden_dim)
        self.linear2 = nn.Linear(hidden_dim,embded_dim)

        self.gelu = GELU()

    def forward(self, x):
        x1 = self.linear1(x)
        x1 = self.gelu(x1)
        x1 = self.linear2(x1)
        x = x + x1
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x, return_attention=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.vanilla_adapter1 = Vanilla_Adapter()
        self.vanilla_adapter2 = Vanilla_Adapter()

    def forward(self, x, return_attention=False):
        if return_attention:
            feat, attn = self.attn(self.norm1(x), True)
            feat = self.vanilla_adapter1(feat)
            x = x + self.drop_path(feat)
            x = x + self.drop_path(self.vanilla_adapter2(self.mlp(self.norm2(x))))
            return x, attn
        else:
            x = x + self.drop_path(self.vanilla_adapter1(self.attn(self.norm1(x))))
            x = x + self.drop_path(self.vanilla_adapter2(self.mlp(self.norm2(x))))
            # x = x + self.drop_path(self.attn(self.norm1(x)))
            # x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        
####if B, T ,E 
class classifier(nn.Module):
    def __init__(self, 
                 embed_dim = 1024,
                 num_classes = 7,
                 Frame = 16
                 ):
        super().__init__()

        self.norm = nn.LayerNorm(embed_dim)
        self.fc1  = nn.Linear(embed_dim,embed_dim)
        self.fc2 = nn.Linear(embed_dim,num_classes)

    def forward(self, x):

        x = x.contiguous().view(-1,16,1024) # b t 1024
        x = self.fc1(x).mean(dim=1) # b t 1024 -> b 1024
        x = self.fc2(self.norm(x))

        return x
    
class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, grad_reverse=0, num_classes=7, num_subjects=0, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

        # Classifier head
        # self.AU_head = nn.Linear(kwargs['embed_dim'], num_classes)
        self.classifier = classifier(embed_dim=embed_dim,num_classes=num_classes)
        self.grad_reverse = grad_reverse  # default is 1.0
        print(f"using ID adversarial: {self.grad_reverse}" )
        print(f"num classes: {num_classes}, num subjects: {num_subjects}")
        if not self.grad_reverse == 0:
            self.ID_head = nn.Linear(kwargs['embed_dim'], num_subjects)
            print(f"activate ID adver head")
        
        # patch_size=16
        embed_dim=1024
        depth=24
        num_heads=16
        mlp_ratio=4
        qkv_bias=True,
        
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
        
        drop_path_rate = 0.1 
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_path_rate,
                attn_drop=0., drop_path=dpr[i], norm_layer=norm_layer, act_layer=nn.GELU)
            for i in range(depth)])

    def forward_features(self, x):
        # B = x.shape[0]
        x = rearrange(x, 'b c t h w-> (b t) c h w', c = x.shape[1], t = x.shape[2], h = x.shape[3], w = x.shape[4])
        # print("org",x.shape) # b, 3, 224, 224
        x = self.patch_embed(x)
        # print("patch_emb",x.shape) #b, 196, 1024
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # for blk in self.blocks:
        #     x = blk(x) + self.tsea(x) # b 197 1024
        for i, blk in enumerate(self.blocks):
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token 여기 들어감 
            outcome = self.fc_norm(x) # b 1024
        else:
            x = self.norm(x)
            outcome = x[:, 0]
        return outcome

    def forward(self, x):
        x = self.forward_features(x) # bt 1024
        x = x.contiguous().view(-1,16,1024) # b t 1024
        x = self.classifier(x)
        # AU_pred = self.AU_head(x)
        
        return x


def vit_small_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    checkpoint = torch.load(r'/home/gpuadmin/MB/FMAE_RAFDB.pth', map_location=lambda storage, loc: storage)
    model = load_pretrained_weights(model, checkpoint)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def load_pretrained_weights(model, checkpoint):
    import collections
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state_dict = collections.OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        # If the pretrained state_dict was saved as nn.DataParallel,
        # keys would contain "module.", which should be ignored.
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    # new_state_dict.requires_grad = False
    model_dict.update(new_state_dict)

    model.load_state_dict(model_dict)
    print('load_weight', len(matched_layers))
    return model