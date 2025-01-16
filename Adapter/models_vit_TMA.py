# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
####models_vit_TMA####
from functools import partial

import torch
import torch.nn as nn
from torch.autograd import Function

import timm.models.vision_transformer
from timm.models.vision_transformer import _cfg, Attention, DropPath, Mlp, partial, LayerScale, _cfg, Block
from timm.models.layers import PatchEmbed, trunc_normal_

from einops import rearrange
import math

# class GELU(nn.Module):
#     def forward(self, x):
#         return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

####models_vit_TMA
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd, **kwargs: None):
        ctx.lambd = lambd
        return x.view_as(x)

    def backward(ctx, grad_output):
        return grad_output * -ctx.lambd, None

class Adapter(nn.Module):
    #https://arxiv.org/pdf/2312.05447
    def __init__(self, D_features, mlp_ratio=0.25,
                 act_layer=nn.GELU, skip_connect=True,
                 attention=True,
                 num_heads=8, qkv_bias=False, attn_drop=0., drop=0.):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = 128#int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        self.attn = Attention(D_hidden_features, num_heads=num_heads,
                              qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop) if attention else nn.Identity()

        self.apply(self._init_weights)
        nn.init.constant_(self.D_fc2.weight, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.attn(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x


class TMAdapter(nn.Module):
    def __init__(self, D_features, num_frames, ratio=0.25):
        super().__init__()
        self.num_frames = num_frames
        self.T_Adapter = Adapter(
            D_features, mlp_ratio=ratio, skip_connect=False, attention=True)
        self.norm = nn.LayerNorm(D_features)
        self.S_Adapter = Adapter(
            D_features, mlp_ratio=ratio, skip_connect=False, attention=False)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x is (BT, HW+1, D)
        bt, n, d = x.shape
        ## temporal adaptation
        xt = rearrange(x, '(b t) n d -> (b n) t d', t=self.num_frames)
        xt = self.T_Adapter(xt)
        x = rearrange(xt, '(b n) t d -> (b t) n d', n=n)

        x = self.S_Adapter(self.norm(x))
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
        
        self.TMA_blocks = nn.ModuleList([
            TMAdapter(D_features=1024,num_frames=16) for _ in range(len(self.blocks))
        ])
        # self.tsea = TemporalSqueezeExcitationAdapter(embded_dim=1024)

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
            x = x + self.TMA_blocks[i](x)
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