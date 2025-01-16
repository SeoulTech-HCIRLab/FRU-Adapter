# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
####models_vit_beforeAttach####
from functools import partial

import torch
import torch.nn as nn
from torch.autograd import Function

import timm.models.vision_transformer
from einops import rearrange
import math

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd, **kwargs: None):
        ctx.lambd = lambd
        return x.view_as(x)

    def backward(ctx, grad_output):
        return grad_output * -ctx.lambd, None

class TemporalSqueezeExcitationAdapter(nn.Module):
    def __init__(self,
                 channel = 197,
                 embded_dim = 1024,
                 Frame = 16,
                 hidden_dim = 128):
        super().__init__()

        self.Frame = Frame

        self.linear1 = nn.Linear(embded_dim ,hidden_dim)
        self.linear2 = nn.Linear(hidden_dim,embded_dim)

        self.T_linear1 = nn.Linear(Frame, Frame)
        self.softmax = nn.Softmax(dim=1)
        self.ln = nn.LayerNorm(hidden_dim)
        
        self.TFormer = TemporalTransformer(frame=Frame,emb_dim=hidden_dim)

    def forward(self, x):
        #x = bt N D 
        bt, n,d = x.shape
        #bN t D ,FC -> GAP(D) 이후 TSE-> t-former(only D) -> FC
        x = rearrange(x, '(b t) n d-> (b n) t d', t = self.Frame, n = n, d = d)

        x = self.linear1(x) # bn t d
        x = self.ln(x) 

        _, _,down = x.shape

        x = rearrange(x, '(b n) t d-> b t (n d)', t = self.Frame, n = n, d = down)

        x1 = x.mean(-1).flatten(1) # bn t 
        x1 = self.T_linear1(x1) # bn t
        x1 = self.softmax(x1).unsqueeze(-1) #bn t 1
        x = x * x1 #bn t d

        x = rearrange(x, 'b t (n d)-> (b n) t d', t = self.Frame, n = n, d = down)

        x = self.TFormer(x)
        x = self.linear2(x)
        #bt n d
        x = rearrange(x, '(b n) t d-> (b t) n d', t = self.Frame, n = n, d = d)
        return x

class TemporalTransformer(nn.Module):
    def __init__(self, 
                 frame = 16,
                 #channel = 8,
                 emb_dim = 49,
                 ):
        super().__init__()
        
        self.proj_Q = nn.Linear(emb_dim,emb_dim)
        self.proj_K = nn.Linear(emb_dim,emb_dim)
        self.proj_V = nn.Linear(emb_dim,emb_dim)
        self.proj_output = nn.Linear(emb_dim,emb_dim)
        
        self.norm = nn.LayerNorm(emb_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        #B C T H W 
        _,_,E = x.shape
            # x = x.contiguous().transpose(1,2) #B T E  where E = C * H * W 
            # x = x.contiguous().view(B,T,-1)
        #x = x + self.Temporal_PositionalEncoding

        x1 = self.norm(x) 

        q = self.proj_Q(x1)
        k = self.proj_K(x1)
        v = self.proj_V(x1)

        q_scaled = q * math.sqrt(1.0 / float(E))

        attn_output_weights = q_scaled @ k.transpose(-2, -1)
        attn_output_weights = self.softmax(attn_output_weights)
        attn_output = attn_output_weights @ v 
        attn_output = self.proj_output(attn_output) #B T E  where E = C * H * W
        attn_output = attn_output + x 

        return attn_output 

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

class TSE_Adapter(nn.Module):

    def __init__(self, in_channels, adapter_channels, kernel_size):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, adapter_channels)
        self.conv = nn.Conv3d(
            adapter_channels, adapter_channels,
            kernel_size=kernel_size,
            stride=(1, 1, 1),
            padding=tuple(x // 2 for x in kernel_size),
            groups=adapter_channels,
        )
        self.fc2 = nn.Linear(adapter_channels, in_channels)

        self.T_linear1 = nn.Linear(16, 16)
        self.softmax = nn.Softmax(dim=1)
        # self.ln = nn.LayerNorm(adapter_channels)

        nn.init.constant_(self.conv.weight, 0.)
        nn.init.constant_(self.conv.bias, 0.)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x):
        T = 16
        BT, L, C = x.size()
        B = BT // T
        Ca = self.conv.in_channels
        H = W = round(math.sqrt(L - 1))
        assert L - 1 == H * W
        x_id = x
        x = x[:, 1:, :]
        x = self.fc1(x) # bt hw c
        
        x = x.view(B, T, H, W, Ca).permute(0, 4, 1, 2, 3).contiguous() #b c t h w 
        x = self.conv(x) #b c t h w 

        x = rearrange(x, 'b c t h w-> b t (h w c)', t = T, c = Ca, h = H, w = W)
        x1 = x.mean(-1).flatten(1) # bn t 
        x1 = self.T_linear1(x1) # bn t
        x1 = self.softmax(x1).unsqueeze(-1) #bn t 1
        x = x * x1 #bn t d

        x = rearrange(x, 'b t (h w c)-> (b t) (h w) c', t = T, c = Ca, h = H, w = W)
        # x = x.permute(0, 2, 3, 4, 1).contiguous().view(BT, L - 1, Ca)
        x = self.fc2(x)
        x_id[:, 1:, :] += x
        return x_id

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
        
        # self.tsea_blocks = nn.ModuleList([
        #     TemporalSqueezeExcitationAdapter(embded_dim=1024) for _ in range(len(self.blocks))
        # ])
        self.tsea = nn.ModuleList([
            TSE_Adapter(in_channels=1024,adapter_channels=128,kernel_size=(3, 1, 1)) for _ in range(len(self.blocks))
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
            x = self.tsea[i](x)
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