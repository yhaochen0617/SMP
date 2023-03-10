import torch 
import torch.nn as nn 
from torch.nn import functional as F
from utils import trunc_normal_

class Mlp(nn.Module):
    def __init__(self,feat_nums=4,expand=8,drop=0.0):
        super(Mlp, self).__init__()
        mid = feat_nums * expand
        self.m1 = nn.Linear(feat_nums, mid)
        self.m2 = nn.Linear(mid, mid)
        self.m21 = nn.Linear(mid, mid)
        self.m3 = nn.Linear(mid, mid//2)
        self.m4 = nn.Linear(mid//2, 1)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = F.gelu(self.m1(x))
        x = F.gelu(self.m2(x))
        x = self.drop(x)
        x = F.gelu(self.m21(x))
        x = self.drop(x)
        x = F.gelu(self.m3(x))
        x = self.drop(x)
        x = self.m4(x)
        x = torch.sigmoid(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=3, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Trans(nn.Module):
    def __init__(self,feat_nums=4,expand=8,drop=0.0,norm=None):  #norm=nn.LayerNorm
        super(Trans, self).__init__()
        mid = feat_nums * expand
        self.m0 = nn.Linear(feat_nums, mid)

        self.att1 = Attention(mid,attn_drop=0.1,proj_drop=0.1)
        self.m11 = nn.Linear(mid, mid)
        self.m12 = nn.Linear(mid, mid)
        
        self.att2 = Attention(mid,attn_drop=0.1,proj_drop=0.1)
        self.m21 = nn.Linear(mid, mid)
        self.m22 = nn.Linear(mid, mid)

        self.m3 = nn.Linear(mid, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    
    def forward(self, x, iters=0):
        x = F.gelu(self.m0(x))

        n,c = x.shape
        
        x = x.reshape(n,1,c)
        x = x + self.att1(x)
        x = x + self.m12(F.gelu(self.m11(x))) 

        x = self.drop(x)

        x = x.reshape(n,1,c)
        x = x + self.att2(x)
        x = x + self.m22(F.gelu(self.m21(x))) 

        x = self.m3(x)
        x = torch.sigmoid(x)
        return x.squeeze()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)  # from the origin timm 
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

