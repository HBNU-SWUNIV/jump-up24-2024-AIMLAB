import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from functools import partial

from timm.models.vision_transformer import PatchEmbed, Block
from util.pos_embed import get_2d_sincos_pos_embed

class NormalizedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty((out_features, in_features)))

        self.reset_parameters()

        self.scale = nn.Parameter(torch.ones(1), requires_grad=True)
        # self.bias = nn.Parameter(torch.empty(out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))        

    def forward(self, input):
        normalized_input = F.normalize(input, p=2, dim=-1)
        normalized_weight = F.normalize(self.weight, p=2, dim=-1)

        output = F.linear(normalized_input, normalized_weight, None)            
        output = self.scale * output + self.bias

        if torch.isnan(output).any() or torch.isinf(output).any():
            print("NaN or Inf in NormalizedLinear output detected")

        return output

class PatchViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, gap_dim=1, dropout_ratio=0.):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for _ in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.norm_pix_loss = norm_pix_loss

        # self.gap_dim = gap_dim
        # if self.gap_dim == 2:
        #     self.fc = nn.Linear(196, 2)
        # else:
        # self.head = nn.Linear(768, 2)
        # self.head = NormalizedLinear(768, 1)
        
        self.dropout = nn.Dropout(dropout_ratio)
        self.head = NormalizedLinear(768, 2)

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.cls_token, std=.02)
        # torch.nn.init.normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward(self, imgs, is_train):
        latent = self.forward_encoder(imgs)     
        latent_gap = latent[:, 1:, :].mean(dim=1)   

        if is_train:
            latent_gap = self.dropout(latent_gap)

        pred = self.head(latent_gap)

        return latent, pred

def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = PatchViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b
