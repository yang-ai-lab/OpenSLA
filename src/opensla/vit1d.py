"""1D Vision Transformer backbone of the frozen waveform encoder.

Adapted from OSF (https://github.com/yang-ai-lab/OSF-Open-Sleep-FM), MIT License,
Copyright (c) 2026 Health Intelligence Lab @ UCLA.

Patchify modes:
- lead_wise=0: 1D patchify (all channels in one patch)
- lead_wise=1: 2D patchify (channel groups)

Note: lead_emb is DEPRECATED and not used in data flow. It is kept only for
checkpoint compatibility. Do NOT add lead_emb usage without careful consideration.
"""
import torch
import torch.nn as nn
from einops import rearrange


class DropPath(nn.Module):
    '''
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    '''
    def __init__(self, drop_prob: float, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob <= 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class PreNorm(nn.Module):
    def __init__(self,
                 dim: int,
                 fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """
    MLP Module with GELU activation fn + dropout.
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int,
                 drop_out_rate=0.):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                 nn.GELU(),
                                 nn.Dropout(drop_out_rate),
                                 nn.Linear(hidden_dim, output_dim),
                                 nn.Dropout(drop_out_rate))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 heads: int = 8,
                 dim_head: int = 64,
                 qkv_bias: bool = True,
                 drop_out_rate: float = 0.,
                 attn_drop_out_rate: float = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == input_dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_drop_out_rate)
        self.to_qkv = nn.Linear(input_dim, inner_dim * 3, bias=qkv_bias)

        if project_out:
            self.to_out = nn.Sequential(nn.Linear(inner_dim, output_dim),
                                        nn.Dropout(drop_out_rate))
        else:
            self.to_out = nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int,
                 heads: int = 8,
                 dim_head: int = 32,
                 qkv_bias: bool = True,
                 drop_out_rate: float = 0.,
                 attn_drop_out_rate: float = 0.,
                 drop_path_rate: float = 0.):
        super().__init__()
        attn = Attention(input_dim=input_dim,
                         output_dim=output_dim,
                         heads=heads,
                         dim_head=dim_head,
                         qkv_bias=qkv_bias,
                         drop_out_rate=drop_out_rate,
                         attn_drop_out_rate=attn_drop_out_rate)
        self.attn = PreNorm(dim=input_dim,
                            fn=attn)
        self.droppath1 = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

        ff = FeedForward(input_dim=output_dim,
                         output_dim=output_dim,
                         hidden_dim=hidden_dim,
                         drop_out_rate=drop_out_rate)
        self.ff = PreNorm(dim=output_dim,
                          fn=ff)
        self.droppath2 = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x):
        x = self.droppath1(self.attn(x)) + x
        x = self.droppath2(self.ff(x)) + x
        return x


class ViT(nn.Module):

    def __init__(self,
                 num_leads: int,
                 seq_len: int,
                 patch_size: int,
                 lead_wise: int = 0,
                 patch_size_ch: int = 4,
                 use_lead_embedding: bool = True,
                 width: int = 768,
                 depth: int = 12,
                 mlp_dim: int = 3072,
                 heads: int = 12,
                 dim_head: int = 64,
                 qkv_bias: bool = True,
                 drop_out_rate: float = 0.,
                 attn_drop_out_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 **kwargs):
        super().__init__()
        assert seq_len % patch_size == 0
        num_patches_time = seq_len // patch_size

        self.lead_wise = lead_wise
        self.use_lead_embedding = bool(use_lead_embedding)
        self.width = width
        self.depth = depth

        if lead_wise == 0:
            self.to_patch_embedding = nn.Conv1d(num_leads, width, kernel_size=patch_size,
                                                stride=patch_size, bias=False)
            N_max = num_patches_time
            self.lead_emb = None
        else:
            self.to_patch_embedding = nn.Conv2d(1, width,
                                                kernel_size=(patch_size_ch, patch_size),
                                                stride=(patch_size_ch, patch_size),
                                                bias=False)
            Lr = num_leads // patch_size_ch
            N_max = Lr * num_patches_time
            self.lead_emb = nn.Embedding(Lr, width) if self.use_lead_embedding else None

        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos_embedding = nn.Parameter(torch.zeros(1, N_max + 1, width))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        self.dropout = nn.Dropout(drop_out_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        for i in range(depth):
            block = TransformerBlock(input_dim=width, output_dim=width,
                                     hidden_dim=mlp_dim, heads=heads, dim_head=dim_head,
                                     qkv_bias=qkv_bias, drop_out_rate=drop_out_rate,
                                     attn_drop_out_rate=attn_drop_out_rate,
                                     drop_path_rate=dpr[i])
            self.add_module(f'block{i}', block)

        self.norm = nn.LayerNorm(width)
        self.head = nn.Identity()


    def to_tokens_2d(self, series: torch.Tensor,
                     patch_size_ch: int | None = None,
                     patch_size_time: int | None = None):
        """Patchify only (no pos embedding). Returns (tokens, meta)."""
        B, L, T = series.shape

        if self.lead_wise == 0:
            x = self.to_patch_embedding(series)         # [B,C,Nt]
            Nt = x.shape[-1]
            x = rearrange(x, 'b c n -> b n c')          # [B,Nt,C]
            meta = dict(lead_wise=0, L=L, Nt=Nt, pz_ch=1)
            return x, meta

        # lead_wise == 1
        if patch_size_ch is None or patch_size_time is None:
            kch, kt = self.to_patch_embedding.kernel_size
            patch_size_ch = patch_size_ch or kch
            patch_size_time = patch_size_time or kt
        assert L % patch_size_ch == 0 and T % patch_size_time == 0

        x = series.unsqueeze(1)                          # [B,1,L,T]
        x = self.to_patch_embedding(x)                   # [B,C,Lr,Nt]
        Lr, Nt = x.shape[-2], x.shape[-1]
        x = rearrange(x, 'b c lr nt -> b (lr nt) c')     # [B,Lr*Nt,C]
        if self.use_lead_embedding and self.lead_emb is not None:
            row_ids = torch.arange(Lr, device=x.device).repeat_interleave(Nt)
            x = x + self.lead_emb(row_ids)[None, :, :]
        meta = dict(lead_wise=1, L=L, Nt=Nt, pz_ch=patch_size_ch)
        return x, meta

    def forward_encoding(self, series: torch.Tensor,
                         return_sequence: bool = False):
        """Encode with CLS token. Returns (cls, patches) or full sequence if return_sequence=True."""
        tokens, meta = self.to_tokens_2d(series)
        B = tokens.size(0)
        cls_tok = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tok, tokens], dim=1)         # [B,N+1,C]

        pe = self.pos_embedding[:, :x.size(1), :].to(x.device)

        x = x + pe

        x = self._run_blocks(x)
        if return_sequence:
            return x
        cls, patches = x[:, 0], x[:, 1:]
        
        return cls, patches


    def _run_blocks(self, x: torch.Tensor):
        x = self.dropout(x)
        for i in range(self.depth):
            x = getattr(self, f'block{i}')(x)
        x = self.norm(x)
        return self.head(x)

    def forward(self, series: torch.Tensor):
        cls, _ = self.forward_encoding(series, return_sequence=False)
        return cls

    def forward_avg_pool(self, series: torch.Tensor):
        """Returns avg-pooled patch embeddings. series: [B,C,T] -> [B,D]"""
        _, patches = self.forward_encoding(series, return_sequence=False)  # [B,N,D]
        return patches.mean(dim=1)  # [B,D]

    def reset_head(self, num_classes=1):
        del self.head
        self.head = nn.Linear(self.width, num_classes)


    
def vit_nano(num_leads, num_classes=1, seq_len=5000, patch_size=50, **kwargs):
    model_args = dict(num_leads=num_leads,
                      num_classes=num_classes,
                      seq_len=seq_len,
                      patch_size=patch_size,
                      width=128,
                      depth=6,
                      heads=4,
                      mlp_dim=512,
                      **kwargs)
    return ViT(**model_args)


def vit_tiny(num_leads, num_classes=1, seq_len=5000, patch_size=50, **kwargs):
    model_args = dict(num_leads=num_leads,
                      num_classes=num_classes,
                      seq_len=seq_len,
                      patch_size=patch_size,
                      width=192,
                      depth=12,
                      heads=3,
                      mlp_dim=768,
                      **kwargs)
    return ViT(**model_args)


def vit_small(num_leads, num_classes=1, seq_len=5000, patch_size=50, **kwargs):
    model_args = dict(num_leads=num_leads,
                      num_classes=num_classes,
                      seq_len=seq_len,
                      patch_size=patch_size,
                      width=384,
                      depth=12,
                      heads=6,
                      mlp_dim=1536,
                      **kwargs)
    return ViT(**model_args)


def vit_middle(num_leads, num_classes=1, seq_len=5000, patch_size=50, **kwargs):
    model_args = dict(num_leads=num_leads,
                      num_classes=num_classes,
                      seq_len=seq_len,
                      patch_size=patch_size,
                      width=512,
                      depth=12,
                      heads=8,
                      mlp_dim=2048,
                      **kwargs)
    return ViT(**model_args)


def vit_base(num_leads, num_classes=1, seq_len=5000, patch_size=50, **kwargs):
    model_args = dict(num_leads=num_leads,
                      num_classes=num_classes,
                      seq_len=seq_len,
                      patch_size=patch_size,
                      width=768,
                      depth=12,
                      heads=12,
                      mlp_dim=3072,
                      **kwargs)
    return ViT(**model_args)


def vit_large(num_leads, num_classes=1, seq_len=5000, patch_size=50, **kwargs):
    return ViT(
        num_leads=num_leads,
        num_classes=num_classes,
        seq_len=seq_len,
        patch_size=patch_size,
        width=1024,
        depth=24,
        heads=16,
        mlp_dim=4096,
        **kwargs
    )

def vit_xl(num_leads, num_classes=1, seq_len=5000, patch_size=50, **kwargs):
    return ViT(
        num_leads=num_leads,
        num_classes=num_classes,
        seq_len=seq_len,
        patch_size=patch_size,
        width=1536,
        depth=24,        
        heads=24,
        mlp_dim=6144,
        **kwargs
    )
