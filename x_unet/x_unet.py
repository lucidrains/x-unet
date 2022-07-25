from functools import partial

import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def is_power_two(n):
    return math.log2(n).is_integer()

def cast_tuple(val, length = None):
    if isinstance(val, list):
        val = tuple(val)

    output = val if isinstance(val, tuple) else ((val,) * default(length, 1))

    if exists(length):
        assert len(output) == length

    return output

def l2norm(t):
    return F.normalize(t, dim = -1)

# helper classes

def Upsample(dim, dim_out):
    return nn.ConvTranspose3d(dim, dim_out, (1, 4, 4), (1, 2, 2), (0, 1, 1))

def Downsample(dim, dim_out):
    return nn.Conv3d(dim, dim_out, (1, 4, 4), (1, 2, 2), (0, 1, 1))

# normalization

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma

# resnet blocks

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (1, 3, 3), padding = (0, 1, 1))
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)
        return self.act(x)

class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        groups = 8,
        nested_unet_depth = 0,
        nested_unet_dim = 32
    ):
        super().__init__()
        self.block1 = Block(dim, dim_out, groups = groups)

        if nested_unet_depth > 0:
            self.block2 = NestedResidualUnet(dim_out, depth = nested_unet_depth, M = nested_unet_dim, add_residual = True)
        else:
            self.block2 = Block(dim_out, dim_out, groups = groups)

        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)

# conv next
# https://arxiv.org/abs/2201.03545

class ConvNextBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        *,
        mult = 2,
        nested_unet_depth = 0,
        nested_unet_dim = 32
    ):
        super().__init__()
        self.ds_conv = nn.Conv3d(dim, dim, (1, 7, 7), padding = (0, 3, 3), groups = dim)

        self.net = nn.Sequential(
            LayerNorm(dim),
            nn.Conv3d(dim, dim_out * mult, (1, 3, 3), padding = (0, 1, 1)),
            nn.GELU(),
            nn.Conv3d(dim_out * mult, dim_out, (1, 3, 3), padding = (0, 1, 1))
        )

        self.nested_unet = NestedResidualUnet(dim_out, depth = nested_unet_depth, M = nested_unet_dim, add_residual = True) if nested_unet_depth > 0 else nn.Identity()

        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):
        h = self.ds_conv(x)
        h = self.net(h)
        h = self.nested_unet(h)
        return h + self.res_conv(x)

# attention

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 64,
        scale = 16
    ):
        super().__init__()
        self.scale = scale
        self.heads = heads
        inner_dim = heads * dim_head
        self.norm = LayerNorm(dim)

        self.to_qkv = nn.Conv3d(dim, inner_dim * 3, 1, bias = False)
        self.to_out = nn.Conv3d(inner_dim, dim, 1, bias = False)

    def forward(self, x):
        f, h, w = x.shape[-3:]

        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) ... -> b h (...) c', h = self.heads), (q, k, v))

        q, k = map(l2norm, (q, k))

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h (f x y) d -> b (h d) f x y', f = f, x = h, y = w)
        return self.to_out(out)

class FeatureMapConsolidator(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_ins = tuple(),
        dim_outs = tuple(),
        resize_fmap_before = True,
        conv_block_fn = None
    ):
        super().__init__()
        assert len(dim_ins) == len(dim_outs)
        self.needs_consolidating = len(dim_ins) > 0

        block_fn = default(conv_block_fn, Block)

        self.fmap_convs = nn.ModuleList([block_fn(dim_in, dim_out) for dim_in, dim_out in zip(dim_ins, dim_outs)])
        self.resize_fmap_before = resize_fmap_before

        self.final_dim_out = dim + (sum(dim_outs) if len(dim_outs) > 0 else 0)

    def resize_fmaps(self, fmaps, target_size):
        return [F.interpolate(fmap, (fmap.shape[-3], target_size, target_size)) for fmap in fmaps]

    def forward(self, x, fmaps = None):
        target_size = x.shape[-1]

        fmaps = default(fmaps, tuple())

        if not self.needs_consolidating:
            return x

        if self.resize_fmap_before:
            fmaps = self.resize_fmaps(fmaps, target_size)

        outs = []
        for fmap, conv in zip(fmaps, self.fmap_convs):
            outs.append(conv(fmap))

        if self.resize_fmap_before:
            outs = self.resize_fmaps(outs, target_size)

        return torch.cat((x, *outs), dim = 1)

# unet

class XUnet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults = (1, 2, 4, 8),
        nested_unet_depths = (0, 0, 0, 0),
        nested_unet_dim = 32,
        channels = 3,
        use_convnext = False,
        resnet_groups = 8,
        consolidate_upsample_fmaps = True
    ):
        super().__init__()
        self.channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv3d(channels, init_dim, (1, 7, 7), padding = (0, 3, 3))

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        num_resolutions = len(in_out)

        # resnet or convnext

        blocks = partial(ConvNextBlock) if use_convnext else partial(ResnetBlock, groups = resnet_groups)

        # whether to use nested unet, as in unet squared paper

        nested_unet_depths = cast_tuple(nested_unet_depths, num_resolutions)

        # modules for all layers

        skip_dims = []

        for ind, ((dim_in, dim_out), nested_unet_depth) in enumerate(zip(in_out, nested_unet_depths)):
            is_last = ind >= (num_resolutions - 1)
            skip_dims.append(dim_in)

            self.downs.append(nn.ModuleList([
                blocks(dim_in, dim_in, nested_unet_depth = nested_unet_depth, nested_unet_dim = nested_unet_dim),
                blocks(dim_in, dim_in, nested_unet_depth = nested_unet_depth, nested_unet_dim = nested_unet_dim),
                Downsample(dim_in, dim_out)
            ]))

        mid_dim = dims[-1]
        self.mid = blocks(mid_dim, mid_dim)
        self.mid_attn = Attention(mid_dim)
        self.mid_after = blocks(mid_dim, mid_dim)
        self.mid_upsample = Upsample(mid_dim, dims[-2])

        for ind, ((dim_in, dim_out), nested_unet_depth) in enumerate(zip(reversed(in_out[:-1]), reversed(nested_unet_depths[:-1]))):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                blocks(dim_out + skip_dims.pop(), dim_out, nested_unet_depth = nested_unet_depth, nested_unet_dim = nested_unet_dim),
                blocks(dim_out, dim_out, nested_unet_depth = nested_unet_depth, nested_unet_dim = nested_unet_dim),
                Upsample(dim_out, dim_in) if not is_last else nn.Identity()
            ]))


        out_dim = default(out_dim, channels)

        if consolidate_upsample_fmaps:
            self.consolidator = FeatureMapConsolidator(
                dim,
                dim_ins = tuple(map(lambda m: dim * m, dim_mults)),
                dim_outs = (dim,) * len(dim_mults),
                conv_block_fn = blocks
            )
        else:
            self.consolidator = FeatureMapConsolidator(dim = dim)

        final_dim_in = self.consolidator.final_dim_out

        self.final_conv = nn.Sequential(
            blocks(final_dim_in + dim, dim),
            nn.Conv3d(dim, out_dim, 3, padding = 1)
        )

    def forward(self, x):
        is_image = x.ndim == 4
        if is_image:
            x = rearrange(x, 'b c h w -> b c 1 h w')

        x = self.init_conv(x)
        r = x.clone()

        down_hiddens = []
        up_hiddens = []

        for block, block2, downsample in self.downs:
            x = block(x)
            x = block2(x)
            down_hiddens.append(x)
            x = downsample(x)

        x = self.mid(x)
        x = self.mid_attn(x) + x
        x = self.mid_after(x)

        up_hiddens.append(x)
        x = self.mid_upsample(x)


        for block, block2, upsample in self.ups:
            x = torch.cat((x, down_hiddens.pop()), dim=1)
            x = block(x)
            x = block2(x)
            up_hiddens.insert(0, x)
            x = upsample(x)

        x = self.consolidator(x, up_hiddens)

        x = torch.cat((x, r), dim = 1)
        out = self.final_conv(x)

        if is_image:
            out = rearrange(out, 'b c 1 h w -> b c h w')

        return out

# RSU

class PixelShuffleUpsample(nn.Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        scale_factor = 2
    ):
        super().__init__()
        self.scale_squared = scale_factor ** 2
        dim_out = default(dim_out, dim)
        conv = nn.Conv2d(dim, dim_out * self.scale_squared, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            nn.PixelShuffle(scale_factor)
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, *rest_dims = conv.weight.shape
        conv_weight = torch.empty(o // self.scale_squared, i, *rest_dims)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o r) ...', r = self.scale_squared)

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        is_video = x.ndim == 5

        if is_video:
            f = x.shape[2]
            x = rearrange(x, 'b c f h w -> (b f) c h w')

        x = self.net(x)

        if is_video:
            x = rearrange(x, '(b f) c h w -> b c f h w', f = f)

        return x

class NestedResidualUnet(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        M = 32,
        add_residual = False,
        groups = 4
    ):
        super().__init__()

        self.depth = depth
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        for ind in range(depth):
            is_first = ind == 0
            dim_in = dim if is_first else M

            down = nn.Sequential(
                nn.Conv2d(dim_in, M, 4, stride = 2, padding = 1),
                nn.GroupNorm(groups, M),
                nn.SiLU()
            )

            up = nn.Sequential(
                PixelShuffleUpsample(2 * M, dim_in),
                nn.GroupNorm(groups, dim_in),
                nn.SiLU()
            )

            self.downs.append(down)
            self.ups.append(up)

        self.mid = nn.Sequential(
            nn.Conv2d(M, M, 3, padding = 1),
            nn.GroupNorm(groups, M),
            nn.SiLU()
        )

        self.add_residual = add_residual

    def forward(self, x, residual = None):
        is_video = x.ndim == 5

        if self.add_residual:
            residual = default(residual, x.clone())

        if is_video:
            f = x.shape[2]
            x = rearrange(x, 'b c f h w -> (b f) c h w')

        *_, h, w = x.shape

        assert h == w, 'only works with square images'
        assert is_power_two(h), 'height and width must be power of two'
        assert (h % (2 ** self.depth)) == 0, 'the unet has too much depth for the image being passed in'

        # hiddens

        hiddens = []

        # unet

        for down in self.downs:
            x = down(x)
            hiddens.append(x.clone().contiguous())

        x = self.mid(x)

        for up in reversed(self.ups):
            x = torch.cat((x, hiddens.pop()), dim = 1)
            x = up(x)

        # adding residual

        if is_video:
            x = rearrange(x, '(b f) c h w -> b c f h w', f = f)

        if self.add_residual:
            x = x + residual
            x = F.silu(x)

        return x
