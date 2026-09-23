from __future__ import annotations
from functools import partial

import torch
from torch import einsum, nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

from torch_einops_utils import safe_cat, tree_map_tensor

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(num, denom):
    return (num % denom) == 0

def cast_tuple(val, length = None):
    if isinstance(val, list):
        val = tuple(val)

    output = val if isinstance(val, tuple) else ((val,) * default(length, 1))

    if exists(length):
        assert len(output) == length

    return output

# helper classes

def Upsample(dim, dim_out):
    return nn.ConvTranspose3d(dim, dim_out, (1, 4, 4), (1, 2, 2), (0, 1, 1))

def Downsample(dim, dim_out):
    return nn.Sequential(
        Rearrange('b c f (h s1) (w s2) -> b (c s1 s2) f h w', s1 = 2, s2 = 2),
        nn.Conv3d(dim * 4, dim_out, 1)
    )

# normalization

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return einsum('b c ..., c -> b c ...', F.normalize(x, dim = 1) * self.scale, self.gamma)

class LayerNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + eps).sqrt() * self.gamma

class WeightStandardizedConv3d(nn.Conv3d):
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight

        mean = reduce(weight, 'o ... -> o 1 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1 1', partial(torch.var, unbiased = False))
        weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

# resnet blocks

class Block(Module):
    def __init__(
        self,
        dim,
        dim_out,
        groups = 8,
        weight_standardize = False,
        frame_kernel_size = 1
    ):
        super().__init__()
        kernel_conv_kwargs = partial(kernel_and_same_pad, frame_kernel_size)
        conv = nn.Conv3d if not weight_standardize else WeightStandardizedConv3d

        self.proj = conv(dim, dim_out, **kernel_conv_kwargs(3, 3))
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return self.act(x)

class ResnetBlock(Module):
    def __init__(
        self,
        dim,
        dim_out,
        frame_kernel_size = 1,
        nested_unet_depth = 0,
        nested_unet_dim = 32,
        weight_standardize = False
    ):
        super().__init__()
        self.block1 = Block(
            dim,
            dim_out,
            weight_standardize = weight_standardize,
            frame_kernel_size = frame_kernel_size
        )

        if nested_unet_depth > 0:
            self.block2 = NestedResidualUnet(
                dim_out,
                depth = nested_unet_depth,
                dim_inner = nested_unet_dim,
                frame_kernel_size = frame_kernel_size,
                weight_standardize = weight_standardize,
                add_residual = True
            )
        else:
            self.block2 = Block(
                dim_out,
                dim_out,
                weight_standardize = weight_standardize,
                frame_kernel_size = frame_kernel_size
            )

        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)

# convnext 2

class GRN(Module):
    """ global response normalization, proposed in updated convnext paper """

    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(dim, 1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(dim, 1, 1, 1))

    def forward(self, x):
        spatial_l2_norm = x.norm(p = 2, dim = (2, 3, 4), keepdim = True)
        feat_norm = spatial_l2_norm / spatial_l2_norm.mean(dim = -1, keepdim = True).clamp(min = self.eps)
        return x * feat_norm * self.gamma + self.bias + x

class ConvNextBlock(Module):
    def __init__(
        self,
        dim,
        dim_out,
        *,
        mult = 2,
        frame_kernel_size = 1,
        nested_unet_depth = 0,
        nested_unet_dim = 32
    ):
        super().__init__()
        kernel_conv_kwargs = partial(kernel_and_same_pad, frame_kernel_size)

        self.ds_conv = nn.Conv3d(dim, dim, **kernel_conv_kwargs(7, 7), groups = dim)

        inner_dim = dim_out * mult

        self.net = nn.Sequential(
            LayerNorm(dim),
            nn.Conv3d(dim, inner_dim, **kernel_conv_kwargs(3, 3), groups = dim_out),
            nn.GELU(),
            GRN(inner_dim),
            nn.Conv3d(inner_dim, dim_out, **kernel_conv_kwargs(3, 3), groups = dim_out)
        )

        self.nested_unet = nn.Identity()

        if nested_unet_depth > 0:
            self.nested_unet = NestedResidualUnet(
                dim_out,
                depth = nested_unet_depth,
                dim_inner = nested_unet_dim,
                add_residual = True
            )

        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):
        h = self.ds_conv(x)
        h = self.net(h)
        h = self.nested_unet(h)
        return h + self.res_conv(x)

# feedforward

def FeedForward(dim, mult = 4.):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Conv3d(dim, inner_dim, 1, bias = False),
        nn.GELU(),
        LayerNorm(inner_dim),   # properly credit assign normformer
        nn.Conv3d(inner_dim, dim, 1, bias = False)
    )

# attention

class Attention(Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 64
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
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

        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h (f x y) d -> b (h d) f x y', f = f, x = h, y = w)
        return self.to_out(out)

class TransformerBlock(Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        **kwargs
    ):
        super().__init__()
        self.attn = Attention(dim, **kwargs)
        self.ff = FeedForward(dim)

    def forward(self, x):
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x

class FeatureMapConsolidator(Module):
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

        block_fn = default(conv_block_fn, Block)

        self.fmap_convs = ModuleList([block_fn(dim_in, dim_out) for dim_in, dim_out in zip(dim_ins, dim_outs)])
        self.resize_fmap_before = resize_fmap_before

        self.final_dim_out = dim + sum(dim_outs)

    def resize_fmaps(self, fmaps, height, width):
        return [F.interpolate(fmap, (fmap.shape[-3], height, width)) for fmap in fmaps]

    def forward(self, x, fmaps = None):
        target_height, target_width = x.shape[-2:]

        fmaps = default(fmaps, tuple())

        if self.resize_fmap_before:
            fmaps = self.resize_fmaps(fmaps, target_height, target_width)

        outs = [conv(fmap) for fmap, conv in zip(fmaps, self.fmap_convs)]

        if not self.resize_fmap_before:
            outs = self.resize_fmaps(outs, target_height, target_width)

        return safe_cat((x, *outs), dim = 1)

# attention residuals (xattnres)

def resize_to(x, shape, pooling = 'max'):
    spatial_shape = x.shape[2:]

    if spatial_shape == shape:
        return x

    dim = len(shape)
    is_down = all(s >= t for s, t in zip(spatial_shape, shape))

    if is_down:
        return getattr(F, f'adaptive_{pooling}_pool{dim}d')(x, shape)

    mode = ('linear', 'bilinear', 'trilinear')[dim - 1]
    return F.interpolate(x, shape, mode = mode, align_corners = False)

# projection and lora

class Project(Module):
    def __init__(
        self,
        dim_in,
        dim_out
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim_out, dim_in) * (dim_in ** -0.5))

    def forward(self, x):
        return einsum('b c ..., d c -> b d ...', x, self.weight)

class LoRA(Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        dim_lora = 8
    ):
        super().__init__()
        self.down = nn.Parameter(torch.randn(dim_lora, dim_in) * (dim_in ** -0.5))
        self.up = nn.Parameter(torch.zeros(dim_out, dim_lora))

    def forward(self, x):
        lora = einsum('b c ..., l c -> b l ...', x, self.down)
        return einsum('b l ..., d l -> b d ...', lora, self.up)

class XAttnRes(Module):
    def __init__(
        self,
        dim: int,
        *,
        dim_ins: tuple[int, ...] | None = None,
        heads: int = 1,
        pooling: str = 'max',
        dim_lora: int | None = 8,
        channel_first: bool = True
    ):
        super().__init__()
        assert divisible_by(dim, heads), f'dimension {dim} must be divisible by heads {heads}'

        self.scale = (dim // heads) ** -0.5
        self.heads = heads
        self.pooling = pooling
        self.channel_first = channel_first

        dim_ins = default(dim_ins, ())
        self.projs = ModuleList([Project(d, dim) if d != dim else nn.Identity() for d in dim_ins])

        self.has_lora = exists(dim_lora) and dim_lora > 0
        if self.has_lora:
            self.norms_in = ModuleList([RMSNorm(d) for d in dim_ins])
            self.loras = ModuleList([LoRA(d, dim, dim_lora = dim_lora) for d in dim_ins])

        self.norm = RMSNorm(dim)
        self.pseudo_query = nn.Parameter(torch.zeros(heads, dim // heads))

    def forward(self, x, history = None):
        if not exists(history) or len(history) == 0:
            return x

        if not self.channel_first:
            x = rearrange(x, 'b ... c -> b c ...')
            history = tree_map_tensor(lambda t: rearrange(t, 'b ... c -> b c ...'), history)

        assert len(history) == len(self.projs), f'history must have {len(self.projs)} entries, but received {len(history)}'

        spatial_shape = x.shape[2:]

        values, keys = [], []

        for ind, h in enumerate(history):
            h = resize_to(h, spatial_shape, pooling = self.pooling)

            value = self.projs[ind](h)
            values.append(value)

            key = value
            if self.has_lora:
                key = key + self.loras[ind](self.norms_in[ind](h))

            keys.append(self.norm(key))

        values.append(x)
        keys.append(self.norm(x))

        values = rearrange(values, 'n b (h d) ... -> b n h d ...', h = self.heads)
        keys = rearrange(keys, 'n b (h d) ... -> b n h d ...', h = self.heads)

        logits = einsum('b n h d ..., h d -> b n h ...', keys, self.pseudo_query) * self.scale
        attn = logits.softmax(dim = 1)

        out = einsum('b n h ..., b n h d ... -> b h d ...', attn, values)
        out = rearrange(out, 'b h d ... -> b (h d) ...')

        if not self.channel_first:
            out = rearrange(out, 'b c ... -> b ... c')

        return out

# unet

def kernel_and_same_pad(*kernel_size):
    paddings = tuple(map(lambda k: k // 2, kernel_size))
    return dict(kernel_size = kernel_size, padding = paddings)

class XUnet(Module):

    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        frame_kernel_size = 1,
        dim_mults: tuple[int, ...] = (1, 2, 4, 8),
        num_blocks_per_stage: tuple[int, ...] = (2, 2, 2, 2),
        num_self_attn_per_stage: tuple[int, ...] = (0, 0, 0, 1),
        nested_unet_depths: tuple[int, ...] = (0, 0, 0, 0),
        nested_unet_dim = 32,
        channels = 3,
        use_convnext = False,
        consolidate_upsample_fmaps = True,
        skip_scale = 2 ** -0.5,
        weight_standardize = False,
        attn_heads: int | tuple[int, ...] = 8,
        attn_dim_head: int | tuple[int, ...] = 32,
        use_xattnres = False,
        xattnres_replace_skips = False,
        xattnres_heads = 1,
        xattnres_pooling = 'max',
        xattnres_dim_lora: int | None = 8,
        xattnres_kwargs: dict | None = None
    ):
        super().__init__()

        self.train_as_images = frame_kernel_size == 1

        self.skip_scale = skip_scale
        self.channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv3d(channels, init_dim, **kernel_and_same_pad(frame_kernel_size, 7, 7))

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.downs = ModuleList([])
        self.ups = ModuleList([])

        num_resolutions = len(in_out)

        # resnet or convnext

        if use_convnext:
            blocks = partial(ConvNextBlock, frame_kernel_size = frame_kernel_size)
        else:
            blocks = partial(
                ResnetBlock,
                weight_standardize = weight_standardize,
                frame_kernel_size = frame_kernel_size
            )

        # whether to use nested unet, as in unet squared paper

        nested_unet_depths = cast_tuple(nested_unet_depths, num_resolutions)

        # number of blocks per stage

        num_blocks_per_stage = cast_tuple(num_blocks_per_stage, num_resolutions)
        assert all([num_blocks > 0 for num_blocks in num_blocks_per_stage])

        # number of self attention blocks per stage

        num_self_attn_per_stage = cast_tuple(num_self_attn_per_stage, num_resolutions)
        assert all([num_self_attn_blocks >= 0 for num_self_attn_blocks in num_self_attn_per_stage])

        # attn kwargs

        attn_heads = cast_tuple(attn_heads, num_resolutions)
        attn_dim_head = cast_tuple(attn_dim_head, num_resolutions)

        # modules for all layers

        skip_dims = []

        down_stage_parameters = [
            in_out,
            nested_unet_depths,
            num_blocks_per_stage,
            num_self_attn_per_stage,
            attn_heads,
            attn_dim_head
        ]

        up_stage_parameters = [reversed(params[:-1]) for params in down_stage_parameters]

        # xattnres setup

        assert not (xattnres_replace_skips and not use_xattnres), 'xattnres_replace_skips can only be True if use_xattnres is also True'

        self.use_xattnres = use_xattnres
        self.xattnres_replace_skips = xattnres_replace_skips

        self.down_xattnres = None
        self.mid_xattnres = None
        self.up_xattnres = None

        if use_xattnres:
            xattnres_kwargs = dict(
                heads = xattnres_heads,
                pooling = xattnres_pooling,
                dim_lora = xattnres_dim_lora,
                **default(xattnres_kwargs, {})
            )

            self.down_xattnres = ModuleList([])
            self.up_xattnres = ModuleList([])
            history_dims = []

            for dim_in, _ in in_out:
                self.down_xattnres.append(
                    XAttnRes(
                        dim_in,
                        dim_ins = tuple(history_dims),
                        **xattnres_kwargs
                    )
                )
                history_dims.append(dim_in)

            mid_dim = dims[-1]
            self.mid_xattnres = XAttnRes(
                mid_dim,
                dim_ins = tuple(history_dims),
                **xattnres_kwargs
            )
            history_dims.append(mid_dim)

            for _, dim_out in reversed(in_out[:-1]):
                self.up_xattnres.append(
                    XAttnRes(
                        dim_out,
                        dim_ins = tuple(history_dims),
                        **xattnres_kwargs
                    )
                )
                history_dims.append(dim_out)

        # downs

        for ind, ((dim_in, dim_out), nested_unet_depth, num_blocks, self_attn_blocks, heads, dim_head) in enumerate(zip(*down_stage_parameters)):
            is_last = ind >= (num_resolutions - 1)
            skip_dims.append(dim_in)

            self.downs.append(ModuleList([
                blocks(
                    dim_in,
                    dim_in,
                    nested_unet_depth = nested_unet_depth,
                    nested_unet_dim = nested_unet_dim
                ),
                ModuleList([
                    blocks(
                        dim_in,
                        dim_in,
                        nested_unet_depth = nested_unet_depth,
                        nested_unet_dim = nested_unet_dim
                    )
                    for _ in range(num_blocks - 1)
                ]),
                ModuleList([
                    TransformerBlock(
                        dim_in,
                        depth = self_attn_blocks,
                        heads = heads,
                        dim_head = dim_head
                    )
                    for _ in range(self_attn_blocks)
                ]),
                Downsample(dim_in, dim_out)
            ]))

        # middle

        mid_dim = dims[-1]
        mid_nested_unet_depth = nested_unet_depths[-1]

        self.mid = blocks(
            mid_dim,
            mid_dim,
            nested_unet_depth = mid_nested_unet_depth,
            nested_unet_dim = nested_unet_dim
        )
        self.mid_attn = Attention(
            mid_dim,
            heads = attn_heads[-1],
            dim_head = attn_dim_head[-1]
        )
        self.mid_after = blocks(
            mid_dim,
            mid_dim,
            nested_unet_depth = mid_nested_unet_depth,
            nested_unet_dim = nested_unet_dim
        )

        self.mid_upsample = Upsample(mid_dim, dims[-2])

        # ups

        for ind, ((dim_in, dim_out), nested_unet_depth, num_blocks, self_attn_blocks, heads, dim_head) in enumerate(zip(*up_stage_parameters)):
            is_last = ind >= (num_resolutions - 1)
            skip_dim = 0 if xattnres_replace_skips else skip_dims.pop()

            self.ups.append(ModuleList([
                blocks(
                    dim_out + skip_dim,
                    dim_out,
                    nested_unet_depth = nested_unet_depth,
                    nested_unet_dim = nested_unet_dim
                ),
                ModuleList([
                    blocks(
                        dim_out,
                        dim_out,
                        nested_unet_depth = nested_unet_depth,
                        nested_unet_dim = nested_unet_dim
                    )
                    for _ in range(num_blocks - 1)
                ]),
                ModuleList([
                    TransformerBlock(
                        dim_out,
                        depth = self_attn_blocks,
                        heads = heads,
                        dim_head = dim_head
                    )
                    for _ in range(self_attn_blocks)
                ]),
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
            nn.Conv3d(dim, out_dim, **kernel_and_same_pad(frame_kernel_size, 3, 3))
        )

    def forward(self, x):
        is_image = x.ndim == 4

        # validations

        assert not (is_image and not self.train_as_images), 'you specified a frame kernel size for the convolutions in this unet, but you are passing in images'
        assert not (not is_image and self.train_as_images), 'you specified no frame kernel size dimension, yet you are passing in a video. fold the frame dimension into the batch'

        # cast images to 1 framed video

        if is_image:
            x = rearrange(x, 'b c h w -> b c 1 h w')

        # initial convolution

        x = self.init_conv(x)

        # residual

        r = x.clone()

        # downs and ups

        down_hiddens = []
        up_hiddens = []
        history = []

        for ind, (init_block, blocks, attn_blocks, downsample) in enumerate(self.downs):
            if self.use_xattnres:
                x = self.down_xattnres[ind](x, history)

            x = init_block(x)

            for block in blocks:
                x = block(x)

            for attn_block in attn_blocks:
                x = attn_block(x)

            if self.use_xattnres:
                history.append(x)

            down_hiddens.append(x)
            x = downsample(x)

        if self.use_xattnres:
            x = self.mid_xattnres(x, history)

        x = self.mid(x)
        x = self.mid_attn(x) + x
        x = self.mid_after(x)

        if self.use_xattnres:
            history.append(x)

        up_hiddens.append(x)
        x = self.mid_upsample(x)

        for ind, (init_block, blocks, attn_blocks, upsample) in enumerate(self.ups):
            if self.use_xattnres:
                x = self.up_xattnres[ind](x, history)

            if not self.xattnres_replace_skips:
                x = torch.cat((x, down_hiddens.pop() * self.skip_scale), dim=1)

            x = init_block(x)

            for block in blocks:
                x = block(x)

            for attn_block in attn_blocks:
                x = attn_block(x)

            if self.use_xattnres:
                history.append(x)

            up_hiddens.insert(0, x)
            x = upsample(x)

        # consolidate feature maps

        x = self.consolidator(x, up_hiddens)

        # final residual

        x = torch.cat((x, r), dim = 1)

        # final convolution

        out = self.final_conv(x)

        if is_image:
            out = rearrange(out, 'b c 1 h w -> b c h w')

        return out

# RSU

class PixelShuffleUpsample(Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        scale_factor = 2
    ):
        super().__init__()
        self.scale_squared = scale_factor ** 2
        dim_out = default(dim_out, dim)
        conv = nn.Conv3d(dim, dim_out * self.scale_squared, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            Rearrange('b (c r s) f h w -> b c f (h r) (w s)', r = scale_factor, s = scale_factor)
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
        x = self.net(x)
        return x

class NestedResidualUnet(Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_inner = 32,
        frame_kernel_size = 1,
        add_residual = False,
        skip_scale = 2 ** -0.5,
        weight_standardize = False
    ):
        super().__init__()

        self.depth = depth
        self.downs = ModuleList([])
        self.ups = ModuleList([])

        conv = WeightStandardizedConv3d if weight_standardize else nn.Conv3d

        for ind in range(depth):
            is_first = ind == 0
            dim_in = dim if is_first else dim_inner

            down = nn.Sequential(
                conv(
                    dim_in,
                    dim_inner,
                    (1, 4, 4),
                    stride = (1, 2, 2),
                    padding = (0, 1, 1)
                ),
                RMSNorm(dim_inner),
                nn.SiLU()
            )

            up = nn.Sequential(
                PixelShuffleUpsample(2 * dim_inner, dim_in),
                RMSNorm(dim_in),
                nn.SiLU()
            )

            self.downs.append(down)
            self.ups.append(up)

        self.mid = nn.Sequential(
            conv(dim_inner, dim_inner, **kernel_and_same_pad(frame_kernel_size, 3, 3)),
            RMSNorm(dim_inner),
            nn.SiLU()
        )

        self.skip_scale = skip_scale
        self.add_residual = add_residual

    def forward(self, x, residual = None):
        if self.add_residual:
            residual = default(residual, x.clone())

        *_, h, w = x.shape

        layers = len(self.ups)

        for dim_name, size in (('height', h), ('width', w)):
            assert divisible_by(size, 2 ** layers), f'{dim_name} dimension {size} must be divisible by {2 ** layers} ({layers} layers in nested unet)'
            assert (size % (2 ** self.depth)) == 0, f'the unet has too much depth for the image {dim_name} ({size}) being passed in'

        # hiddens

        hiddens = []

        # unet

        for down in self.downs:
            x = down(x)
            hiddens.append(x.clone().contiguous())

        x = self.mid(x)

        for up in reversed(self.ups):
            x = torch.cat((x, hiddens.pop() * self.skip_scale), dim = 1)
            x = up(x)

        # adding residual

        if self.add_residual:
            x = x + residual
            x = F.silu(x)

        return x
