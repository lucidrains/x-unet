import pytest
import torch
from x_unet import XUnet, XAttnRes

param = pytest.mark.parametrize

@param('dim_lora', (8, None))
@param('heads', (1, 2, 4))
def test_xattnres(dim_lora, heads):
    xattnres = XAttnRes(dim = 64, dim_ins = (32, 64, 128), heads = heads, dim_lora = dim_lora)

    x = torch.randn(2, 64, 32, 32)
    history = [
        torch.randn(2, 32, 64, 64),
        torch.randn(2, 64, 32, 32),
        torch.randn(2, 128, 16, 16)
    ]

    out = xattnres(x, history)
    assert out.shape == x.shape

def test_xattnres_heads_divisibility():
    with pytest.raises(AssertionError):
        XAttnRes(dim = 32, heads = 3)

def test_xattnres_uniform_at_init():
    xattnres = XAttnRes(64, dim_ins = (64, 64), heads = 2)

    x = torch.randn(2, 64, 16, 16)
    history = [torch.randn(2, 64, 16, 16), torch.randn(2, 64, 16, 16)]

    out = xattnres(x, history)
    assert torch.allclose(out, (x + history[0] + history[1]) / 3, atol = 1e-5)

def test_xattnres_empty_history():
    xattnres = XAttnRes(32)

    x = torch.randn(2, 32, 16, 16)
    assert torch.equal(xattnres(x), x)
    assert torch.equal(xattnres(x, []), x)

def test_xattnres_1d():
    xattnres = XAttnRes(32, dim_ins = (16, 32))

    x = torch.randn(2, 32, 64)
    history = [torch.randn(2, 16, 128), torch.randn(2, 32, 64)]

    assert xattnres(x, history).shape == x.shape

def test_xattnres_3d():
    xattnres = XAttnRes(32, dim_ins = (16, 64), pooling = 'avg')

    x = torch.randn(2, 32, 4, 16, 16)
    history = [torch.randn(2, 16, 4, 32, 32), torch.randn(2, 64, 4, 8, 8)]

    assert xattnres(x, history).shape == x.shape

@param('channel_first', (True, False))
def test_xattnres_channel_first_and_last(channel_first):
    xattnres = XAttnRes(32, dim_ins = (16, 64), channel_first = channel_first)

    if channel_first:
        x = torch.randn(2, 32, 16, 16)
        history = [torch.randn(2, 16, 32, 32), torch.randn(2, 64, 8, 8)]
    else:
        x = torch.randn(2, 16, 16, 32)
        history = [torch.randn(2, 32, 32, 16), torch.randn(2, 8, 8, 64)]

    assert xattnres(x, history).shape == x.shape

@param('replace_skips', (False, True))
def test_xunet_2d_with_xattnres(replace_skips):
    unet = XUnet(dim = 16, dim_mults = (1, 2, 4, 8), use_xattnres = True, xattnres_replace_skips = replace_skips)

    img = torch.randn(2, 3, 64, 64)
    out = unet(img)
    assert out.shape == img.shape

def test_xunet_3d_with_xattnres():
    unet = XUnet(dim = 16, dim_mults = (1, 2, 4, 8), frame_kernel_size = 3, use_xattnres = True)

    video = torch.randn(2, 3, 4, 64, 64)
    out = unet(video)
    assert out.shape == video.shape

@param('channels, out_dim', ((3, 3), (1, 8), (3, 1)))
def test_xunet_out_dim(channels, out_dim):
    unet = XUnet(dim = 16, channels = channels, out_dim = out_dim, dim_mults = (1, 2, 4, 8), use_xattnres = True)

    img = torch.randn(2, channels, 64, 64)
    out = unet(img)
    assert out.shape == (2, out_dim, 64, 64)

def test_xunet_convnext_with_xattnres():
    unet = XUnet(dim = 16, dim_mults = (1, 2, 4, 8), use_convnext = True, use_xattnres = True, xattnres_heads = 2, xattnres_pooling = 'avg')

    img = torch.randn(2, 3, 64, 64)
    out = unet(img)
    assert out.shape == img.shape

def test_xunet_weight_standardize():
    unet = XUnet(dim = 16, dim_mults = (1, 2, 4, 8), weight_standardize = True, use_xattnres = True)

    img = torch.randn(2, 3, 64, 64)
    out = unet(img)
    assert out.shape == img.shape

def test_xunet_nested_unets():
    unet = XUnet(dim = 16, dim_mults = (1, 2, 4, 8), nested_unet_depths = (2, 2, 1, 0), use_xattnres = True)

    img = torch.randn(2, 3, 64, 64)
    out = unet(img)
    assert out.shape == img.shape

def test_xunet_without_xattnres():
    unet = XUnet(dim = 16, dim_mults = (1, 2, 4, 8))

    img = torch.randn(2, 3, 64, 64)
    out = unet(img)
    assert out.shape == img.shape

def test_xattnres_backward():
    xattnres = XAttnRes(32, dim_ins = (16, 32), heads = 2, dim_lora = 4)

    x = torch.randn(2, 32, 16, 16, requires_grad = True)
    history = [
        torch.randn(2, 16, 32, 32, requires_grad = True),
        torch.randn(2, 32, 8, 8, requires_grad = True)
    ]

    out = xattnres(x, history)
    out.sum().backward()

    assert x.grad is not None
    assert all(h.grad is not None for h in history)
    assert xattnres.pseudo_query.grad is not None

@param('resize_fmap_before', (True, False))
def test_feature_map_consolidator(resize_fmap_before):
    from x_unet.x_unet import FeatureMapConsolidator
    c = FeatureMapConsolidator(16, dim_ins = (32, 64), dim_outs = (16, 16), resize_fmap_before = resize_fmap_before)

    x = torch.randn(2, 16, 1, 32, 32)
    fmaps = [torch.randn(2, 32, 1, 16, 16), torch.randn(2, 64, 1, 8, 8)]
    out = c(x, fmaps)
    assert out.shape == (2, 48, 1, 32, 32)
