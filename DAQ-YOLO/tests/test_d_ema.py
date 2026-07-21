import torch

from daq_yolo.modules.d_ema import D_EMA


def test_dema_shape_and_gradient() -> None:
    module = D_EMA(64, G_max=8, T_c=4)
    x = torch.randn(2, 64, 20, 20, requires_grad=True)
    output = module(x)
    assert output.shape == x.shape
    output.mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert module.alpha.grad is not None


def test_dema_group_count_is_valid_divisor() -> None:
    module = D_EMA(30, G_max=8, T_c=4)
    assert 30 % module.groups == 0
    assert module.groups <= min(8, 30 // 4)
