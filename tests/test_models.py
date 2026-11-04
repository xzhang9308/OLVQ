import pytest
import torch
from olvq.model import MODELS, ROUTES, ModelConfig, build_model
from olvq.core.lattice_cells import training_rate_bits

torch.set_num_threads(1)

@pytest.mark.parametrize('backbone', MODELS)
@pytest.mark.parametrize('route', ROUTES)
def test_forward_backward(backbone, route):
    torch.manual_seed(42)
    model = build_model(ModelConfig(model=backbone, route=route, train_samples=16, eval_samples=16, rate_positions=1))
    optimizer = model.configure_optimizer()
    model.train()
    size = 256 if backbone == 'tcm' else 128
    x = torch.rand(1, 3, size, size)
    out = model(x)
    loss = (out['x_hat'] - x).square().mean() + training_rate_bits(model) / (size * size)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    optimizer.step()
    assert out['x_hat'].shape == x.shape

    # Tiny proposal budget checks execution, not numerical accuracy.
    for cell in model.cells:
        if route == "nonorthogonal":
            cell.proposal_chains = 2
            cell.proposal_sweeps = 1
            cell.eval_order = 8
    model.eval()
    with torch.no_grad():
        output = model(x)
    assert output['x_hat'].shape == x.shape
    assert torch.isfinite(output['x_hat']).all()
    assert all(torch.isfinite(p).all() and (p > 0).all() for p in output['likelihoods'].values())


@pytest.mark.parametrize('backbone', MODELS)
@pytest.mark.parametrize('route', ROUTES)
def test_high_quality_configuration(backbone, route):
    model = build_model(ModelConfig(model=backbone, quality=6, route=route, train_samples=16, eval_samples=16))
    optimizer = model.configure_optimizer()
    assert optimizer.param_groups
    assert model.config.quality == 6
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) > 0
