"""Actual bitstreams, source-free subprocess decoding, exact tensor comparisons."""
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
import pytest
import torch
from olvq.model import MODELS, ROUTES, ModelConfig, build_model
from olvq.cli import seed_all, save, digest, open_codec
from types import SimpleNamespace
from olvq.core.lattice_fast_codec import create_entropy_head, capture_lattice_examples, wire_symbols

DECODE_SCRIPT = '''
import sys, torch
from types import SimpleNamespace
from olvq.cli import open_codec, seed_all
seed_all(42)
torch.set_num_threads(1)
checkpoint,head,stream,out,device=sys.argv[1:]
codec=open_codec(SimpleNamespace(checkpoint=checkpoint,head=head,device=device,threads=1))
x,stats=codec.decompress(open(stream,'rb').read())
torch.save(dict(x=x.cpu(),y=stats['y_hat'].cpu(),z=stats['z_hat'].cpu(),symbols=stats['symbol_sha256']),out)
'''

@pytest.mark.parametrize('quality', [2, 6])
@pytest.mark.parametrize('backbone', MODELS)
@pytest.mark.parametrize('route', ROUTES)
def test_real_codec_matrix(tmp_path, backbone, route, quality):
    torch.set_num_threads(1); seed_all(42)
    device = os.environ.get('OLVQ_TEST_DEVICE', 'cpu')
    model = build_model(ModelConfig(model=backbone, route=route, quality=quality, train_samples=16, eval_samples=16),device=device).eval()
    # Unequal learned geometries test slice/group bindings, not just the identity lattice.
    with torch.no_grad():
        # Random untrained encoders otherwise often emit only zero lattice symbols.
        # Exercise signed nonzero coordinates without using out-of-range input pixels.
        final = [m for m in model.net.g_a if isinstance(m, torch.nn.Conv2d)][-1]
        assert isinstance(final, torch.nn.Conv2d) and final.bias is not None
        final.bias.add_(torch.linspace(-2., 2., len(final.bias), device=device))
        for i, cell in enumerate(model.cells):
            if route == 'e8':
                cell.lattice_shape.base.add_(.01 * torch.randn_like(cell.lattice_shape.base))
                cell.quantizer.log_scale.add_(.015 * (i+1))
            elif route == 'orthogonal':
                cell.rotation_params.add_(.01 * torch.randn_like(cell.rotation_params))
                cell.log_scale.add_(.015 * (i+1))
            else:
                cell.basis.shape.add_(.003 * torch.randn_like(cell.basis.shape))
                cell.basis.log_step.add_(.015 * (i+1))
    checkpoint = tmp_path/'model.pt'
    save(checkpoint,dict(format='olvq-release-v1',config=asdict(model.config),state_dict=model.state_dict()))
    x = torch.rand(1,3,65,91,device=device)
    reference = capture_lattice_examples(model,x,verify_reference=True)
    assert bool(reference['symbols'].ne(0).any()), 'Nonzero symbol coverage required'
    head = create_entropy_head(model,hidden=64,embedding_dim=8)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    logs, groups = reference['log_scales'].to(device), reference['group_ids'].to(device)
    wire = wire_symbols(model,reference['symbols'].to(device))
    for _ in range(2):
        optimizer.zero_grad()
        loss=head.nll_bits(wire,logs,groups).mean()
        assert torch.isfinite(loss)
        loss.backward(); optimizer.step()
    head_file=tmp_path/'head.pt'
    save(head_file,dict(format='olvq-head-v1',model_sha256=digest(checkpoint),device_type=torch.device(device).type,
        configuration=head.configuration(),state_dict=head.cpu().state_dict()))
    codec=open_codec(SimpleNamespace(checkpoint=checkpoint,head=head_file,device=device,threads=1))
    blob,encoded=codec.compress(x,collect_stats=True)
    assert encoded['symbol_sha256'] == reference['symbol_sha256']
    assert torch.equal(encoded['y_hat'],reference['y_hat'])
    assert torch.equal(encoded['z_hat'],reference['z_hat'])
    estimate=codec.estimate(x,integer_cdf=True,verify_reference=True)
    assert abs(encoded['estimated_bits']-estimate['estimated_bits']) < 1e-5
    assert abs(encoded['ideal_bits']-estimate['ideal_bits']) < .01
    (tmp_path/'image.olvq').write_bytes(blob)
    # Child receives only weights, head and bitstream; no source image or reference.
    subprocess.run([sys.executable,'-c',DECODE_SCRIPT,str(checkpoint),str(head_file),str(tmp_path/'image.olvq'),str(tmp_path/'decoded.pt'),device],check=True,timeout=240)
    decoded=torch.load(tmp_path/'decoded.pt',map_location='cpu',weights_only=False)
    assert decoded['symbols'] == reference['symbol_sha256']
    for key,ref in [('x','x_hat'),('y','y_hat'),('z','z_hat')]:
        assert torch.equal(decoded[key],reference[ref].cpu()), (backbone,route,quality,key)
    with pytest.raises(ValueError,match='checksum'):
        codec.decompress(blob[:-1]+bytes([blob[-1]^1]))
