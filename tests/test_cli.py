"""Isolated process tests: all generated artifacts stay in pytest's temp directory."""
from types import SimpleNamespace
import numpy as np
import torch
import subprocess
import sys
import pytest
from PIL import Image
from olvq.model import MODELS
from olvq.cli import load_model, open_codec, image as read_image, seed_all


def run(*args):
    subprocess.run([sys.executable, '-m', 'olvq.cli', *map(str, args)], check=True, timeout=240)


@pytest.mark.parametrize('backbone', MODELS)
@pytest.mark.parametrize('route', ['e8', 'orthogonal', 'nonorthogonal'])
def test_checkpoint_and_independent_codec(tmp_path, route, backbone):
    image = tmp_path / 'input.png'
    size = 256 if backbone == 'tcm' else 64
    Image.new('RGB', (size, size), (96, 120, 160)).save(image)
    train = tmp_path / 'train'
    run('train', '--data', image, '--output', train, '--route', route,
        '--model', backbone, '--random-init', '--steps', 1, '--crop', size, '--batch-size', 1,
        '--train-samples', 16, '--eval-samples', 16)
    checkpoint = train / 'step-1.pt'
    model, saved = load_model(checkpoint, 'cpu')
    assert saved['step'] == 1 and model.route == route
    head = tmp_path / 'head.pt'
    run('fit-head', '--data', image, '--checkpoint', checkpoint, '--output', head,
        '--steps', 2, '--images', 1, '--crop', size, '--batch-vectors', 16)
    stream = tmp_path / 'image.olvq'
    run('encode', '--input', image, '--output', stream, '--checkpoint', checkpoint, '--head', head)
    decoded = tmp_path / 'decoded.png'
    run('decode', '--input', stream, '--output', decoded, '--checkpoint', checkpoint, '--head', head)
    assert Image.open(decoded).size == (size, size)
    seed_all(42)
    codec = open_codec(SimpleNamespace(checkpoint=checkpoint, head=head, device='cpu', threads=1))
    with torch.no_grad():
        ref = codec.estimate(read_image(image), verify_reference=True)['x_hat']
    expected = ref[0].clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    assert np.array_equal(np.asarray(Image.open(decoded)), expected)
