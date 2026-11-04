<h1 align="center">OLVQ</h1>

<h3 align="center">
  Learning Optimal Lattice Vector Quantizers<br>
  for End-to-end Neural Image Compression
</h3>

<p align="center">Xi Zhang &nbsp;·&nbsp; Xiaolin Wu</p>
<p align="center"><strong>Official PyTorch Implementation · NeurIPS 2024</strong></p>

<p align="center">
  <a href="https://proceedings.neurips.cc/paper_files/paper/2024/hash/c09916f6f0c428a97a09a53648e5002e-Abstract-Conference.html"><img src="https://img.shields.io/badge/NeurIPS-2024-6846C1?style=flat-square" alt="NeurIPS 2024 Paper"></a>
  <a href="https://arxiv.org/abs/2411.16119"><img src="https://img.shields.io/badge/arXiv-2411.16119-B31B1B?style=flat-square" alt="arXiv:2411.16119"></a>
  <a href="#installation"><img src="https://img.shields.io/badge/PyTorch-2.2.2-EE4C2C?style=flat-square&amp;logo=pytorch&amp;logoColor=white" alt="PyTorch 2.2.2"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-BSD--3--Clause-238636?style=flat-square" alt="BSD 3-Clause License"></a>
</p>

<p align="center">
  <a href="#installation">Installation</a> &nbsp;·&nbsp;
  <a href="#training">Training</a> &nbsp;·&nbsp;
  <a href="#probability-head-and-real-coding">Real Coding</a> &nbsp;·&nbsp;
  <a href="#citation">Citation</a>
</p>

## Features

This release provides training, probability-head fitting, and real image encoding/decoding for three learnable 8D lattice routes:

- **`e8`** (default): E8 initialization with learned rotation and scale.
- **`orthogonal`**: I8 initialization with learned orthogonal geometry and scale.
- **`nonorthogonal`**: near-I8 initialization with a learned nonorthogonal basis and approximate lattice search.

All three routes support **`bmshj2018-hyperprior`**, **`mbt2018-mean`**, **`mbt2018`**, **`cheng2020-anchor`**, **`cheng2020-attn`**, and **`tcm`**.

Training jointly adapts the lattice, entropy model, and encoder/decoder boundary layers. Probability-head fitting freezes the main model and preserves its quantization and reconstruction. This release extends the paper's implementation with the three-route workflow and real coding.

## Installation

Use Python 3.10. From this directory, run:

```bash
python -m pip install -e '.[test]'
olvq --help
```

Dependencies include PyTorch 2.2.2 and CompressAI 1.2.6. For GPU execution, install the appropriate PyTorch CUDA build first. Use `--device cpu` instead of `--device cuda` to run on CPU.

## Training

Provide a directory of training images; subdirectories are scanned automatically. Images are randomly cropped without resizing and must be at least the crop size (default: 256×256).

```bash
olvq train --model cheng2020-attn --route e8 --quality 2 \
  --data /path/to/train-images --steps 10000 \
  --output runs/e8-q2 --device cuda
```

Select another model with `--model`, another route with `--route`, or a quality from `1`, `2`, `4`, `6` with `--quality`. Defaults are Cheng Attention, E8, quality 2, and an effective batch of 16 crops per update. Use `--rd-lambda` to override the quality's rate-distortion weight.

CompressAI pretrained weights are downloaded automatically. TCM requires an author checkpoint with `N=64`, `M=320`, and five slices:

```bash
olvq train --model tcm --source /path/to/tcm-q2.pth --quality 2 \
  --data /path/to/train-images --output runs/tcm-e8-q2 --device cuda
```

Checkpoints are saved every 500 steps and at completion. Resume by adding `--resume runs/e8-q2/step-500.pt` to the original training command, keeping its training settings and input files unchanged.

## Probability head and real coding

Fit one probability head per trained checkpoint, using training images only. The default is 6,000 updates on 160 training crops.

```bash
olvq fit-head --checkpoint runs/e8-q2/step-10000.pt \
  --data /path/to/train-images --output runs/e8-q2/head.pt --device cuda

olvq encode --checkpoint runs/e8-q2/step-10000.pt --head runs/e8-q2/head.pt \
  --input input.png --output image.olvq --device cuda

olvq decode --checkpoint runs/e8-q2/step-10000.pt --head runs/e8-q2/head.pt \
  --input image.olvq --output reconstruction.png --device cuda
```

These commands work for all six models and all three routes. `encode` reports actual file bpp; `decode` requires only the bitstream and matching model/head files, not the original image. Model/head file sizes are excluded from image bpp.

Use the same device profile and software environment for encoding and decoding. Network computation runs on the selected device; entropy coding runs on CPU. The stream format is `LFC2`; older `LFC1` files must be re-encoded. Weights and datasets are not bundled; only load trusted checkpoints.

## Numerical evaluation

To evaluate numerical probability-based bpp and PSNR without producing bitstreams:

```bash
olvq evaluate --checkpoint runs/e8-q2/step-10000.pt \
  --data /path/to/test-images --output results/e8-q2.json --device cuda
```

Numerical integration is slower than the learned probability head; `--samples` controls its budget. Estimated bpp and actual file bpp are different metrics. For comparisons, match image padding: Cheng Attention coding uses symmetric multiples of 128, while numerical evaluation uses right/bottom padding. The other codecs match numerical evaluation; TCM pads to multiples of 128 with a minimum of 256, and the remaining models to multiples of 64.

## Tests and license

```bash
python -m pytest -q
# Optional CUDA codec tests:
OLVQ_TEST_DEVICE=cuda python -m pytest tests/test_codec_matrix.py -q
```

Tests cover all model/route combinations, including independent decoding and exact quantized-symbol and reconstruction checks. They verify functionality, not rate-distortion performance.

Released under [BSD-3-Clause](LICENSE). TCM retains its MIT license; see [third-party notices](NOTICE.md).

## Citation

If you use OLVQ in your research, please cite:

```bibtex
@article{zhang2024learning,
  title={Learning optimal lattice vector quantizers for end-to-end neural image compression},
  author={Zhang, Xi and Wu, Xiaolin},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={106497--106518},
  year={2024}
}
```
