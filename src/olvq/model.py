"""Six-backbone public model API. E8 is the default route."""
from dataclasses import asdict, dataclass
from pathlib import Path
import torch
from torch import nn
from compressai import zoo
from .core.backbone_adapter import RotationLIC, ENTROPY, AR
from .core.lattice_cells import GeometryRateMixin, Nonorthogonal8Cell, SubsampledOrthogonal8Cell
from .core.sampled_cells import SampledRotationCell, CentralNonorthogonal8Cell, _reconstruct_only

MODELS = ('bmshj2018-hyperprior', 'mbt2018-mean', 'mbt2018', 'cheng2020-anchor', 'cheng2020-attn', 'tcm')
ROUTES = ('e8', 'orthogonal', 'nonorthogonal')

@dataclass
class ModelConfig:
    model: str = 'cheng2020-attn'
    quality: int = 2
    route: str = 'e8'
    train_samples: int = 8192
    eval_samples: int = 8192
    train_chunk: int = 8
    eval_chunk: int = 64
    rate_positions: int = 4
    train_order: int = 48
    eval_order: int = 64

    def validate(self):
        if self.model not in MODELS or self.route not in ROUTES:
            raise ValueError('Unsupported model or route')
        if self.quality not in (1, 2, 4, 6):
            raise ValueError('Supported quality identifiers: 1, 2, 4, 6')
        for k, v in asdict(self).items():
            if k not in ('model', 'route') and (not isinstance(v, int) or isinstance(v, bool) or v < 1):
                raise ValueError('Expected positive integer: ' + k)

class OLVQModel(GeometryRateMixin, RotationLIC):
    def __init__(self, net, config):
        nn.Module.__init__(self); config.validate()
        self.config = config; self.net = net; self.family = config.model
        self.route = config.route; self.variant = 'joint'
        self.mc = config.eval_samples; self.mc_seed = 104729
        self.train_chunk, self.eval_chunk = config.train_chunk, config.eval_chunk
        widths = ([net.M // net.num_slices] * net.num_slices if self.family == 'tcm'
                  else [net.N if self.family.startswith('cheng') else net.M])
        cells = []
        for width in widths:
            if self.route == 'e8':
                cell = SampledRotationCell(width, net.gaussian_conditional, mc=config.eval_samples,
                    train_mc=config.train_samples, chunk=config.train_chunk, rate_positions=config.rate_positions)
            elif self.route == 'orthogonal':
                cell = SubsampledOrthogonal8Cell(width, net.gaussian_conditional, mc=config.eval_samples,
                    train_mc=config.train_samples, chunk=config.train_chunk)
            else:
                cell = CentralNonorthogonal8Cell(width, net.gaussian_conditional,
                    samples=config.eval_samples, train_samples=config.train_samples, chunk=config.train_chunk,
                    group_proposals=True, train_order=config.train_order, eval_order=config.eval_order,
                    train_proposal_chains=64, train_proposal_sweeps=16,
                    proposal_chains=128, proposal_sweeps=64, direction_chunk=2048)
            cell.rate_positions = config.rate_positions; cell.training_rate_terms = []; cells.append(cell)
        self.cells = nn.ModuleList(cells)
        self._capture_enabled = False; self._captured = []
        for p in self.parameters(): p.requires_grad_(False)

    def train(self, mode=True):
        nn.Module.train(self, mode)
        for cell in self.cells: cell.integration_chunk = self.train_chunk if mode else self.eval_chunk
        self.net.eval(); self.net.entropy_bottleneck.train(mode)
        return self

    def quantize(self, y, s, m, index=0, mask=None):
        if self._capture_enabled:
            if self.training or mask is not None: raise ValueError('Capture requires unmasked evaluation')
            self._captured.append(tuple(v.detach() for v in (y, s, m)))
            return _reconstruct_only(self.cells[index], y, s, m), torch.ones_like(y, dtype=torch.float64)
        q, p = super().quantize(y, s, m, index, mask)
        return q, p.double()

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 3 or any(n % 64 for n in x.shape[-2:]):
            raise ValueError('Expected NCHW RGB with height/width divisible by 64; CLI pads evaluation images')
        if self.family == 'tcm':
            if any(n < 256 or n % 128 for n in x.shape[-2:]):
                raise ValueError('TCM requires dimensions >=256 and divisible by 128')
            return self._slices(x)
        n = self.net; y = n.g_a(x)
        zh, zp = self._z(y, absolute=self.family == 'bmshj2018-hyperprior', ste=self.training)
        params = n.h_s(zh)
        if self.family in AR: yh, yp = self._ar(y, params)
        else:
            if self.family == 'bmshj2018-hyperprior': s, m = params, torch.zeros_like(params)
            else: s, m = params.chunk(2, 1)
            yh, yp = self.quantize(y, s, m)
        return {'x_hat': n.g_s(yh), 'likelihoods': {'y': yp, 'z': zp}}

    def configure_optimizer(self, geometry_lr=1e-4, scale_lr=1e-5, entropy_lr=3e-6, boundary_lr=1e-6):
        for p in self.parameters(): p.requires_grad_(False)
        groups = {'geometry': [], 'scale': [], 'entropy': [], 'boundary': []}
        for cell in self.cells:
            if self.route == 'e8':
                groups['geometry'].append(cell.lattice_shape.base); groups['scale'].append(cell.quantizer.log_scale)
            elif self.route == 'orthogonal':
                groups['geometry'].append(cell.rotation_params); groups['scale'].append(cell.log_scale)
            else:
                groups['geometry'].append(cell.basis.shape); groups['scale'].append(cell.basis.log_step)
        for name, p in self.net.named_parameters():
            if name.startswith(ENTROPY + ('lrp_transforms.',)) and not name.endswith('.quantiles'):
                groups['entropy'].append(p)
        for block in (self.net.g_a[-1], self.net.g_s[0]): groups['boundary'].extend(block.parameters())
        ids = [id(p) for ps in groups.values() for p in ps]
        if len(ids) != len(set(ids)): raise RuntimeError('Optimizer groups overlap')
        rates = dict(geometry=geometry_lr, scale=scale_lr, entropy=entropy_lr, boundary=boundary_lr)
        for ps in groups.values():
            for p in ps: p.requires_grad_(True)
        return torch.optim.Adam([dict(params=ps, lr=rates[k], base_lr=rates[k], name=k) for k, ps in groups.items() if ps])

def build_model(config=None, *, pretrained=False, source=None, device='cpu'):
    config = config or ModelConfig(); config.validate()
    if config.model == 'tcm':
        if pretrained and source is None: raise ValueError('TCM requires --source with the author checkpoint for the selected quality')
        from .vendor.tcm import TCM
        net = TCM(N=64)
    else:
        net = zoo.image_models[config.model](quality=config.quality, metric='mse', pretrained=pretrained and source is None)
    if source is not None:
        state = torch.load(Path(source), map_location='cpu', weights_only=False)
        state = state.get('state_dict', state)
        state = {k.removeprefix('module.'): v for k, v in state.items()}
        net.load_state_dict(state)  # Both CompressAI and TCM default to strict loading.
    return OLVQModel(net, config).to(device)
