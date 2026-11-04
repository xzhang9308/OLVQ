"""Portable train, evaluate, entropy-head fitting and real-codec entry points."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from torch.nn import functional as F
from .model import ModelConfig, MODELS, ROUTES, build_model
from .core.lattice_cells import clear_training_rate, training_rate_bits

LAMBDAS = {1: .0018, 2: .0035, 4: .013, 6: .0483}

def seed_all(seed):
    random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False

def files(path):
    p = Path(path)
    found = [p] if p.is_file() else sorted(x for x in p.rglob('*') if x.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp'))
    if not found: raise ValueError('No images found: ' + str(p))
    return found

def image(path):
    with Image.open(path) as im: return pil_to_tensor(im.convert('RGB')).float().div(255).unsqueeze(0)

def crop(x, size):
    h, w = x.shape[-2:]
    if h < size or w < size: raise ValueError('Training images must be at least the crop size; images are not resized')
    top, left = random.randrange(h-size+1), random.randrange(w-size+1)
    return x[..., top:top+size, left:left+size]

def save(path, state):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp'); torch.save(state, tmp); tmp.replace(path)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def load_model(path, device):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state.get('format') != 'olvq-release-v1': raise ValueError('Expected a release checkpoint, not a research checkpoint')
    model = build_model(ModelConfig(**state['config']), device=device)
    # CompressAI resizes its entropy CDF buffers when loading the backbone.
    model.net.load_state_dict({k[4:]: v for k, v in state['state_dict'].items() if k.startswith('net.')})
    model.load_state_dict(state['state_dict'], strict=True)
    return model, state

def train(args):
    seed_all(args.seed); paths = files(args.data)
    rd_lambda = args.rd_lambda if args.rd_lambda is not None else LAMBDAS[args.quality]
    if not math.isfinite(rd_lambda) or rd_lambda <= 0: raise ValueError('--rd-lambda must be finite and positive')
    if args.crop % 64: raise ValueError('--crop must be divisible by 64')
    if args.resume:
        model, old = load_model(args.resume, args.device)
        settings = old['training']
        if settings != dict(steps=args.steps, batch_size=args.batch_size, crop=args.crop, seed=args.seed, rd_lambda=rd_lambda, data=[str(p.resolve()) for p in paths]):
            raise ValueError('Resume must preserve steps, batch, crop, seed and input file list')
    else:
        cfg = ModelConfig(args.model, args.quality, args.route, train_samples=args.train_samples, eval_samples=args.eval_samples)
        model = build_model(cfg, pretrained=not args.random_init, source=args.source, device=args.device); old = None
    optimizer = model.configure_optimizer()
    start = 0
    if old:
        optimizer.load_state_dict(old['optimizer']); start = old['step']
        random.setstate(old['python_rng']); torch.set_rng_state(old['torch_rng'])
        if args.device.startswith('cuda') and old.get('cuda_rng') is not None: torch.cuda.set_rng_state_all(old['cuda_rng'])
    if model.family == 'tcm' and (args.crop < 256 or args.crop % 128): raise ValueError('TCM crop must be >=256 and divisible by 128')
    model.train()
    for step in range(start, args.steps):
        fraction = step / max(1, args.steps-1)
        for group in optimizer.param_groups: group['lr'] = group['base_lr'] * (.1 + .9 * .5 * (1 + math.cos(math.pi * fraction)))
        optimizer.zero_grad(set_to_none=True); loss_sum = 0.
        # Microbatch 1 keeps the effective pixel budget independent of GPU memory.
        for _ in range(args.batch_size):
            x = crop(image(random.choice(paths)), args.crop).to(args.device)
            clear_training_rate(model); out = model(x); pixels = x.shape[0]*x.shape[-2]*x.shape[-1]
            rate = (training_rate_bits(model) - out['likelihoods']['z'].double().log2().sum()) / pixels
            loss = rate + rd_lambda * 255**2 * (out['x_hat']-x).square().mean()
            if not bool(torch.isfinite(loss)): raise RuntimeError('Nonfinite training loss')
            (loss / args.batch_size).backward(); loss_sum += float(loss.detach()) / args.batch_size
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
        if not bool(torch.isfinite(norm)): raise RuntimeError('Nonfinite gradient')
        optimizer.step()
        print(json.dumps(dict(step=step+1, rd_loss=loss_sum)), flush=True)
        if (step+1) % args.save_every == 0 or step+1 == args.steps:
            state = dict(format='olvq-release-v1', config=asdict(model.config), step=step+1,
                state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                training=dict(steps=args.steps,batch_size=args.batch_size,crop=args.crop,seed=args.seed,rd_lambda=rd_lambda,data=[str(p.resolve()) for p in paths]),
                python_rng=random.getstate(),torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if args.device.startswith('cuda') else None)
            save(Path(args.output)/('step-%d.pt'%(step+1)),state)

@torch.no_grad()
def evaluate(args):
    model, _ = load_model(args.checkpoint, args.device); model.eval()
    if args.samples:
        for cell in model.cells:
            for key in ('mc_samples','samples','voronoi_mc_samples'):
                if hasattr(cell,key): setattr(cell,key,args.samples)
    rows = []
    for p in files(args.data):
        x = image(p).to(args.device); h,w = x.shape[-2:]; unit = 128 if model.family == 'tcm' else 64
        minimum = 256 if model.family == 'tcm' else 64
        ph, pw = max(minimum, math.ceil(h/unit)*unit), max(minimum, math.ceil(w/unit)*unit)
        padded=F.pad(x,(0,pw-w,0,ph-h))
        out=model(padded); recon=out['x_hat'][...,:h,:w].clamp(0,1)
        mse=float((recon-x).square().mean()); bits=sum(float(-v.double().log2().sum()) for v in out['likelihoods'].values())
        rows.append(dict(image=p.name,bpp=bits/(h*w),psnr=-10*math.log10(max(mse,1e-15))))
        print(json.dumps(rows[-1]),flush=True)
    result=dict(rate_kind='numerical_probability_estimate',images=len(rows),means={k:sum(r[k] for r in rows)/len(rows) for k in ('bpp','psnr')})
    Path(args.output).parent.mkdir(parents=True,exist_ok=True);Path(args.output).write_text(json.dumps(result,indent=2)+'\n')

def codec_modules():
    from .core import lattice_fast_codec as codec
    return codec

def fit_head(args):
    seed_all(args.seed); model,_=load_model(args.checkpoint,args.device);model.eval();codec=codec_modules()
    for p in model.parameters():p.requires_grad_(False)
    # Each backbone uses its own decoder-known context traversal.
    head=codec.create_entropy_head(model,hidden=64,embedding_dim=8)
    pieces=[]; paths=files(args.data)
    for i in range(args.images):
        x=crop(image(paths[i%len(paths)]),args.crop).to(args.device)
        with torch.no_grad(): captured=codec.capture_lattice_examples(model,x,verify_reference=True)
        pieces.append({k:captured[k].detach().cpu() for k in ('symbols','log_scales','group_ids')})
        print('capture',i+1,args.images,flush=True)
    data={k:torch.cat([p[k] for p in pieces]).to(args.device) for k in pieces[0]}
    symbols=codec.wire_symbols(model,data['symbols']); opt=torch.optim.Adam(head.parameters(),lr=1e-3)
    for step in range(args.steps):
        ids=torch.randint(len(data['log_scales']),(args.batch_vectors,),device=args.device)
        sym=symbols[ids] if torch.is_tensor(symbols) else type(symbols)(*(v[ids] for v in symbols))
        opt.zero_grad(set_to_none=True)
        loss=head.nll_bits(sym,data['log_scales'][ids],data['group_ids'][ids]).mean()
        if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite entropy-head loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(head.parameters(),10.)
        if not bool(torch.isfinite(norm)):raise RuntimeError('Nonfinite entropy-head gradient')
        opt.step()
        if (step+1)%100==0:print('fit',step+1,float(loss.detach()),flush=True)
    save(args.output,dict(format='olvq-head-v1',model_sha256=digest(args.checkpoint),device_type=torch.device(args.device).type,
        configuration=head.configuration(),state_dict=head.cpu().state_dict(),steps=args.steps,seed=args.seed))

def open_codec(args):
    model,_=load_model(args.checkpoint,args.device);model.eval();module=codec_modules()
    saved=torch.load(args.head,map_location='cpu',weights_only=False)
    if saved.get('format')!='olvq-head-v1' or saved['model_sha256']!=digest(args.checkpoint):raise ValueError('Head does not match model checkpoint')
    if saved['device_type']!=torch.device(args.device).type:raise ValueError('Use the same CPU/CUDA profile as head fitting')
    head=module.create_entropy_head(model,hidden=64,embedding_dim=8)
    if head.configuration()!=saved['configuration']:raise ValueError('Head architecture mismatch')
    head.load_state_dict(saved['state_dict'],strict=True);head.eval()
    identity=digest(args.checkpoint)+digest(args.head)
    return module.LatticeFastCodec(model,head,identity,threads=args.threads)

@torch.no_grad()
def encode(args):
    codec=open_codec(args);x=image(args.input).to(args.device)
    stream,_=codec.compress(x,collect_stats=True);recon,_=codec.decompress(stream)
    ref=codec.estimate(x,integer_cdf=False,verify_reference=True)
    if not torch.equal(recon,ref['x_hat']):raise RuntimeError('Decoded reconstruction differs')
    Path(args.output).parent.mkdir(parents=True,exist_ok=True);Path(args.output).write_bytes(stream)
    print(json.dumps(dict(actual_bpp=len(stream)*8/(x.shape[-2]*x.shape[-1]),bytes=len(stream))))

@torch.no_grad()
def decode(args):
    codec=open_codec(args);x,_=codec.decompress(Path(args.input).read_bytes())
    arr=(x[0].clamp(0,1).mul(255).round().byte().permute(1,2,0).cpu().numpy())
    Path(args.output).parent.mkdir(parents=True,exist_ok=True);Image.fromarray(arr).save(args.output)

def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    for name in ('train','evaluate','fit-head','encode','decode'):
        s=sub.add_parser(name);s.add_argument('--device',default='cpu');s.add_argument('--threads',type=int,default=1)
        s.add_argument('--output',required=True)
        if name!='train':s.add_argument('--checkpoint',required=True)
        if name in ('train','evaluate','fit-head'):s.add_argument('--data',required=True)
        if name in ('encode','decode'):
            s.add_argument('--head',required=True);s.add_argument('--input',required=True)
        if name in ('train','fit-head'):
            s.add_argument('--seed',type=int,default=42);s.add_argument('--steps',type=int,default=10000 if name=='train' else 6000)
            s.add_argument('--crop',type=int,default=256)
        if name=='train':
            s.add_argument('--model',choices=MODELS,default='cheng2020-attn');s.add_argument('--route',choices=ROUTES,default='e8')
            s.add_argument('--quality',type=int,choices=(1,2,4,6),default=2);s.add_argument('--source');s.add_argument('--resume');s.add_argument('--rd-lambda',type=float)
            s.add_argument('--random-init',action='store_true',help='Only for functional testing, not pretrained adaptation')
            s.add_argument('--batch-size',type=int,default=16);s.add_argument('--save-every',type=int,default=500)
            s.add_argument('--train-samples',type=int,default=8192);s.add_argument('--eval-samples',type=int,default=8192)
        if name=='evaluate':s.add_argument('--samples',type=int)
        if name=='fit-head':s.add_argument('--images',type=int,default=160);s.add_argument('--batch-vectors',type=int,default=4096)
    a=p.parse_args()
    for key in ('threads','steps','crop','batch_size','save_every','images','batch_vectors','samples'):
        value=getattr(a,key,None)
        if value is not None and value<=0:p.error('--'+key.replace('_','-')+' must be positive')
    torch.set_num_threads(a.threads);seed_all(getattr(a,'seed',42))
    {'train':train,'evaluate':evaluate,'fit-head':fit_head,'encode':encode,'decode':decode}[a.command](a)

if __name__=='__main__':main()
