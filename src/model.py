"""Model architectures and sampling algorithms for HiCES and baseline methods."""

import os
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import inception_v3
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from einops import rearrange
import numpy as np

from omegaconf import OmegaConf

os.environ.setdefault("TORCH_HOME", ".cache/torch")

# ---------------------------------------------------------------------------
# Building blocks -----------------------------------------------------------
# ---------------------------------------------------------------------------

def _conv(ic, oc, k=3, s=1, p=1):
    return nn.Conv2d(ic, oc, k, s, p, bias=False)


class ResidualBlock(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.gn1 = nn.GroupNorm(8, ic)
        self.act = nn.SiLU()
        self.c1 = _conv(ic, oc)
        self.gn2 = nn.GroupNorm(8, oc)
        self.c2 = _conv(oc, oc)
        self.skip = _conv(ic, oc, k=1, p=0) if ic != oc else nn.Identity()

    def forward(self, x):
        h = self.c1(self.act(self.gn1(x)))
        h = self.c2(self.act(self.gn2(h)))
        return h + self.skip(x)


class Down(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.res = ResidualBlock(ic, oc)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        x = self.res(x)
        # Only pool if spatial dimensions are > 1 to avoid 0x0 output
        if x.shape[2] > 1 and x.shape[3] > 1:
            x = self.pool(x)
        return x


class Up(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.block = ResidualBlock(ic, oc)

    def forward(self, x, target_size=None):
        # Conditionally upsample based on target size if provided
        if target_size is not None:
            if x.shape[2:] != target_size:
                x = F.interpolate(x, size=target_size, mode="nearest")
        else:
            x = self.up(x)
        return self.block(x)


class RiskHead(nn.Module):
    def __init__(self, in_c: int, out_c: int = 1):
        super().__init__()
        self.c = nn.Conv2d(in_c, out_c, 1)

    def forward(self, f):
        return self.c(f)


# ---------------------------------------------------------------------------
# UNet ----------------------------------------------------------------------
# ---------------------------------------------------------------------------

class HiCESUNet(nn.Module):
    """Memory-friendly U-Net plus risk-prediction head."""

    def __init__(self, cfg):
        super().__init__()
        ch = int(cfg.base_channels)
        mults = list(cfg.channel_mult)
        self.tile = 8  # leaf tile size for certificate

        self.in_conv = _conv(3, ch)
        self.d1 = Down(ch, ch * mults[0])
        self.d2 = Down(ch * mults[0], ch * mults[1])
        self.d3 = Down(ch * mults[1], ch * mults[2])
        self.d4 = Down(ch * mults[2], ch * mults[3])

        mid_c = ch * mults[3]
        self.mid = ResidualBlock(mid_c, mid_c)

        self.u4 = Up(mid_c, ch * mults[2])
        self.u3 = Up(ch * mults[2], ch * mults[1])
        self.u2 = Up(ch * mults[1], ch * mults[0])
        self.u1 = Up(ch * mults[0], ch)
        self.out_conv = nn.Sequential(nn.GroupNorm(8, ch), nn.SiLU(), _conv(ch, 3))

        self.risk_head = RiskHead(ch * mults[0], cfg.risk_head.out_channels)

    def forward(self, x, *, return_risk: bool = False):
        x0 = self.in_conv(x)
        x1 = self.d1(x0)
        x2 = self.d2(x1)
        x3 = self.d3(x2)
        x4 = self.d4(x3)
        h = self.mid(x4)
        # Use target sizes from skip connections to guide upsampling
        h = self.u4(h, target_size=x3.shape[2:])
        if h.shape[2:] == x3.shape[2:]:
            h = h + x3
        h = self.u3(h, target_size=x2.shape[2:])
        if h.shape[2:] == x2.shape[2:]:
            h = h + x2
        h = self.u2(h, target_size=x1.shape[2:])
        risk_feats = h
        if h.shape[2:] == x1.shape[2:]:
            h = h + x1
        h = self.u1(h, target_size=x0.shape[2:])
        if h.shape[2:] == x0.shape[2:]:
            out = self.out_conv(h + x0)
        else:
            out = self.out_conv(h)
        if return_risk:
            risk_map = self.risk_head(risk_feats)
            return out, risk_map
        return out


class BaselineUNet(HiCESUNet):
    """Same architecture but risk head is disabled."""

    def __init__(self, cfg):
        super().__init__(cfg)
        for p in self.risk_head.parameters():
            p.requires_grad_(False)
        nn.init.constant_(self.risk_head.c.weight, 0.0)
        if self.risk_head.c.bias is not None:
            nn.init.constant_(self.risk_head.c.bias, 0.0)

    def forward(self, x, *, return_risk: bool = False):  # type: ignore
        if return_risk:
            out = super().forward(x, return_risk=False)
            dummy = torch.zeros(x.size(0), 1, x.size(2) // self.tile, x.size(3) // self.tile, device=x.device)
            return out, dummy
        return super().forward(x, return_risk=False)


# ---------------------------------------------------------------------------
# Certificate helpers -------------------------------------------------------
# ---------------------------------------------------------------------------

def _spectral_risk(img: torch.Tensor, patch: int = 8) -> torch.Tensor:
    B, C, H, W = img.shape
    patches = img.unfold(2, patch, patch).unfold(3, patch, patch)  # B,C,h,w,t,t
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, C, patch, patch)
    freq = torch.fft.fft2(patches, norm="ortho")
    mag2 = freq.abs().pow(2)
    risk = mag2.mean(dim=(1, 2, 3)).reshape(B, H // patch, W // patch)
    return risk


# ---------------------------------------------------------------------------
# Tiling utilities ----------------------------------------------------------
# ---------------------------------------------------------------------------

def pack_tiles(img: torch.Tensor, active: torch.Tensor, tile: int) -> Tuple[torch.Tensor, List[Tuple[int, int, int]]]:
    B, C, H, W = img.shape
    h_tiles, w_tiles = H // tile, W // tile
    assert active.shape == (B, h_tiles, w_tiles)

    tiles: List[torch.Tensor] = []
    indices: List[Tuple[int, int, int]] = []
    for b in range(B):
        for i in range(h_tiles):
            for j in range(w_tiles):
                if active[b, i, j]:
                    y0, y1 = i * tile, (i + 1) * tile
                    x0, x1 = j * tile, (j + 1) * tile
                    tiles.append(img[b, :, y0:y1, x0:x1])
                    indices.append((b, i, j))
    if tiles:
        return torch.stack(tiles), indices
    return torch.empty(0, img.size(1), tile, tile, device=img.device), []


def scatter_tiles(tile_tensor: torch.Tensor, base: torch.Tensor, indices: List[Tuple[int, int, int]], tile: int):
    for k, (b, i, j) in enumerate(indices):
        y0, y1 = i * tile, (i + 1) * tile
        x0, x1 = j * tile, (j + 1) * tile
        base[b, :, y0:y1, x0:x1] = tile_tensor[k]


# ---------------------------------------------------------------------------
# Diffusion training loss ---------------------------------------------------
# ---------------------------------------------------------------------------

def diffusion_loss_fn(model: HiCESUNet, x0: torch.Tensor, cfg) -> Dict[str, torch.Tensor]:
    B = x0.size(0)
    device = x0.device
    T = 1000
    betas = torch.linspace(1e-4, 0.02, T, device=device)
    alphas = 1.0 - betas
    alphas_cum = torch.cumprod(alphas, dim=0)

    t = torch.randint(0, T, (B,), device=device)
    a_bar = alphas_cum[t][:, None, None, None]
    noise = torch.randn_like(x0)
    x_t = (a_bar.sqrt() * x0) + ((1 - a_bar).sqrt() * noise)

    pred_eps, risk_pred = model(x_t, return_risk=True)
    mse = F.mse_loss(pred_eps, noise)

    gt_risk = _spectral_risk(x_t, patch=model.tile)
    risk_pred_up = F.interpolate(risk_pred, size=gt_risk.shape[-2:], mode="nearest").squeeze(1)
    risk_loss = F.l1_loss(torch.log(risk_pred_up + 1e-8), torch.log(gt_risk + 1e-8))

    lam_h = float(cfg.training.loss_weights.lambda_H)
    loss = mse + lam_h * risk_loss

    tau = float(cfg.model.risk_certificate.tau_max)
    gt_bin = gt_risk <= tau
    pred_bin = risk_pred_up <= tau
    tp = (gt_bin & pred_bin).sum()
    fp = (~gt_bin & pred_bin).sum()
    tn = (~gt_bin & ~pred_bin).sum()
    fn = (gt_bin & ~pred_bin).sum()

    return {"loss": loss, "mse": mse, "risk": risk_loss, "tp": tp, "fp": fp, "tn": tn, "fn": fn}


# ---------------------------------------------------------------------------
# Samplers ------------------------------------------------------------------
# ---------------------------------------------------------------------------

def uniform_sampler(model: HiCESUNet, x: torch.Tensor, cfg):
    S = int(cfg.training.s_max)
    scheduler = DPMSolverMultistepScheduler(
        beta_start=1e-4, beta_end=0.02, beta_schedule="linear", num_train_timesteps=1000, prediction_type="epsilon"
    )
    scheduler.set_timesteps(S, device=x.device)
    for t in scheduler.timesteps:
        eps = model(x)
        x = scheduler.step(eps, t, x).prev_sample
    return x, S, 0.0


def hierarchical_sampler(model: HiCESUNet, x: torch.Tensor, cfg):
    """Hierarchical certificate-guided sampler with real tile-level pruning."""
    S_max = int(cfg.training.s_max)
    tile = model.tile
    tau = float(cfg.model.risk_certificate.tau_max)

    scheduler = DPMSolverMultistepScheduler(
        beta_start=1e-4, beta_end=0.02, beta_schedule="linear", num_train_timesteps=1000, prediction_type="epsilon"
    )
    scheduler.set_timesteps(S_max, device=x.device)

    B, _, H, W = x.shape
    active = torch.ones(B, H // tile, W // tile, dtype=torch.bool, device=x.device)
    per_tile_calls = torch.zeros_like(active, dtype=torch.float32)

    for t in scheduler.timesteps:
        # Pack active tiles -------------------------------------------------
        packed, indices = pack_tiles(x, active, tile)
        if packed.numel() == 0:
            break  # everything frozen

        eps_tiles, risk_tiles = model(packed, return_risk=True)

        # Scatter epsilon back into full tensor ----------------------------
        eps_full = torch.zeros_like(x)
        scatter_tiles(eps_tiles, eps_full, indices, tile)

        # DPM-Solver step on full tensor -----------------------------------
        x = scheduler.step(eps_full, t, x).prev_sample

        # Evaluate risk & update mask --------------------------------------
        risk_scalar = risk_tiles.mean(dim=(1, 2, 3))  # per packed tile
        for k_idx, (b, i, j) in enumerate(indices):
            if risk_scalar[k_idx] <= tau:
                active[b, i, j] = False
        per_tile_calls += active.float()

    avg_calls = per_tile_calls.mean().item()
    cert_ratio = (~active).float().mean().item()
    return x, avg_calls, cert_ratio


def afder_sampler(model: HiCESUNet, x: torch.Tensor, cfg):
    eps_target = float(cfg.model.acceleration.eps_target)
    target_budget = int(cfg.model.acceleration.target_step_budget)

    scheduler = DPMSolverMultistepScheduler(
        beta_start=1e-4, beta_end=0.02, beta_schedule="linear", num_train_timesteps=1000, prediction_type="epsilon"
    )
    scheduler.set_timesteps(target_budget, device=x.device)

    cum_error = 0.0
    step_used = 0
    for t in scheduler.timesteps:
        eps = model(x)
        x_next = scheduler.step(eps, t, x).prev_sample
        step_error = (eps ** 2).mean().item()
        cum_error += step_error
        x = x_next
        step_used += 1
        if cum_error <= eps_target:
            break
    return x, step_used, 0.0


# ---------------------------------------------------------------------------
# Metrics -------------------------------------------------------------------
# ---------------------------------------------------------------------------

@torch.inference_mode()
def compute_val_metrics(model: HiCESUNet, loader, cfg, device, *, method: str, num_images: int = 64) -> Dict[str, float]:
    model.eval()
    real_imgs, gen_imgs, steps_list, certs = [], [], [], []
    sampler_map = {
        "hices": hierarchical_sampler,
        "proposed": hierarchical_sampler,
        "afder": afder_sampler,
        "comparative": afder_sampler,
        "baseline": uniform_sampler,
        "uniform": uniform_sampler,
    }
    sampler_fn = sampler_map.get(method.lower(), uniform_sampler)

    it = iter(loader)
    while len(real_imgs) < num_images:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        for img in batch:
            if len(real_imgs) >= num_images:
                break
            real_imgs.append(img)
            noise = torch.randn_like(img)
            x_noisy = img + noise
            x_gen, steps, cert = sampler_fn(model, x_noisy.unsqueeze(0).to(device), cfg)
            gen_imgs.append(x_gen.squeeze(0).cpu())
            steps_list.append(float(steps))
            certs.append(float(cert))

    real = torch.stack(real_imgs).to(device)
    fake = torch.stack(gen_imgs).to(device)

    inc = _get_inception(device)
    resize = transforms.Resize((299, 299), antialias=True)

    def _feats(xb):
        xb = resize(xb).clamp(0, 1)
        return inc(xb).cpu()

    feats_r = _feats(real)
    feats_f = _feats(fake)

    mu_r, mu_f = feats_r.mean(0), feats_f.mean(0)
    cov_r = torch.from_numpy(np.cov(feats_r.numpy(), rowvar=False))
    cov_f = torch.from_numpy(np.cov(feats_f.numpy(), rowvar=False))
    covmean = _matrix_sqrt(cov_r @ cov_f + 1e-6 * torch.eye(2048))
    fid = (mu_r - mu_f).pow(2).sum() + torch.trace(cov_r + cov_f - 2 * covmean)

    return {
        "val_fid": float(fid.item()),
        "avg_a_steps": float(np.mean(steps_list)),
        "cert_ratio": float(np.mean(certs)),
    }


# ---------------------------------------------------------------------------
# Aux -----------------------------------------------------------------------
# ---------------------------------------------------------------------------

def _get_inception(device):
    m = inception_v3(pretrained=True, aux_logits=True, transform_input=False).to(device)
    m.fc = nn.Identity()
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _matrix_sqrt(cov: torch.Tensor):
    eigval, eigvec = torch.linalg.eigh(cov)
    eigval = torch.clamp(eigval, min=0)
    sqrt_diag = torch.diag(torch.sqrt(eigval))
    return eigvec @ sqrt_diag @ eigvec.t()


# ---------------------------------------------------------------------------
# Optuna helper -------------------------------------------------------------
# ---------------------------------------------------------------------------

def apply_best_optuna_params(cfg, trial):
    params = trial.params if hasattr(trial, "params") else trial

    def _assign(path: List[str], value):
        node = cfg
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value

    for k, v in params.items():
        if "." in k:
            _assign(k.split("."), v)
            continue
        for sect in (
            "training",
            "training.loss_weights",
            "model",
            "model.risk_certificate",
            "model.acceleration",
        ):
            if OmegaConf.select(cfg, f"{sect}.{k}") is not None:
                OmegaConf.update(cfg, f"{sect}.{k}", v, merge=False)
                break