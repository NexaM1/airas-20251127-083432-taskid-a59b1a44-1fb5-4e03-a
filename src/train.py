import os
import sys
import random
import logging
from collections import defaultdict
from typing import Dict

import hydra
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import wandb
import optuna

from src.preprocess import build_dataloaders
from src.model import (
    HiCESUNet,
    BaselineUNet,
    diffusion_loss_fn,
    compute_val_metrics,
    apply_best_optuna_params,
)

# ---------------------------------------------------------------------------
# Helpers -------------------------------------------------------------------
# ---------------------------------------------------------------------------

def _flatten_cfg(cfg):
    """Merge cfg.run.* into the root so that all accesses become flat."""
    if "run" in cfg and cfg.run is not None:
        # Handle both new format (run as string) and legacy format (run as DictConfig)
        if not isinstance(cfg.run, str):
            cfg = OmegaConf.merge(cfg, cfg.run)
    return cfg


def _resolve_device(cfg):
    dev_str = str(cfg.hardware.device)
    if dev_str.lower() in {"auto", "detect", "cuda if torch.cuda.is_available() else cpu"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if "cuda" in dev_str and not torch.cuda.is_available():
        return "cpu"
    return dev_str


# ---------------------------------------------------------------------------
# Logging --------------------------------------------------------------------
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("train")


# ---------------------------------------------------------------------------
# Optuna objective -----------------------------------------------------------
# ---------------------------------------------------------------------------

def _objective(trial: optuna.Trial, base_cfg):
    # Clone & flatten ------------------------------------------------------
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    cfg = _flatten_cfg(cfg)
    apply_best_optuna_params(cfg, trial)

    cfg.training.epochs = 1
    cfg.training.steps_per_epoch = 1
    cfg.training.batch_size = min(2, cfg.training.batch_size)

    # Data -----------------------------------------------------------------
    train_loader, val_loader = build_dataloaders(cfg, for_optuna=True)
    device = torch.device(_resolve_device(cfg))

    # Model + optimiser ----------------------------------------------------
    model = HiCESUNet(cfg.model).to(device)
    optimiser = optim.AdamW(model.parameters(), lr=float(cfg.training.learning_rate))

    batch = next(iter(train_loader)).to(device)
    losses = diffusion_loss_fn(model, batch, cfg)
    optimiser.zero_grad()
    losses["loss"].backward()
    optimiser.step()

    metrics = compute_val_metrics(model, val_loader, cfg, device, method="proposed", num_images=4)
    return metrics["val_fid"]


# ---------------------------------------------------------------------------
# Full training -------------------------------------------------------------
# ---------------------------------------------------------------------------

@hydra.main(config_path="../config", config_name="config")
def main(cfg):  # pylint: disable=too-many-statements
    cfg = _flatten_cfg(cfg)

    # ---------------------------------------------------------------------
    # Mode-specific patching ----------------------------------------------
    # ---------------------------------------------------------------------
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
        cfg.training.epochs = 1
        cfg.training.steps_per_epoch = 2
        cfg.training.batch_size = min(4, cfg.training.batch_size)
        cfg.logging.log_interval = 1
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"
    else:
        raise ValueError("mode must be either 'trial' or 'full'")

    # ---------------------------------------------------------------------
    # Determinism ----------------------------------------------------------
    # ---------------------------------------------------------------------
    torch.manual_seed(int(cfg.training.seed))
    random.seed(int(cfg.training.seed))
    cudnn.deterministic = True
    cudnn.benchmark = False

    # ---------------------------------------------------------------------
    # Optuna hyper-parameter optimisation ---------------------------------
    # ---------------------------------------------------------------------
    if int(cfg.optuna.n_trials) > 0:
        study = optuna.create_study(direction=str(cfg.optuna.direction))
        study.optimize(lambda t: _objective(t, cfg), n_trials=int(cfg.optuna.n_trials))
        log.info("[Optuna] best %.4f with params %s", study.best_value, study.best_trial.params)
        apply_best_optuna_params(cfg, study.best_trial)

    # ---------------------------------------------------------------------
    # WandB initialisation -------------------------------------------------
    # ---------------------------------------------------------------------
    if cfg.wandb.mode != "disabled":
        wandb_run = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=str(cfg.run_id),
            resume="allow",
            mode=cfg.wandb.mode,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        log.info("WandB URL: %s", wandb_run.url)
    else:
        os.environ["WANDB_MODE"] = "disabled"
        wandb_run = None

    # ---------------------------------------------------------------------
    # Data -----------------------------------------------------------------
    # ---------------------------------------------------------------------
    train_loader, val_loader = build_dataloaders(cfg)
    device = torch.device(_resolve_device(cfg))

    # ---------------------------------------------------------------------
    # Model ---------------------------------------------------------------
    # ---------------------------------------------------------------------
    if cfg.method.lower() in {"hices", "proposed"}:
        model = HiCESUNet(cfg.model).to(device)
    else:
        model = BaselineUNet(cfg.model).to(device)

    optimiser = optim.AdamW(
        model.parameters(),
        lr=float(cfg.training.learning_rate),
        betas=tuple(cfg.training.betas),
        weight_decay=float(cfg.training.weight_decay),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimiser,
        T_max=max(1, int(cfg.training.epochs) * int(cfg.training.steps_per_epoch)),
        eta_min=1e-6,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.training.mixed_precision and device.type == "cuda")

    global_step = 0
    confusion_totals: Dict[str, int] = defaultdict(int)
    best_primary_metric = float("inf")  # FID (lower=better)

    for epoch in range(int(cfg.training.epochs)):
        model.train()
        data_it = iter(train_loader)
        for step in range(int(cfg.training.steps_per_epoch)):
            try:
                batch = next(data_it)
            except StopIteration:
                data_it = iter(train_loader)
                batch = next(data_it)
            batch = batch.to(device)

            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                loss_dict = diffusion_loss_fn(model, batch, cfg)
            total_loss = loss_dict["loss"]

            optimiser.zero_grad()
            if scaler.is_enabled():
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimiser)
                nn.utils.clip_grad_norm_(model.parameters(), float(cfg.training.gradient_clip_norm))
                scaler.step(optimiser)
                scaler.update()
            else:
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), float(cfg.training.gradient_clip_norm))
                optimiser.step()
            scheduler.step()

            # accumulate confusion counts ---------------------------------
            for k in ("tp", "fp", "tn", "fn"):
                if k in loss_dict:
                    confusion_totals[k] += int(loss_dict[k])

            # real-time WandB logging -------------------------------------
            if wandb_run and global_step % int(cfg.logging.log_interval) == 0:
                log_dict = {f"train_{k}": float(v) for k, v in loss_dict.items() if k not in {"tp", "fp", "tn", "fn"}}
                log_dict["lr"] = scheduler.get_last_lr()[0]
                wandb.log(log_dict, step=global_step)
            global_step += 1

        # -----------------------------------------------------------------
        # Validation ------------------------------------------------------
        # -----------------------------------------------------------------
        model.eval()
        val_metrics = compute_val_metrics(
            model,
            val_loader,
            cfg,
            device,
            method=cfg.method,
            num_images=64 if cfg.mode == "full" else 8,
        )
        best_primary_metric = min(best_primary_metric, val_metrics["val_fid"])
        if wandb_run:
            wandb.log({f"val_{k}": v for k, v in val_metrics.items()}, step=global_step)

    # ---------------------------------------------------------------------
    # Wrap-up --------------------------------------------------------------
    # ---------------------------------------------------------------------
    if wandb_run:
        wandb_run.summary["best_val_fid"] = best_primary_metric
        for k, v in confusion_totals.items():
            wandb_run.summary[f"confusion_{k}"] = v
        wandb_run.finish()


if __name__ == "__main__":
    main()