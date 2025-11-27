"""Main orchestrator – spawns `src.train` as a subprocess so that each Hydra
run is isolated while adhering to the CLI required by the specification."""

import sys
import subprocess
import hydra


@hydra.main(config_path="../config", config_name="config")
def main(cfg):
    run_id = cfg.run.run_id if "run" in cfg and cfg.run is not None else cfg.run_id

    # Build CLI overrides --------------------------------------------------
    overrides = [
        f"run={run_id}",
        f"results_dir={cfg.results_dir}",
        f"mode={cfg.mode}",
    ]
    if cfg.mode == "trial":
        overrides += [
            "wandb.mode=disabled",
            "optuna.n_trials=0",
            "training.epochs=1",
            "training.steps_per_epoch=2",
        ]
    elif cfg.mode == "full":
        overrides += ["wandb.mode=online"]
    else:
        raise ValueError("mode must be 'trial' or 'full'")

    cmd = [sys.executable, "-u", "-m", "src.train", *overrides]
    print("EXEC:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()