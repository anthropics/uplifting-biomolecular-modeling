import argparse
import logging
from pathlib import Path

import lightning.pytorch as pl
import lightning.pytorch.callbacks as pl_callbacks
import torch
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig

from atlasfold.train.config import load_config, print_config, save_config, to_dict
from atlasfold.train.monomer.datamodule import TrainingDataModule
from atlasfold.train.monomer.train_module import TrainingModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an AtlasFold monomer model.")
    parser.add_argument(
        "config",
        type=str,
        help="Path to the yaml configuration file.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="Name of the experiment for logging purposes.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        help="Output root directory for saving logs and checkpoints.",
    )
    # Easy overrides
    parser.add_argument(
        "--num_gpus",
        type=int,
        help="Number of GPUs to use for training.",
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        help="Number of nodes to use for training.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        help="Number of workers to use for dataloader.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        help="Path to a checkpoint file to resume training from.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    parser.add_argument(
        "--override",
        type=str,
        nargs="+",
        help="Override configuration options using 'key=value' format.",
    )
    return parser.parse_args()


def parse_config(args) -> DictConfig:
    cfg = load_config(args.config, override_args=args.override)

    # Override some config options with command line args
    if args.out_dir is not None:
        cfg.train.out_dir = args.out_dir
    if args.experiment_name is not None:
        cfg.train.name = args.experiment_name
    if args.num_gpus is not None:
        cfg.train.trainer.devices = args.num_gpus
    if args.num_nodes is not None:
        cfg.train.trainer.num_nodes = args.num_nodes
    if args.num_workers is not None:
        cfg.train.data.num_workers = args.num_workers
    if args.compile:
        cfg.train.compile.enabled = True
    if args.wandb:
        cfg.train.wandb.use = True

    # Apply global_hparams overrides
    apply_global_hparams_overrides(cfg)

    if args.debug:
        # Enable debug mode settings
        print("Debug mode is enabled: Single GPU, 0 workers, no wandb.")
        cfg.train.trainer.devices = 1
        cfg.train.trainer.num_nodes = 1
        cfg.train.trainer.accumulate_grad_batches = 1
        cfg.train.trainer.log_every_n_steps = 1
        cfg.train.trainer.limit_train_batches = 50
        cfg.train.trainer.limit_val_batches = 10
        cfg.train.trainer.enable_checkpointing = True
        cfg.train.data.num_workers = 0
        cfg.train.wandb.use = False

    return cfg


def apply_global_hparams_overrides(cfg: DictConfig) -> None:
    # Estimate number of GPUs
    train_cfg = cfg.train
    if train_cfg.trainer.devices == "auto":
        num_gpus: int = torch.cuda.device_count()
    else:
        num_gpus = cfg.train.trainer.devices
    assert isinstance(num_gpus, int), "num_gpus should be an integer or 'auto'."
    assert num_gpus > 0, "No GPUs available for training."
    world_size: int = train_cfg.trainer.num_nodes * num_gpus

    # Compute parameters dependent on global_hparams
    batch_size: int = train_cfg.data.batch_size
    global_batch_size: int = train_cfg.data.global_batch_size
    if global_batch_size % (batch_size * world_size) != 0:
        raise ValueError(
            f"Global batch size {global_batch_size} is not "
            f"divisible by (batch_size {batch_size} * world_size {world_size})"
        )
    # compute accumulate_grad_batches
    accumulate_grad_batches: int = global_batch_size // (batch_size * world_size)
    train_cfg.trainer.accumulate_grad_batches = accumulate_grad_batches
    # compute limit_train_batches
    train_cfg.trainer.limit_train_batches *= accumulate_grad_batches


def build_trainer(cfg, debug: bool = False) -> pl.Trainer:
    train_cfg = cfg.train
    pl_trainer_cfg = train_cfg.trainer
    save_dir = Path(train_cfg.out_dir) / train_cfg.name

    callbacks = []
    if train_cfg.wandb.use:
        from lightning.pytorch.loggers import WandbLogger

        wandb_logger = WandbLogger(
            name=train_cfg.name,
            project=train_cfg.wandb.project,
            entity=train_cfg.wandb.entity,
            config=to_dict(cfg),
            save_dir=save_dir,
        )
        loggers = [wandb_logger]

        @rank_zero_only
        def _save_config() -> None:
            config_out = Path(wandb_logger.experiment.dir) / "train_config.yaml"
            save_config(cfg, config_out)
            wandb_logger.experiment.save("train_config.yaml")

        _save_config()

    else:
        loggers = None  # use default logger

    # Learning rate monitor
    lr_monitor = pl_callbacks.LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    # Model summary
    model_summary = pl_callbacks.ModelSummary(max_depth=2)
    callbacks.append(model_summary)

    # TQDM
    tqdm_refresh_rate = 1 if debug else cfg.train.trainer.log_every_n_steps
    tqdm_callback = pl_callbacks.TQDMProgressBar(refresh_rate=tqdm_refresh_rate)
    callbacks.append(tqdm_callback)

    if not debug:
        checkpoint_callback = pl_callbacks.ModelCheckpoint(
            monitor="val/top/lddt",
            save_top_k=-1,
            filename="epoch{epoch:04d}_step{step:08d}_lddt{val/rank/lddt:.4f}",
            mode="max",
            auto_insert_metric_name=False,
        )
        callbacks.append(checkpoint_callback)

    trainer = pl.Trainer(
        default_root_dir=save_dir,
        logger=loggers,
        callbacks=callbacks,
        accelerator=pl_trainer_cfg.accelerator,
        strategy=pl_trainer_cfg.strategy,
        devices=pl_trainer_cfg.devices,
        num_nodes=pl_trainer_cfg.num_nodes,
        precision=pl_trainer_cfg.precision,
        max_epochs=pl_trainer_cfg.max_epochs,
        limit_train_batches=pl_trainer_cfg.limit_train_batches,
        limit_val_batches=pl_trainer_cfg.limit_val_batches,
        log_every_n_steps=pl_trainer_cfg.log_every_n_steps,
        enable_checkpointing=pl_trainer_cfg.enable_checkpointing,
        accumulate_grad_batches=pl_trainer_cfg.accumulate_grad_batches,
        gradient_clip_val=pl_trainer_cfg.gradient_clip_val,
        use_distributed_sampler=False,
        benchmark=True,
        # reload_dataloaders_every_n_epochs=1,
    )
    return trainer


def train(args) -> None:
    # To ignore warning
    torch.set_float32_matmul_precision("high")

    cfg = parse_config(args)

    trainer = build_trainer(cfg, args.debug)
    is_global_zero = trainer.is_global_zero

    # Set random seed
    pl.seed_everything(cfg.train.seed, workers=True, verbose=False)

    model_module = TrainingModule(cfg)
    data_module = TrainingDataModule(cfg.train.data)

    # Print config
    if is_global_zero:
        print_config(cfg)

    if cfg.train.load_opt_state:
        trainer.fit(
            model_module,
            datamodule=data_module,
            ckpt_path=args.resume_from_checkpoint,
        )
    else:
        # Manually load the model state dict without optimizer state dict.
        assert args.resume_from_checkpoint is not None, (
            "resume_from_checkpoint must be specified if load_opt_state is False."
        )
        ckpt = torch.load(args.resume_from_checkpoint, map_location="cpu")
        if cfg.train.init_from_ema:
            live_weights = ckpt["state_dict"]
            ema_weights = {k: v for k, v in ckpt["ema"]["params"].items()}
            ema_weights = {f"model.{k}": v for k, v in ema_weights.items()}

            module_group_names = model_module.model.get_module_group_names()

            keys_to_replace = []
            for module_name in cfg.train.init_from_ema:
                assert module_name in module_group_names, (
                    f"Module '{module_name}' not found in model. "
                    f"Available moduels: {module_group_names}"
                )
                for submodule in module_group_names[module_name]:
                    module_to_replace = f"model.{submodule}"
                    keys_to_replace.append(module_to_replace)
            keys_to_replace = tuple(keys_to_replace)
            if is_global_zero:
                logging.info(
                    f"Replacing the following modules with EMA weights: {keys_to_replace}"
                )

            is_replaced = []
            for k, v in ema_weights.items():
                if not k.startswith(keys_to_replace):
                    continue
                is_replaced.append(k)
                live_weights[k] = v
            if is_global_zero:
                logging.info(
                    f"Replaced {len(is_replaced)} keys with EMA weights: {is_replaced}"
                )

        model_module.load_state_dict(ckpt["state_dict"], strict=False)
        model_module.ema.load_state_dict(ckpt["ema"], strict=False)
        model_module.last_lr_step = ckpt["global_step"]
        trainer.fit_loop.load_state_dict(ckpt["loops"]["fit_loop"])
        trainer.fit(model_module, datamodule=data_module)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    args = parse_args()
    train(args)
