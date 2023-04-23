import time
from pathlib import Path

import hydra
import torch
import torch._dynamo.config
from lightning import Trainer
from lightning.pytorch.callbacks import (
    DeviceStatsMonitor,
    EarlyStopping,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.tuner import Tuner

from config.config import Config
from data.datamodule import SimpleDataModule
from model.model import SimpleModel


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: Config):
    model = SimpleModel(cfg)
    # compiled_model = torch.compile(model, mode="reduce-overhead")
    compiled_model = model
    datamodule = SimpleDataModule(cfg)
    callbacks = []

    if cfg.train.checkpoint:
        callbacks.append(
            ModelCheckpoint(
                dirpath=Path(f"checkpoints/{int(time.time())}"),
                filename="{epoch}-val_loss={val/loss:.3f}",
                monitor="val/loss",
                save_top_k=3,
                mode="min",
                auto_insert_metric_name=False,
                save_weights_only=True,
            )
        )

    if cfg.train.monitor:
        callbacks.append(DeviceStatsMonitor())

    if cfg.train.early_stop:
        callbacks.append(
            EarlyStopping(
                monitor="val/loss",
                min_delta=0.00,
                patience=3,
                verbose=False,
                mode="min",
            )
        )

    if cfg.train.fast_dev_run:
        logger = TensorBoardLogger(
            save_dir="logs",
            name="fast_dev_run",
            log_graph=True,
        )
    else:
        logger = WandbLogger(project=cfg.train.project, save_dir="logs")

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")

    trainer = Trainer(
        accelerator="auto",
        accumulate_grad_batches=cfg.train.acc,
        callbacks=callbacks,
        detect_anomaly=True,
        devices="auto",
        fast_dev_run=cfg.train.fast_dev_run,
        logger=[logger],
        max_epochs=-1,
        num_sanity_val_steps=2,
        precision=cfg.train.precision,
    )
    tuner = Tuner(trainer)

    if cfg.train.auto_lr and not cfg.train.fast_dev_run:
        tuner.lr_find(model=compiled_model, datamodule=datamodule)

    if cfg.train.auto_batch and not cfg.train.fast_dev_run:
        tuner.scale_batch_size(model=compiled_model, datamodule=datamodule)

    trainer.fit(model=compiled_model, datamodule=datamodule)
    trainer.test(model=compiled_model, datamodule=datamodule)

    onnx_dir = Path("onnx")
    onnx_dir.mkdir(parents=True, exist_ok=True)
    compiled_model.to_onnx(file_path=onnx_dir / "model.onnx", export_params=True)


if __name__ == "__main__":
    torch._dynamo.config.verbose = True
    main()
