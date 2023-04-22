from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from lightning.pytorch.callbacks import DeviceStatsMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.tuner import Tuner
from omegaconf import DictConfig

from data.datamodule import SimpleDataModule
from model.model import SimpleModel


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    model = SimpleModel(cfg)
    datamodule = SimpleDataModule(cfg)
    callbacks = []

    if cfg.train.checkpoint:
        callbacks.append(
            ModelCheckpoint(
                dirpath=to_absolute_path("checkpoints"),
                filename="{epoch}-{val_loss:.3f}",
                monitor="val_loss",
                save_top_k=3,
                mode="min",
                save_weights_only=True,
            )
        )

    if cfg.train.monitor:
        callbacks.append(DeviceStatsMonitor())
    logger = WandbLogger(
        project=cfg.train.project,
        save_dir=to_absolute_path("logs"),
    )
    trainer = Trainer(
        accelerator="auto",
        accumulate_grad_batches=cfg.train.acc,
        callbacks=callbacks,
        detect_anomaly=True,
        devices="auto",
        fast_dev_run=cfg.train.fast_dev_run,
        logger=[logger],
        num_sanity_val_steps=2,
    )
    tuner = Tuner(trainer)

    if cfg.train.auto_lr:
        tuner.lr_find(model=model, datamodule=datamodule)

    if cfg.train.auto_batch:
        tuner.scale_batch_size(model=model, datamodule=datamodule)

    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)

    onnx_dir = Path("onnx")
    onnx_dir.mkdir(parents=True, exist_ok=True)
    model.to_onnx(
        file_path=to_absolute_path(str(onnx_dir / "model.onnx")),
        export_params=True,
    )


if __name__ == "__main__":
    main()
