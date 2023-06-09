import logging
import hydra
import sys
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, ValueNode

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.plugins import DDPPlugin

logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> Trainer:
    logger.info(f"Training with the following config:\n{OmegaConf.to_yaml(cfg)}")
     
    data = instantiate(cfg.db)['db']
    train_dl = data.train_dataloader()
    val_dl = data.val_dataloader()

    try:
        cfg.network.network.args.extend = cfg.db.db.args.extend
    except Exception:
        pass

    network = instantiate(cfg.network)['network']

    # If using callbacks, include them in the Trainer object
    trainer = Trainer(**cfg.pl_trainer,
        logger=False,
        strategy=None, accelerator="gpu",
        )
    trainer.fit(network, train_dl, val_dl)

    return 0

if __name__ == "__main__":
    main()
