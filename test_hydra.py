import logging
import hydra
import os
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> Trainer:
    logger.info(f"Training with the following config:\n{OmegaConf.to_yaml(cfg)}")
     
    data = instantiate(cfg.db)['db']
    test_dl = data.test_dataloader()
    cfg.network.network.args.extend = cfg.db.db.args.extend
    
    network = instantiate(cfg.network)['network']
    
    trainer = Trainer(**cfg.pl_trainer, logger=False)

    trainer.test(network, test_dl)

    return 0

if __name__ == "__main__":
    main()