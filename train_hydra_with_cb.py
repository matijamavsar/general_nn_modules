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

    # (Optional) Define whether you want the trainer to
    # early_stop or create checkpoints based on some metric
    #######################################################
    METRIC = 'val_error'
    MODE = 'min'
    print('----------------------')
    print('Using metric:', METRIC, '\nmode:', MODE)
    print('----------------------')
    #######################################################

    # If using metric, define the callbacks
    early_stop_callback = EarlyStopping(
        monitor=METRIC, min_delta=0.00,
        patience=100, verbose=False,
        mode=MODE)
    checkpoint_callback = ModelCheckpoint(
        monitor=METRIC, mode=MODE,
        filename='model_best_' + METRIC,
        save_last=True)

    # If using callbacks, include them in the Trainer object
    trainer = Trainer(**cfg.pl_trainer, callbacks=[
        early_stop_callback, 
        checkpoint_callback
        ], logger=False,
        strategy=None, accelerator="gpu",
        )
    trainer.fit(network, train_dl, val_dl)
    print('Best checkpoint path:', checkpoint_callback.best_model_path)

    return 0

if __name__ == "__main__":
    main()
