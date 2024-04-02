import pytorch_lightning as pl
from tqdm import tqdm
from transformers.utils import logging

from zlie.grte import Grte


def grte():
    pl.seed_everything(42)
    from datasets.dataset_grte import GRTEDataModule
    from config.args_grte import get_arguments
    args, dataModule = get_arguments()
    model = Grte(args)
    trainer = pl.Trainer(
        gpus=1,
        num_sanity_val_steps=0,
        progress_bar_refresh_rate=1,
        check_val_every_n_epoch=1,
        limit_val_batches=0.01,
        limit_train_batches=0.01
    )
    trainer.fit(model, dataModule)
    pass

def baseline():
    pl.seed_everything(42)
    from config.args_baseline import get_arguments
    from datasets.dataset_baseline import BaselineDataModule
    from zlie.baseline import Baseline
    args = get_arguments()
    dataModule = BaselineDataModule(args)
    args.t_total = len(dataModule.train_dataloader()) * args.num_train_epochs
    model = Baseline(args)


    # model = model.load_from_checkpoint('./lightning_logs/version_0/checkpoints/epoch=0-step=13749.ckpt', config=args)
    ### 在真正跑模型之前，一定要把数据集验证一遍
    # for batch in tqdm(dataModule.train_dataloader()):
    #     pass
    # for batch in tqdm(dataModule.val_dataloader()):
    #     pass
    trainer = pl.Trainer(
        gpus=1,
        num_sanity_val_steps=0,
        progress_bar_refresh_rate=1,
        check_val_every_n_epoch=1,
        # limit_val_batches=0.1,
        # limit_train_batches=0.005,
        max_epochs=args.num_train_epochs
    )
    trainer.fit(model, dataModule)
    pass
if __name__ == '__main__':
    # grte()
    baseline()