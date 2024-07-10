import pytorch_lightning as l
import torch
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader
import model.BernNet

import os

import datamodule.data_set as data_set

class DataModule(l.LightningDataModule):
    
     # Initializer
    def __init__(self, data_dir: str = '/workspaces/Container/begin/data_set', 
                 batch_size_train: int = 2, 
                 batch_size_test: int = 1, 
                 num_workers: int = 8, 
                 ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        self.num_workers = num_workers
        self.dataset = data_set

    # prepare data
    # must be implemented in the subclass
    def prepare_data(self) -> None:
        return super().prepare_data()

    # setup data
    # must be implemented in the subclass
    def setup(self, stage: str):
        print(stage)
        if stage == "fit":
            self.dataset_train = self.dataset(
                data_dir = os.path.join(self.data_dir, "train"),
                num_data = 1000,
            )
            self.dataset_val = self.dataset(
                data_dir = os.path.join(self.data_dir, "dev"),
                num_data = 998,
            )
        elif stage == "test":
            self.dataset_test = self.dataset(
                data_dir = os.path.join(self.data_dir, "test"),
                num_data = 1000
            )

    # train dataloaders settings
    # must be implemented in the subclass
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.dataset_train,
            batch_size = self.batch_size_train,
            shuffle = True,
            num_workers = self.num_workers,
            pin_memory = False
        )

    # evaluation dataloaders settings
    # must be implemented in the subclass
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.dataset_val,
            batch_size = self.batch_size_test,
            shuffle = False,
            num_workers = self.num_workers,
            pin_memory = False
        )

    # test dataloaders settings
    # must be implemented in the subclass
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.dataset_test,
            batch_size = self.batch_size_test,
            shuffle = False,
            num_workers = self.num_workers,
            pin_memory = False
        )

if __name__ == '__main__':
    # cli = l.LightningCLI(
    #     TrainModule,
    #     cfg.data_m,
    #     seed_everything_default=1744,
    #     save_config_kwargs={'overwrite': True},
    #     # parser_kwargs={"default_config_files": ["config/default.yaml"],
    #     #    "parser_mode": "omegaconf"
    #     #    },
    # )
    print(model.BernNet)