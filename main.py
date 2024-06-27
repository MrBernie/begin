import dependency as d
import lightning_module as m

import model
from model.BlockNet import BlockNet

import pytorch_lightning as l
import torch.nn as nn

class TrainModule(l.LightningModule):
    name: str
    import_path: str = 'main.TrainModule'
    def __init__(self, 
                 model: nn.Module = BlockNet
                 ):
        super().__init__()
        self.model = model
        args = locals().copy()