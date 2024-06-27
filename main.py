import dependency as d
import lightning_module as m
from packaging.version import Version

import model
from model.BlockNet import BlockNet

import pytorch_lightning as l
import torch
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

        if compile != False:
            assert Version(torch.__version__) >= Version('2.0.0'), torch.__version__
            self.model = torch.compile(model, dynamic=Version(torch.__version__) >= Version('2.1.0'))
        else:
            self.arch = model

if __name__ == '__main__':
    TrainModule = TrainModule()