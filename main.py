import dependency as d
import lightning_module as m
from packaging.version import Version

import model
from begin.model.BernNet import BernNet

import pytorch_lightning as l
import torch
import torch.nn as nn

class TrainModule(l.LightningModule):

    name: str
    import_path: str = 'main.TrainModule'

    def __init__(self, 
                 model: nn.Module
                 ):
        super().__init__()
        self.model = BernNet(
            input_dim = 12,
            output_dim = 4,
            hidden_dim = 96,
            time_conv_dim = 192,
            encoder_kernel_size = 5,
            num_layers = 6,
            num_freqs = 129,
            num_heads = (2, 2),
            dropout = (0, 0, 0)
        )
        args = locals().copy()

        if compile:
            print("Compiling the model!")
            assert Version(torch.__version__) >= Version(
                '2.0.0'), torch.__version__
            self.model = torch.compile(self.model)

        for k, v in args.items():
            if k == 'self' or k == '__class__' or hasattr(self, k):
                continue
            setattr(self, k, v)
    
    def forward(self, x):

        

        return self.model(x)

if __name__ == '__main__':
    print(model)