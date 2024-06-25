from typing import *

import torch
import torch.nn as nn

class BlockNet(nn.Module):
    def __init__(self,
                 input_dim: int = 12,
                 output_dim: int = 4,
                 hidden_dim: int = 192,
                 encoder_kernel_size: int = 5,
                 ):
        super(BlockNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.encoder_kernel_size = encoder_kernel_size
        self.blockNetLayer = BlockNetLayer(hidden_dim = hidden_dim)
        self.encoder = nn.Conv1d(in_channels = self.input_dim,out_channels = self.hidden_dim,kernel_size = self.encoder_kernel_size,stride = 1,padding = "same")
        self.decoder = nn.Linear(in_features = self.hidden_dim, out_features = self.output_dim)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, F, T, C = x.shape
        x = self.encoder(x.reshape(B * F, T, C).permute(0, 2, 1)).permute(0, 2, 1)
        C = x.shape[2]
        x.reshape(B, F, T, C)
        x = self.blockNetLayer(x)
        x = self.decoder(x)
        return x.contiguous()

class BlockNetLayer(nn.Module):
    def __init__(self,
                 hidden_dim: int = 96
                 ):
        super(BlockNetLayer, self).__init__()
        self.hidden_dim = hidden_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #  Todo
        return x

if __name__ == '__main__':
    x = torch.randn((1, 129, 251, 12))  #.cuda() # 251 = 4 second; 129 = 8 kHz; 257 = 16 kHz
    blockNet = BlockNet(
        input_dim = 12,
        output_dim = 4,
        hidden_dim = 96,
        encoder_kernel_size = 5
    ) 
    import time
    ts = time.time()
    y = blockNet(x)
    te = time.time()
    print(blockNet)
    print(y.shape)
    print(te - ts)

    # blockNet = blockNet.to('meta')
    # x = x.to('meta')
    from torch.utils.flop_counter import FlopCounterMode # requires torch>=2.1.0
    with FlopCounterMode(blockNet, display=True) as fcm:
        y = blockNet(x)
        flops_forward_eval = fcm.get_total_flops()
        res = y.sum()
        res.backward()
        flops_backward_eval = fcm.get_total_flops() - flops_forward_eval

    params_eval = sum(param.numel() for param in blockNet.parameters())
    print(f"flops_forward={flops_forward_eval/1e9:.2f}G, flops_back={flops_backward_eval/1e9:.2f}G, params={params_eval/1e6:.2f} M")