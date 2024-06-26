from typing import *

import torch
import torch.nn as nn
from torch import Tensor


class BlockNet(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int,
                 time_conv_dim: int,
                 encoder_kernel_size: int,
                 num_freqs: int,
                 num_heads: Tuple[int, int] = (2, 2),
                 dropout: Tuple[int, int, int] = (0, 0, 0)
                 ):
        super(BlockNet, self).__init__()
        self.blockNetLayer = BlockNetLayer(hidden_dim = hidden_dim, time_conv_dim = time_conv_dim, num_freqs = num_freqs, num_heads = num_heads, dropout = dropout)
        self.encoder = nn.Conv1d(in_channels = input_dim,out_channels = hidden_dim,kernel_size = encoder_kernel_size,stride = 1,padding = "same")
        self.decoder = nn.Linear(in_features = hidden_dim, out_features = output_dim)


    def forward(self, x: Tensor) -> Tensor:
        B, F, T, C = x.shape
        x = self.encoder(x.reshape(B * F, T, C).permute(0, 2, 1)).permute(0, 2, 1)
        C = x.shape[2]
        x = x.reshape(B, F, T, C)

        setattr(self.blockNetLayer, "need_weights", False)
        x = self.blockNetLayer(x)

        x = self.decoder(x)
        return x.contiguous()

class BlockNetLayer(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            time_conv_dim: int,
            num_freqs: int,
            num_heads: Tuple,
            dropout: Tuple,
            kernel_size: Tuple[int, int] = (5 ,3),
            conv_groups: Tuple[int, int] = (8, 8)
    ) -> None:
        super().__init__()
        f_conv_groups = conv_groups[0]
        t_conv_groups = conv_groups[1]
        f_kernel_size = kernel_size[0]
        t_kernel_size = kernel_size[1]

        # cross-band block
        # For the frequency convolution
        self.freq_conv_layer_norm= nn.LayerNorm(num_freqs)
        self.freq_conv = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=f_kernel_size,groups=f_conv_groups ,padding='same')
        self.freq_conv_PReLU = nn.PReLU(hidden_dim)

        # For the frequency multi-head self-attention
        self.freq_mhsa_layer_norm = nn.LayerNorm(hidden_dim)
        self.freq_mhsa = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads[0], batch_first=True)
        self.freq_mhsa_dropout = nn.Dropout(dropout[0])

        # For the frequency linear layer
        self.freq_linear_layer_norm = nn.LayerNorm(num_freqs)
        self.freq_linear = nn.Linear(num_freqs, num_freqs)

        # narrow-band block
        # For the time multi-head self-attention
        self.t_mhsa_layer_norm= nn.LayerNorm(hidden_dim)
        self.t_mhsa = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads[1], batch_first=True)
        self.t_mhsa_dropout = nn.Dropout(dropout[1])

        # For the time linear
        self.t_linear_layer_norm = nn.LayerNorm(hidden_dim)
        self.t_linear = nn.Linear(hidden_dim, hidden_dim)
        self.t_linear_silu = nn.SiLU()

        # For the time convolution
        self.t_conv_down = nn.Conv1d(in_channels=hidden_dim, out_channels=time_conv_dim, kernel_size=t_kernel_size,groups=t_conv_groups ,padding='same')
        self.t_conv_forward = nn.Conv1d(in_channels=time_conv_dim, out_channels=time_conv_dim, kernel_size=t_kernel_size,groups=t_conv_groups ,padding='same')
        self.t_conv_layer_norm = nn.GroupNorm(num_groups= t_conv_groups,num_channels=time_conv_dim)
        self.t_conv_up = nn.Conv1d(in_channels=time_conv_dim, out_channels=hidden_dim, kernel_size=t_kernel_size,groups=t_conv_groups ,padding='same')
        self.t_conv_silu = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        x_1 = x
        x_2 = x
        attn = []
        # cross-band block
        x_1 = x + self.frequency_conv(x_1)
        x_1_, attn_output = self.frequency_mhsa(x_1)
        attn.append(attn_output)
        x_1_ = self.frequency_linear(x_1_)
        x_1 = x_1 + x_1_
        x_1 = x + self.frequency_conv(x_1)

        # narrow-band block
        x_2_, attn_output  = self.time_mhsa(x)
        attn.append(attn_output)
        x_2 = x + x_2_
        x_2 = self.time_linear(x_2)
        x_2 = x + self.time_conv(x_2)

        return x
    
    def frequency_conv(self, x: Tensor) -> Tensor:
        B, F, T, H = x.shape
        x = x.permute(0, 2, 3, 1)  # [B,T,H,F]
        x = x.reshape(B * T, H, F)
        x = self.freq_conv_layer_norm(x)
        x = self.freq_conv(x)
        x = self.freq_conv_PReLU(x)
        x = x.reshape(B, T, H, F)
        x = x.permute(0, 3, 1, 2) # [B,F,T,H]
        return x
    
    def frequency_mhsa(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        B, F, T, H = x.shape
        x = self.freq_mhsa_layer_norm(x)
        x = x.permute(0, 2, 1, 3)  # [B,T,F,H]
        x = x.reshape(B * T, F, H)
        need_weights = False if hasattr(self, "need_weights") else self.need_weights
        x, attn = self.freq_mhsa.forward(x, x, x, need_weights=need_weights, average_attn_weights=False)
        x = x.reshape(B, T, F, H)
        x = self.freq_mhsa_dropout(x)
        x = x.permute(0, 2, 1, 3)
        return x, attn
    
    def frequency_linear(self, x: Tensor) -> Tensor:
        B, F, T, H = x.shape
        x = x.permute(0, 2, 3, 1)  # [B,T,H,F]
        x = x.reshape(B * T, H, F)
        x = self.freq_linear_layer_norm(x)
        x = self.freq_linear(x)
        x = x.reshape(B, T, H, F)
        x = x.permute(0, 3, 1, 2) # [B,F,T,H]
        return x

    def time_mhsa(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        B, F, T, H = x.shape
        x = self.t_mhsa_layer_norm(x)
        x = x.reshape(B * F, T, H)
        need_weights = False if hasattr(self, "need_weights") else self.need_weights
        x, attn = self.t_mhsa.forward(x, x, x, need_weights=need_weights, average_attn_weights=False)
        x = x.reshape(B, F, T, H)
        x = self.t_mhsa_dropout(x)
        return x, attn
    
    def time_linear(self, x: Tensor) -> Tensor:
        B, F, T, H = x.shape
        x = x.reshape(B * F, T, H) # [B*F,T,H]
        x = self.t_linear_layer_norm(x)
        x = self.t_linear(x)
        x = self.t_linear_silu(x)
        x = x.reshape(B, F, T, H)
        return x
    
    def time_conv(self, x: Tensor) -> Tensor:
        B, F, T, H = x.shape
        x = x.transpose(2, 3) # [B,F,H,T]
        x = x.reshape(B * F, H, T)
        x = self.t_conv_down(x)
        x = self.t_conv_silu(x)
        x = self.t_conv_forward(x)
        x = self.t_conv_layer_norm(x)
        x = self.t_conv_silu(x)
        x = self.t_conv_up(x)
        x = self.t_conv_silu(x)
        x = x.reshape(B, F, H, T)
        x = x.transpose(2, 3) # [B,F,T,H]
        return x
    





if __name__ == '__main__':
    x = torch.randn((1, 129, 251, 12))  # [B, F, T, C]
    blockNet = BlockNet(
        input_dim = 12,
        output_dim = 4,
        hidden_dim = 96,
        time_conv_dim = 192,
        encoder_kernel_size = 5,
        num_freqs = 129,
        num_heads = (2, 2),
        dropout = (0, 0, 0)
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