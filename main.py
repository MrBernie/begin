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
                 input_dim: int = 12,
                 output_dim: int = 4,
                 hidden_dim: int = 96,
                 time_conv_dim: int = 192,
                 encoder_kernel_size: int = 5,
                 num_layers: int = 6,
                 num_freqs: int = 129,
                 num_heads: tuple = (2, 2),
                 dropout: tuple = (0, 0, 0),
                 learning_rate: float = 1e-3
                 ):
        super().__init__()
        self.save_hyperparameters()
        torch.set_float32_matmul_precision('medium')
        self.model = BernNet(
            input_dim = self.hparams.input_dim,
            output_dim = self.hparams.output_dim,
            hidden_dim = self.hparams.hidden_dim,
            encoder_kernel_size=self.hparams.encoder_kernel_size,
            num_layers=self.hparams.num_layers,
            num_freqs=self.hparams.num_freqs,
            num_heads=self.hparams.num_heads,
            dropout=self.hparams.dropout
        )
        self.learning_rate = learning_rate 
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
        # Todo: stft
        self.model(x)

        return None
    
    def training_step(self, batch, batch_idx):
        mic_sig_batch = batch[0]  # [B, C, T]
        gt_batch = batch[1] # [B,Spk,C,T]
        pred_batch = self.forward(mic_sig_batch)

        loss, evidence, U = self.ce_loss_uncertainty(
            pred_batch=pred_batch, gt_batch=gt_batch, current_epoch=self.current_epoch)

        self.log("train/loss", loss, prog_bar=True,
                 on_epoch=True, sync_dist=True)

        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        mic_sig_batch = batch[0]  # [B, C, T]
        gt_batch = batch[1]
        pred_batch = self.forward(mic_sig_batch)

        loss, evidence, U = self.ce_loss_uncertainty(
            pred_batch=pred_batch, gt_batch=gt_batch, current_epoch=self.current_epoch)

        self.log("train/loss", loss, prog_bar=True,
                 on_epoch=True, sync_dist=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.hparams.lr)

        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=0.8988, last_epoch=-1)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100, eta_min=0)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                # 'monitor': 'valid/loss',
            }
        }

if __name__ == '__main__':
    print(model)