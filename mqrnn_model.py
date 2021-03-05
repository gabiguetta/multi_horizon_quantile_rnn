from typing import List

import pytorch_lightning as pl
import torch
from torch import nn

LSTM_DEPTH = 30
LSTM_NUM_LAYERS = 2
COVARIATE_SIZE = 3  # for yearly. monthly, daily
DYNAMIC_CONTEXT_SIZE = 8
STATIC_CONTEXT_SIZE = 10


class MQRNNEncoder(pl.LightningModule):
    """Encoder network for encoder-decoder forecast model."""

    def __init__(self, input_size: int, num_layers: int, hidden_units: int):
        super().__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_units = hidden_units

        self.encoder = nn.LSTM(input_size=self.input_size,
                               hidden_size=self.hidden_units,
                               num_layers=self.num_layers,
                               batch_first=True)

    def forward(self, x):
        output, (henc, cenc) = self.encoder(x.view(x.shape[0], x.shape[1], self.input_size))
        return output, henc, cenc


class MQRNNDecoder(pl.LightningModule):
    """Encoder network for encoder-decoder forecast model."""

    def __init__(self, hidden_in_len: int, future_len: int, covariate_size: int,
                 dynamic_context_size: int, static_context_size: int, quantile_size: int):
        super().__init__()
        self.hidden_in_len = hidden_in_len
        self.future_len = future_len
        self.covariate_size = covariate_size
        self.dynamic_context_size = dynamic_context_size
        self.static_context_size = static_context_size
        self.quantile_size = quantile_size
        self.global_mlp_in_len = hidden_in_len + future_len * covariate_size

        self.global_mlp = nn.Sequential(
            nn.Linear(self.global_mlp_in_len, self.global_mlp_in_len * 4),
            nn.ReLU(),
            nn.Linear(self.global_mlp_in_len * 4, self.global_mlp_in_len * 2),
            nn.ReLU(),
            nn.Linear(self.global_mlp_in_len * 2, self.static_context_size * 1 + self.dynamic_context_size * self.future_len),
            nn.ReLU()
        )

        self.local_mlp_in_len = dynamic_context_size * 1 + static_context_size * 1 + covariate_size * 1

        self.local_mlp = nn.Sequential(
            nn.Linear(self.local_mlp_in_len, self.local_mlp_in_len * 2),
            nn.ReLU(),
            nn.Linear(self.local_mlp_in_len * 2, self.local_mlp_in_len * 4),
            nn.ReLU(),
            nn.Linear(self.local_mlp_in_len * 4, self.quantile_size),
            nn.Tanh()
        )

    def forward(self, hidden_in, future_covariates):

        reshaped_future_covariates = future_covariates.reshape(future_covariates.shape[0], -1)
        combined_global_input = torch.cat((hidden_in, reshaped_future_covariates), -1)
        global_mlp_out = self.global_mlp(combined_global_input)
        local_mlp_outs = torch.Tensor()
        static_context = global_mlp_out[:, -self.static_context_size:]
        for t in range(self.future_len):
            current_dynamic_context = global_mlp_out[:, t * self.dynamic_context_size: (t+1) * self.dynamic_context_size]
            current_covariate = reshaped_future_covariates[:, t * self.covariate_size: (t+1) * self.covariate_size]
            combined_local_input = torch.cat((current_dynamic_context, static_context, current_covariate), dim=1)
            local_mlp_outs = torch.cat((local_mlp_outs, self.local_mlp(combined_local_input).unsqueeze(-1)), dim=-1)
        return local_mlp_outs


class MQRNNModel(pl.LightningModule):
    """Encoder network for encoder-decoder forecast model."""

    def __init__(self, fct_len: int = 24, encoder_num_layers: int = LSTM_NUM_LAYERS,
                 encoder_hidden_units: int = LSTM_DEPTH, lr: float = 1e-3, quantiles=None):
        super().__init__()
        self.fct_len = fct_len
        self.encoder_num_layers = encoder_num_layers
        self.encoder_hidden_units = encoder_hidden_units
        self.lr = lr
        self.quantiles = quantiles or [0.1, 0.5, 0.9]

        self.encoder = MQRNNEncoder(input_size=1 + COVARIATE_SIZE, num_layers=self.encoder_num_layers,
                                    hidden_units=self.encoder_hidden_units)
        self.decoder = MQRNNDecoder(hidden_in_len=LSTM_DEPTH, future_len=self.fct_len, covariate_size=COVARIATE_SIZE,
                                    dynamic_context_size=DYNAMIC_CONTEXT_SIZE, static_context_size=STATIC_CONTEXT_SIZE,
                                    quantile_size=len(self.quantiles))

    def forward(self, x, future_covariates):
        covariates = torch.cat((x[:, :, 1:], future_covariates), dim=1)
        decoders_outputs = torch.Tensor()
        enc_output, _, _ = self.encoder(x)
        num_decoders = enc_output.shape[1]
        for fct in range(num_decoders):
            fut_cov = covariates[:, fct+1:fct+1+self.fct_len, :]
            hidden_in = enc_output[:, fct, :]
            decoder_out = self.decoder(hidden_in, future_covariates=fut_cov)
            decoders_outputs = torch.cat((decoders_outputs, decoder_out.unsqueeze(-1)), dim=-1)
        return decoders_outputs

    def training_step(self, batch, batch_idx):
        return self.get_batch_loss(batch)

    def validation_step(self, batch, batch_idx):
        loss = self.get_batch_loss(batch)
        self.log('val_mse', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def get_batch_loss(self, batch):
        x, y = batch
        future_covariates = y[:, :, 1:]
        fcts = self(x, future_covariates)
        loss = 0
        total_obs = torch.cat((x[:, :, 0], y[:, :, 0]), dim=-1)
        for i in range(fcts.shape[-1]):
            fct = fcts[:, :, :, i]
            gt = total_obs[:, i+1:i+1+self.fct_len]
            loss += MQRNNModel.qunatile_loss(fct, gt, self.quantiles)
        return loss

    @staticmethod
    def qunatile_loss(logits, y, quantiles: List[float]):
        acc_loss = torch.tensor(0.)
        for i, q in enumerate(quantiles):
            e = logits[:, i, :] - y
            acc_loss += torch.mean(torch.max(q * e, (q - 1) * e))
        return acc_loss
