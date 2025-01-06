import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


class Encoder(nn.Module):
    def __init__(
        self,
        d_in,
        n_features,
        k,
        d_hidden=None,
        n_hidden=0,
        preencoder_bias=None,
        apply_preencoder_bias=True,
        # encoder_weights=None,
        # encoder_bias=None,
        dtype=torch.float32,
        device="cpu",
    ):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.n_hidden = n_hidden
        self.n_features = n_features
        self.k = k
        self.apply_preencoder_bias = apply_preencoder_bias
        self.dtype = dtype
        self.device = device
        self._trainable = True

        if preencoder_bias is None and apply_preencoder_bias:
            logging.warning(
                "No preencoder bias was probided, but apply_preencoder bias is True. An independent preencoder bias will be trained; set apply_preencoder_bias=False if this is not desired."
            )

        if preencoder_bias is not None and apply_preencoder_bias:
            self.register_buffer("preencoder_bias", preencoder_bias)
        elif apply_preencoder_bias:
            self.preencoder_bias = nn.Parameter(
                torch.zeros(d_in, dtype=dtype, device=device)
            )
        else:
            self.preencoder_bias = None

        self.W_enc = nn.Parameter(
            nn.init.kaiming_uniform_(
                torch.empty(n_features, d_in, dtype=dtype, device=device)
            )
        )
        self.b_enc = nn.Parameter(torch.zeros(n_features, dtype=dtype, device=device))

        if n_hidden > 0:
            assert d_hidden is not None, "Hidden_dim must be provided if n_hidden > 0"
            mlp_layers = []
            mlp_layers.append(nn.Linear(d_in, d_hidden, bias=True, device=device))
            mlp_layers.append(nn.ReLU())
            for _ in range(n_hidden - 1):
                mlp_layers.append(
                    nn.Linear(d_hidden, d_hidden, bias=True, device=device)
                )
                mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Linear(d_hidden, n_features, bias=True, device=device))
            self.mlp_path = nn.Sequential(*mlp_layers)
        else:
            self.mlp_path = None

        self.device = device

    def top_k(self, x):
        topk_values, _ = torch.topk(x, self.k, dim=-1)
        threshold = topk_values[..., -1].unsqueeze(-1)
        return x * (x >= threshold)

    def forward(self, x):
        if self.preencoder_bias is not None:
            x = x - self.preencoder_bias
        y = x @ self.W_enc + self.b_enc
        if self.mlp_path is not None:
            y += self.mlp_path(x)
        activations = F.relu(y)
        return self.top_k(activations)

    @property
    def trainable(self):
        return self._trainable

    @trainable.setter
    def trainable(self, value):
        self.W_enc.requires_grad = value
        self.b_enc.requires_grad = value
        self._trainable = value
