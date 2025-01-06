import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, n_features, d_out, dtype=torch.float32, device="cpu"):
        super().__init__()
        self.n_features = n_features
        self.d_out = d_out
        self.dtype = dtype
        self.device = device
        self._trainable = True

        self.W_dec = nn.Parameter(
            nn.init.kaiming_uniform_(
                torch.empty(n_features, d_out, dtype=dtype, device=device)
            )
        )
        self.b_dec = nn.Parameter(torch.zeros(d_out, dtype=dtype, device=device))
        self.normalize_weights()

    def forward(self, x):
        return x @ self.W_dec + self.b_dec

    def normalize_weights(self):
        self.W_dec.data = F.normalize(self.W_dec.data, p=2, dim=1)

    def remove_parallel_gradient_component(self):
        parallel_component = torch.einsum(
            "nd, nd -> n",
            self.W_dec.grad,
            self.W_dec.data,
        )
        self.W_dec.grad -= torch.einsum(
            "n, nd -> nd",
            parallel_component,
            self.W_dec.data,
        )

    @property
    def trainable(self):
        return self._trainable

    @trainable.setter
    def trainable(self, value):
        self.W_dec.requires_grad = value
        self.b_dec.requires_grad = value
        self._trainable = value
