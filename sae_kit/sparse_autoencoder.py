import torch
import torch.nn as nn
import torch.nn.functional as F
from sae_lens import SAE as SAELensAutoencoder
import logging
from .encoder import Encoder
from .decoder import Decoder


class SparseAutoencoder(nn.Module):
    def __init__(
        self,
        d_in=None,
        n_features=None,
        k=None,
        d_hidden=None,
        n_hidden=0,
        apply_preencoder_bias=True,
        dtype=torch.float32,
        device="cpu",
        encoder=None,
        decoder=None,
    ):
        super().__init__()

        # Initialize from provided components or parameters
        if encoder is not None and decoder is not None:
            self._init_from_encoder_decoder(encoder, decoder)
        elif encoder is not None:
            self._init_from_encoder(encoder, dtype, device)
        elif decoder is not None:
            self._init_from_decoder(
                decoder, k, d_hidden, n_hidden, apply_preencoder_bias, dtype, device
            )
        elif all(x is not None for x in [d_in, n_features, k]):
            self._init_from_params(
                d_in,
                n_features,
                k,
                d_hidden,
                n_hidden,
                apply_preencoder_bias,
                dtype,
                device,
            )
        else:
            raise ValueError(
                "Must either provide encoder and/or decoder, or all of: d_in, n_features, k"
            )

    def _init_from_params(
        self,
        d_in,
        n_features,
        k,
        d_hidden,
        n_hidden,
        apply_preencoder_bias,
        dtype,
        device,
    ):
        self.d_in = d_in
        self.n_features = n_features
        self.k = k
        self.d_hidden = d_hidden
        self.n_hidden = n_hidden
        self.apply_preencoder_bias = apply_preencoder_bias

        # Create decoder first to use its bias for encoder
        self.decoder = Decoder(n_features, d_in, dtype, device)
        self.encoder = Encoder(
            d_in,
            n_features,
            k,
            d_hidden,
            n_hidden,
            preencoder_bias=self.decoder.b_dec if apply_preencoder_bias else None,
            apply_preencoder_bias=apply_preencoder_bias,
            dtype=dtype,
            device=device,
        )

    def _init_from_encoder_decoder(self, encoder, decoder):
        assert (
            encoder.n_features == decoder.n_features
        ), "Encoder and decoder must have the same number of features"
        assert (
            encoder.d_in == decoder.d_out
        ), "Encoder input dimension must match decoder output dimension"

        self.d_in = encoder.d_in
        self.n_features = encoder.n_features
        self.k = encoder.k
        self.d_hidden = encoder.d_hidden
        self.n_hidden = encoder.n_hidden
        self.apply_preencoder_bias = encoder.apply_preencoder_bias
        self.encoder = encoder
        self.decoder = decoder

    def _init_from_encoder(self, encoder, dtype, device):
        self.d_in = encoder.d_in
        self.n_features = encoder.n_features
        self.k = encoder.k
        self.d_hidden = encoder.d_hidden
        self.n_hidden = encoder.n_hidden
        self.apply_preencoder_bias = encoder.apply_preencoder_bias
        self.encoder = encoder

        # Create decoder normally
        self.decoder = Decoder(self.n_features, self.d_in, dtype=dtype, device=device)

        # If encoder has a preencoder bias, modify both components
        if encoder.preencoder_bias is not None:
            logging.warning(
                "When initializing an SEA from a given encoder and no given decoder, the existing encoder's preencoder_bias will become a buffer in the encoder, and the new decoder will adopt the trainable parameter as decoder.b_dec."
            )
            # Convert encoder's preencoder_bias from Parameter to buffer
            bias_data = encoder.preencoder_bias.data
            del encoder.preencoder_bias
            # Set decoder bias to match encoder's original bias
            self.decoder.b_dec.data.copy_(bias_data)
            # Make encoder use decoder's bias
            encoder.register_buffer("preencoder_bias", self.decoder.b_dec)

    def _init_from_decoder(
        self, decoder, k, d_hidden, n_hidden, apply_preencoder_bias, dtype, device
    ):
        if k is None:
            raise ValueError("Must provide k when initializing from decoder only")

        self.decoder = decoder
        self.d_in = decoder.d_out
        self.n_features = decoder.n_features
        self.k = k
        self.d_hidden = d_hidden
        self.n_hidden = n_hidden
        self.apply_preencoder_bias = apply_preencoder_bias

        # Create new encoder using decoder's bias
        self.encoder = Encoder(
            self.d_in,
            self.n_features,
            self.k,
            d_hidden=d_hidden,
            n_hidden=n_hidden,
            preencoder_bias=self.decoder.b_dec if apply_preencoder_bias else None,
            apply_preencoder_bias=apply_preencoder_bias,
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def from_pretrained(
        release: str,
        sae_id: str,
        k: int,
        d_hidden=None,
        n_hidden=0,
        device: str = "cpu",
    ):
        sae_lens_autoencoder, _, _ = SAELensAutoencoder.from_pretrained(
            release, sae_id, device
        )

        n_features = sae_lens_autoencoder.cfg.d_sae
        d_in = sae_lens_autoencoder.cfg.d_in
        apply_preencoder_bias = sae_lens_autoencoder.cfg.apply_b_dec_to_input

        decoder = Decoder(
            n_features,
            d_in,
            dtype=sae_lens_autoencoder.dtype,
            device=device,
        )
        decoder.W_dec.data = sae_lens_autoencoder.W_dec.data
        decoder.b_dec.data = sae_lens_autoencoder.b_dec.data

        encoder = Encoder(
            d_in,
            n_features,
            k,
            d_hidden,
            n_hidden,
            preencoder_bias=decoder.b_dec if apply_preencoder_bias else None,
            apply_preencoder_bias=apply_preencoder_bias,
            dtype=sae_lens_autoencoder.dtype,
            device=device,
        )
        encoder.W_enc.data = sae_lens_autoencoder.W_enc.data
        encoder.b_enc.data = sae_lens_autoencoder.b_enc.data

        return SparseAutoencoder(encoder=encoder, decoder=decoder)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        return self.decode(self.encode(x))
