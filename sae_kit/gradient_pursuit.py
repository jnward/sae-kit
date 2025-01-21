import torch
from torch.optim.adam import Adam
from torch.nn import functional as F
from .decoder import Decoder
from tqdm import tqdm


def gradient_pursuit(activations: torch.Tensor, decoder: Decoder, k: int):
    n = decoder.n_features
    b = activations.shape[0]

    unbiased_targets = activations.to(decoder.device) - decoder.b_dec
    feature_acts = torch.zeros(
        b,
        n,
        dtype=unbiased_targets.dtype,
        device=unbiased_targets.device,
    )

    for _ in range(k):
        feature_acts = _gradient_pursuit_step(
            unbiased_targets, decoder.W_dec, feature_acts
        )
    return feature_acts


def _gradient_pursuit_step(
    targets: torch.Tensor, dictionary: torch.Tensor, feature_acts: torch.Tensor
):
    reconstructions = feature_acts @ dictionary
    residuals = targets - reconstructions

    feature_correlations = residuals @ dictionary.T
    top_feature_idx = torch.argmax(feature_correlations, dim=-1)

    hot_features = feature_acts > 0
    hot_features[torch.arange(hot_features.shape[0]), top_feature_idx] = True

    grad = feature_correlations * hot_features.float()

    grad_signals = grad @ dictionary
    step_size = (grad_signals * residuals).sum(1) / (grad_signals**2).sum(1)

    feature_acts += grad * step_size.unsqueeze(-1)

    feature_acts = F.relu(feature_acts)
    return feature_acts


def gradient_descent(
    activations: torch.Tensor, decoder: Decoder, lr=0.01, lambda_=0.01, n_steps=1000
):
    batch_size, d_model = activations.shape
    feature_acts = torch.zeros(
        batch_size, decoder.n_features, device=activations.device, requires_grad=True
    )

    optimizer = Adam([feature_acts], lr)

    unbiased_targets = activations - decoder.b_dec

    pbar = tqdm(range(n_steps))
    for step in pbar:
        optimizer.zero_grad()

        reconstruction = feature_acts @ decoder.W_dec

        mse_loss = F.mse_loss(reconstruction, unbiased_targets)
        l1_loss = torch.norm(feature_acts, p=1, dim=1).mean()

        loss = mse_loss + lambda_ * l1_loss

        loss.backward()
        optimizer.step()

        # want feature activations to be nonnegative
        with torch.no_grad():
            feature_acts.data = F.relu(feature_acts.data)

        if step % 10 == 0:
            current_l0 = (feature_acts.data > 1e-4).sum(dim=1).float().mean()
            pbar.set_description(
                f"Step {step}, Avg L0: {current_l0:.2f}, MSE: {mse_loss.item():.6f}"
            )

    return feature_acts.detach()


def gradient_descent_topk(
    activations: torch.Tensor, decoder: Decoder, k: int, lr=0.01, n_steps=1000
):
    """
    Finds sparse activations using gradient descent with top-k sparsity constraint.

    Args:
        activations: Target activations to reconstruct (batch_size x d_model)
        decoder: Decoder object containing W_dec and b_dec
        k: Number of non-zero features to maintain per sample
        n_steps: Number of optimization steps
    """
    batch_size, d_model = activations.shape
    feature_acts = torch.zeros(
        batch_size, decoder.n_features, device=activations.device, requires_grad=True
    )

    optimizer = Adam([feature_acts], lr)
    unbiased_targets = activations - decoder.b_dec

    pbar = tqdm(range(n_steps))
    for step in pbar:
        optimizer.zero_grad()

        # Forward pass
        reconstruction = feature_acts @ decoder.W_dec
        loss = F.mse_loss(reconstruction, unbiased_targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Apply top-k sparsity and non-negativity constraints
        with torch.no_grad():
            # First apply ReLU for non-negativity
            feature_acts.data = F.relu(feature_acts.data)

            # Then keep only top-k activations per sample
            values, _ = torch.topk(feature_acts.data, k=k, dim=1)
            threshold = values[:, -1].unsqueeze(1)  # Get smallest kept value per sample
            feature_acts.data = feature_acts.data * (feature_acts.data >= threshold)

        if step % 10 == 0:
            current_l0 = (feature_acts.data > 1e-4).sum(dim=1).float().mean()
            pbar.set_description(
                f"Step {step}, Avg L0: {current_l0:.2f}, MSE: {loss.item():.6f}"
            )

    return feature_acts.detach()
