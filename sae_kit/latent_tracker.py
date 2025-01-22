import torch


class LatentTracker:
    def __init__(self, n_latents, dead_threshold=10_000_000, device="cuda"):
        self.n_latents = n_latents
        self.dead_threshold = dead_threshold
        self.last_activation = torch.zeros(n_latents, device=device)
        self.current_step = 0
        self.device = device
        self.update_buffer = torch.zeros(n_latents, device=device)

    def update(self, features):
        """Update activation tracking for each latent"""
        with torch.no_grad():
            # Use where instead of boolean indexing
            self.update_buffer.zero_()
            self.update_buffer.masked_fill_(
                (features > 0).any(dim=0), self.current_step
            )
            # Use max to only update when we see a newer activation
            self.last_activation = torch.maximum(
                self.last_activation, self.update_buffer
            )
            self.current_step += features.shape[0]

    def get_dead_latents(self):
        """Return boolean mask of dead latents"""
        with torch.no_grad():
            steps_since_activation = self.current_step - self.last_activation
            return steps_since_activation >= self.dead_threshold
