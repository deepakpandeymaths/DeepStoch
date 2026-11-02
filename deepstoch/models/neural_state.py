"""Optional neural state-space model with a probabilistic output head."""

from __future__ import annotations


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("Neural state-space support requires the optional 'deep-learning' dependencies; install with pip install .[deep-learning]") from exc
    return torch, nn


def create_neural_state_space(input_size: int, state_size: int = 32):
    """Create a sequence encoder, latent transition, and mean/scale decoder."""
    _, nn = _torch()

    class NeuralStateSpace(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.GRU(input_size, state_size, batch_first=True)
            self.transition = nn.Sequential(nn.Linear(state_size, state_size), nn.Tanh(), nn.Linear(state_size, state_size))
            self.mean_head = nn.Linear(state_size, 1)
            self.log_scale_head = nn.Linear(state_size, 1)

        def forward(self, values):
            _, hidden = self.encoder(values)
            latent = hidden[-1]
            next_state = self.transition(latent)
            mean = self.mean_head(next_state).squeeze(-1)
            scale = self.log_scale_head(next_state).clamp(min=-6.0, max=3.0).exp().squeeze(-1)
            return mean, scale

    return NeuralStateSpace()


def train_neural_state_space(sequence_batch, *, input_size: int, epochs: int = 50, learning_rate: float = 0.001, seed: int = 42):
    """Train with a Gaussian negative log-likelihood objective."""
    torch, _ = _torch()
    torch.manual_seed(seed)
    model = create_neural_state_space(input_size)
    features = torch.tensor(sequence_batch.sequences, dtype=torch.float32)
    targets = torch.tensor(sequence_batch.targets, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses: list[float] = []
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        mean, scale = model(features)
        distribution = torch.distributions.Normal(mean, scale)
        loss = -distribution.log_prob(targets).mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))
    return model, losses
