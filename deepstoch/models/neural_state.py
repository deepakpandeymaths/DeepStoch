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


def train_neural_state_space(sequence_batch, *, input_size: int, validation_batch=None, epochs: int = 50, learning_rate: float = 0.001, patience: int = 10, checkpoint_path=None, seed: int = 42):
    """Train with Gaussian NLL, chronological validation, early stopping, and optional checkpointing."""
    torch, _ = _torch()
    torch.manual_seed(seed)
    model = create_neural_state_space(input_size)
    feature_values = [value for sequence in sequence_batch.sequences for row in sequence for value in row]
    feature_width = len(sequence_batch.sequences[0][0])
    feature_mean = [sum(row[column] for sequence in sequence_batch.sequences for row in sequence) / (len(sequence_batch.sequences) * len(sequence_batch.sequences[0])) for column in range(feature_width)]
    feature_std = [(sum((row[column] - feature_mean[column]) ** 2 for sequence in sequence_batch.sequences for row in sequence) / max(1, len(feature_values) // feature_width - 1)) ** 0.5 for column in range(feature_width)]
    feature_std = [value if value > 1e-8 else 1.0 for value in feature_std]
    target_mean = sum(sequence_batch.targets) / len(sequence_batch.targets)
    target_std = (sum((value - target_mean) ** 2 for value in sequence_batch.targets) / max(1, len(sequence_batch.targets) - 1)) ** 0.5
    target_std = target_std if target_std > 1e-8 else 1.0
    normalize = lambda sequences: [[[((value - feature_mean[column]) / feature_std[column]) for column, value in enumerate(row)] for row in sequence] for sequence in sequences]
    features = torch.tensor(normalize(sequence_batch.sequences), dtype=torch.float32)
    targets = torch.tensor([(value - target_mean) / target_std for value in sequence_batch.targets], dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses: list[float] = []
    validation_losses: list[float] = []
    best_validation = float("inf")
    best_state = None
    stale_epochs = 0
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        mean, scale = model(features)
        distribution = torch.distributions.Normal(mean, scale)
        loss = -distribution.log_prob(targets).mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))
        if validation_batch is not None:
            model.eval()
            with torch.no_grad():
                validation_features = torch.tensor(normalize(validation_batch.sequences), dtype=torch.float32)
                validation_targets = torch.tensor([(value - target_mean) / target_std for value in validation_batch.targets], dtype=torch.float32)
                validation_mean, validation_scale = model(validation_features)
                validation_loss = float(-torch.distributions.Normal(validation_mean, validation_scale).log_prob(validation_targets).mean().item())
            validation_losses.append(validation_loss)
            model.train()
            if validation_loss < best_validation:
                best_validation = validation_loss
                best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
                stale_epochs = 0
                if checkpoint_path is not None:
                    torch.save({"model_state": model.state_dict(), "epoch": len(losses), "validation_loss": validation_loss, "input_size": input_size, "feature_mean": feature_mean, "feature_std": feature_std, "target_mean": target_mean, "target_std": target_std}, checkpoint_path)
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, losses, validation_losses
