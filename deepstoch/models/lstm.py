"""Optional PyTorch LSTM forecaster for Phase 4."""

from __future__ import annotations


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("LSTM support requires the optional 'deep-learning' dependencies; install with pip install .[deep-learning]") from exc
    return torch, nn


def create_lstm(input_size: int, hidden_size: int = 32, num_layers: int = 1):
    """Create a compact sequence-to-one LSTM without importing torch at module load."""
    torch, nn = _torch()

    class LSTMForecaster(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.recurrent = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
            self.output = nn.Linear(hidden_size, 1)

        def forward(self, values):
            sequence_output, _ = self.recurrent(values)
            return self.output(sequence_output[:, -1, :]).squeeze(-1)

    return LSTMForecaster()


def train_lstm(sequence_batch, *, input_size: int, epochs: int = 50, learning_rate: float = 0.001, seed: int = 42):
    """Train an LSTM on an in-memory sequence batch and return model and losses."""
    torch, _ = _torch()
    torch.manual_seed(seed)
    model = create_lstm(input_size)
    features = torch.tensor(sequence_batch.sequences, dtype=torch.float32)
    targets = torch.tensor(sequence_batch.targets, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = torch.nn.MSELoss()
    losses: list[float] = []
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_function(model(features), targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))
    return model, losses
