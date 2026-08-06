"""Shared N-adaptive training configuration."""

EPOCHS_CONFIG = {
    500: 30,
    1000: 40,
    2000: 50,
    5000: 60,
    10000: 80,
    20000: 100,
}

DEFAULT_TEMPORAL_ENCODER_ORDER = ["cnn", "lstm", "spatiotemporal"]


def get_hyperparameters(n_size, temporal_encoder):
    """Return N-adaptive hyperparameters for a temporal encoder."""
    if temporal_encoder == "spatiotemporal":
        hidden_dim = 64 if n_size <= 5000 else 96
    elif n_size <= 1000:
        hidden_dim = 64 if temporal_encoder == "lstm" else 48
    elif n_size <= 2000:
        hidden_dim = 80 if temporal_encoder == "lstm" else 64
    elif n_size <= 5000:
        hidden_dim = 96 if temporal_encoder == "lstm" else 80
    else:
        hidden_dim = 128 if temporal_encoder == "lstm" else 96

    if n_size <= 1000:
        batch_size = 64
    elif n_size <= 2000:
        batch_size = 96
    elif n_size <= 5000:
        batch_size = 128
    else:
        batch_size = 160

    if temporal_encoder == "spatiotemporal":
        lr = 0.001 if n_size >= 10000 else 0.002
    elif n_size <= 1000:
        lr = 0.005 if temporal_encoder == "lstm" else 0.007
    elif n_size <= 2000:
        lr = 0.004 if temporal_encoder == "lstm" else 0.006
    elif n_size <= 5000:
        lr = 0.003 if temporal_encoder == "lstm" else 0.005
    else:
        lr = 0.002 if temporal_encoder == "lstm" else 0.004

    if n_size <= 2000:
        patience = 8
    elif n_size <= 5000:
        patience = 10
    else:
        patience = 12

    return {
        "hidden_dim": hidden_dim,
        "batch_size": batch_size,
        "lr": lr,
        "patience": patience,
    }
