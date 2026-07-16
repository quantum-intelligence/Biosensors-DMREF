"""
conv_ae.py
==========
1-D convolutional autoencoder for voltammetry traces.

Convolution slides kernels along the voltage axis, so the encoder learns
peak-shaped detectors (height / position / width) rather than treating each
sample point independently. Trained unsupervised on reconstruction, it can use
every trace in the dataset (no concentration label required), which is what lets
a heavier model work despite any single group having few labelled samples.

The encoder output (a small latent vector per trace) is later fed to a
per-group gradient-boosting regressor.
"""

from __future__ import annotations


def _tf():
    import tensorflow as tf
    return tf


def next_multiple(n: int, base: int) -> int:
    """Smallest multiple of `base` that is >= n (so striding divides evenly)."""
    return ((n + base - 1) // base) * base


def pad_to(traces, target_len: int):
    """Right-pad each trace (edge values) to target_len. traces: (n, L)."""
    import numpy as np
    L = traces.shape[1]
    if L == target_len:
        return traces.astype("float32")
    if L > target_len:
        return traces[:, :target_len].astype("float32")
    return np.pad(traces, ((0, 0), (0, target_len - L)), mode="edge").astype("float32")


def global_scale_fit(traces):
    """Global min/max over ALL values (preserves relative peak magnitudes)."""
    import numpy as np
    return float(np.min(traces)), float(np.max(traces))


def global_scale_apply(traces, gmin, gmax):
    """Scale to ~[0,1] using global stats; clip so unseen extremes stay bounded."""
    import numpy as np
    z = (traces - gmin) / (gmax - gmin + 1e-12)
    return np.clip(z, 0.0, 1.0).astype("float32")


def build_conv_autoencoder(input_len: int, latent_dim: int = 24,
                           filters=(16, 32, 64), kernel_sizes=(9, 7, 5),
                           learning_rate: float = 1e-3):
    """
    Build + compile a 1-D conv autoencoder.

    input_len must be divisible by 2**len(filters) (pad traces beforehand).
    Returns (autoencoder, encoder).
    """
    tf = _tf()
    from tensorflow.keras import layers, Model

    n_down = len(filters)
    assert input_len % (2 ** n_down) == 0, \
        f"input_len {input_len} must be divisible by {2**n_down}"

    inp = layers.Input(shape=(input_len, 1), name="trace_in")

    # ── Encoder: strided convs halve the length each step ────────────────────
    x = inp
    for f, k in zip(filters, kernel_sizes):
        x = layers.Conv1D(f, k, strides=2, padding="same", activation="elu")(x)
        x = layers.BatchNormalization()(x)
    conv_len = input_len // (2 ** n_down)
    conv_filters = filters[-1]
    x = layers.Flatten()(x)
    latent = layers.Dense(latent_dim, activation="linear", name="latent")(x)

    # ── Decoder: mirror back up to the original length ───────────────────────
    d = layers.Dense(conv_len * conv_filters, activation="elu")(latent)
    d = layers.Reshape((conv_len, conv_filters))(d)
    for f, k in zip(reversed(filters[:-1]), reversed(kernel_sizes[1:])):
        d = layers.Conv1DTranspose(f, k, strides=2, padding="same", activation="elu")(d)
        d = layers.BatchNormalization()(d)
    d = layers.Conv1DTranspose(filters[0], kernel_sizes[0], strides=2,
                               padding="same", activation="elu")(d)
    out = layers.Conv1D(1, 3, padding="same", activation="sigmoid", name="recon")(d)

    autoencoder = Model(inp, out, name="conv_autoencoder")
    encoder = Model(inp, latent, name="conv_encoder")
    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    return autoencoder, encoder
