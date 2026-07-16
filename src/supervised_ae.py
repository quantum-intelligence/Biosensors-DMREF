"""
supervised_ae.py
================
A supervised (joint) autoencoder: the latent is trained to BOTH reconstruct the
trace AND predict concentration. This forces the bottleneck to retain
concentration-relevant structure, unlike a plain/variational AE that only
optimizes reconstruction (and may compress away the very signal we need).

Hidden-layer sizes and latent dim are configurable so the architecture can be
resized. We extract the latent and hand it to a strong tree regressor downstream
(the AE's own head is only there to shape the latent during training).

TensorFlow imported lazily.
"""

from __future__ import annotations


def build_supervised_ae(input_dim: int, hidden=(256, 128), latent_dim: int = 24,
                        alpha: float = 2.0, learning_rate: float = 1e-3):
    """
    Returns (model, encoder).
      model   : two outputs {reconstruction (linear), regression (linear)}
      encoder : Model(input -> latent) for feature extraction
    Inputs are assumed standardized (linear reconstruction, not sigmoid).
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    inp = layers.Input(shape=(input_dim,), name="in")
    x = inp
    for h in hidden:
        x = layers.Dense(h, activation="elu")(x)
        x = layers.BatchNormalization()(x)
    latent = layers.Dense(latent_dim, activation="linear", name="latent")(x)

    # decoder (mirror)
    d = latent
    for h in reversed(hidden):
        d = layers.Dense(h, activation="elu")(d)
        d = layers.BatchNormalization()(d)
    recon = layers.Dense(input_dim, activation="linear", name="reconstruction")(d)

    # regression head off the latent (shapes the latent to be concentration-aware)
    r = layers.Dense(max(16, latent_dim), activation="elu")(latent)
    r = layers.Dropout(0.1)(r)
    pred = layers.Dense(1, activation="linear", name="regression")(r)

    model = Model(inp, [recon, pred], name="supervised_ae")
    encoder = Model(inp, latent, name="encoder")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss={"reconstruction": "mse", "regression": "mse"},
        loss_weights={"reconstruction": 1.0, "regression": alpha},
    )
    return model, encoder
    