"""
model.py
========
The asymmetric autoencoder with a deep residual regression head, exactly as in
the notebook (Section 2.3):

  Encoder : 2 dense layers (does the heavy lifting)
  Latent  : linear (codes may be +/-)
  Decoder : 1 dense layer (reconstruction is an auxiliary task)
  Reg head: 3 dense layers + a residual skip from the latent space

Two outputs: 'reconstruction' (MSE) and 'regression' (MSE), weighted so the
regression dominates.

TensorFlow is imported lazily so the data/feature modules can be used (and
tested) without TF installed.
"""

from __future__ import annotations


def _tf():
    import tensorflow as tf
    return tf


def build_asymmetric_autoencoder(input_dim: int, hp: dict):
    """Build + compile the model. Returns (model, encoder)."""
    tf = _tf()
    from tensorflow.keras import layers, Model

    inp = layers.Input(shape=(input_dim,), name="input")

    # ── ENCODER ──────────────────────────────────────────────────────────────
    x = layers.Dense(hp["enc1_dim"], activation="elu", name="enc_h1")(inp)
    x = layers.BatchNormalization(name="enc_bn1")(x)
    x = layers.Dense(hp["enc2_dim"], activation="elu", name="enc_h2")(x)
    x = layers.BatchNormalization(name="enc_bn2")(x)
    latent = layers.Dense(hp["latent_dim"], activation="linear", name="latent_space")(x)

    # ── DECODER (lightweight) ────────────────────────────────────────────────
    d = layers.Dense(hp["dec1_dim"], activation="elu", name="dec_h1")(latent)
    d = layers.BatchNormalization(name="dec_bn1")(d)
    recon = layers.Dense(input_dim, activation="sigmoid", name="reconstruction")(d)

    # ── REGRESSION HEAD (deep + residual) ────────────────────────────────────
    r = layers.Dense(hp["reg_h1_units"], activation="elu", name="reg_h1")(latent)
    r = layers.LayerNormalization(name="reg_ln1")(r)
    r = layers.Dropout(hp["reg_dropout1"], name="reg_drop1")(r)

    r = layers.Dense(hp["reg_h2_units"], activation="elu", name="reg_h2")(r)
    r = layers.LayerNormalization(name="reg_ln2")(r)
    r = layers.Dropout(hp["reg_dropout2"], name="reg_drop2")(r)

    r = layers.Dense(hp["reg_h3_units"], activation="elu", name="reg_h3")(r)
    r = layers.LayerNormalization(name="reg_ln3")(r)
    latent_skip = layers.Dense(hp["reg_h3_units"], activation="linear", name="reg_skip")(latent)
    r = layers.Add(name="reg_residual")([r, latent_skip])

    pred = layers.Dense(1, activation="linear", name="regression")(r)

    model = Model(inputs=inp, outputs=[recon, pred], name="asymmetric_autoencoder")
    encoder = Model(inputs=inp, outputs=latent, name="encoder")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp["learning_rate"]),
        loss={"reconstruction": "mse", "regression": "mse"},
        loss_weights={"reconstruction": hp["recon_weight"],
                      "regression": hp["reg_weight"]},
    )
    return model, encoder


def make_tunable_builder(input_dim: int):
    """
    Return a `build(hp)` function for keras-tuner. Search space matches the
    notebook's specialized-model tuner (Section 2.12).
    """
    tf = _tf()
    from tensorflow.keras import layers, Model

    def build(hp):
        cfg = {
            "enc1_dim": hp.Int("enc1_dim", 128, 512, step=64),
            "enc2_dim": hp.Int("enc2_dim", 64, 256, step=64),
            "latent_dim": hp.Int("latent_dim", 8, 48, step=8),
            "dec1_dim": hp.Int("dec1_dim", 64, 256, step=64),
            "reg_h1_units": hp.Int("reg_h1_units", 32, 128, step=32),
            "reg_h2_units": hp.Int("reg_h2_units", 16, 64, step=16),
            "reg_h3_units": hp.Int("reg_h3_units", 8, 32, step=8),
            "reg_dropout1": hp.Float("reg_dropout1", 0.0, 0.4, step=0.1),
            "reg_dropout2": hp.Float("reg_dropout2", 0.0, 0.3, step=0.1),
            "learning_rate": hp.Choice("learning_rate", [1e-2, 1e-3, 5e-4, 1e-4]),
            "recon_weight": hp.Float("recon_weight", 0.1, 1.0, step=0.1),
            "reg_weight": hp.Float("reg_weight", 1.0, 5.0, step=0.5),
        }
        model, _ = build_asymmetric_autoencoder(input_dim, cfg)
        return model

    return build
