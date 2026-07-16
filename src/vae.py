"""
vae.py
======
A small dense variational autoencoder for dimensionality reduction.

Unlike a plain AE, a VAE regularizes the latent space (KL term) toward a smooth
Gaussian, which tends to give codes that generalize better for a downstream
regressor -- helpful with few samples.

Implemented in the canonical Keras 3 form (Sampling layer + Model subclass with
a custom train_step) so it works cleanly on the Keras 3 / TensorFlow stack. We
only need the latent *mean* as features, so `mu_encoder` is returned as a plain
functional model that saves/loads without issues.
"""

from __future__ import annotations


def build_vae(input_dim: int, latent_dim: int = 12, hidden=(128, 64),
              beta: float = 1.0, learning_rate: float = 1e-3):
    """
    Returns (vae, mu_encoder).
      vae        - trainable model; call vae.fit(X, epochs=...) (X is input only)
      mu_encoder - Model(input -> z_mean) for extracting latent features
    """
    import tensorflow as tf
    import keras
    from keras import layers, ops

    class Sampling(layers.Layer):
        """Reparameterization trick: sample z from (z_mean, z_log_var)."""
        def call(self, inputs):
            z_mean, z_log_var = inputs
            batch = ops.shape(z_mean)[0]
            dim = ops.shape(z_mean)[1]
            eps = keras.random.normal(shape=(batch, dim))
            return z_mean + ops.exp(0.5 * z_log_var) * eps

    # -- Encoder --
    enc_in = keras.Input(shape=(input_dim,))
    h = enc_in
    for u in hidden:
        h = layers.Dense(u, activation="elu")(h)
        h = layers.BatchNormalization()(h)
    z_mean = layers.Dense(latent_dim, name="z_mean")(h)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(h)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(enc_in, [z_mean, z_log_var, z], name="encoder")
    mu_encoder = keras.Model(enc_in, z_mean, name="mu_encoder")

    # -- Decoder --
    dec_in = keras.Input(shape=(latent_dim,))
    d = dec_in
    for u in reversed(hidden):
        d = layers.Dense(u, activation="elu")(d)
        d = layers.BatchNormalization()(d)
    dec_out = layers.Dense(input_dim)(d)
    decoder = keras.Model(dec_in, dec_out, name="decoder")

    class VAE(keras.Model):
        def __init__(self, encoder, decoder, beta=1.0, **kw):
            super().__init__(**kw)
            self.encoder = encoder
            self.decoder = decoder
            self.beta = beta
            self.total_loss_tracker = keras.metrics.Mean(name="loss")
            self.recon_tracker = keras.metrics.Mean(name="recon")
            self.kl_tracker = keras.metrics.Mean(name="kl")

        @property
        def metrics(self):
            return [self.total_loss_tracker, self.recon_tracker, self.kl_tracker]

        def _compute(self, data):
            x = data[0] if isinstance(data, (tuple, list)) else data
            z_mean, z_log_var, z = self.encoder(x)
            recon = self.decoder(z)
            recon_loss = ops.mean(ops.sum(ops.square(x - recon), axis=1))
            kl = -0.5 * ops.mean(
                ops.sum(1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var), axis=1))
            return recon_loss, kl

        def train_step(self, data):
            with tf.GradientTape() as tape:
                recon_loss, kl = self._compute(data)
                total = recon_loss + self.beta * kl
            grads = tape.gradient(total, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            self.total_loss_tracker.update_state(total)
            self.recon_tracker.update_state(recon_loss)
            self.kl_tracker.update_state(kl)
            return {m.name: m.result() for m in self.metrics}

        def test_step(self, data):
            recon_loss, kl = self._compute(data)
            self.total_loss_tracker.update_state(recon_loss + self.beta * kl)
            return {"loss": self.total_loss_tracker.result()}

    vae = VAE(encoder, decoder, beta)
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate))
    return vae, mu_encoder