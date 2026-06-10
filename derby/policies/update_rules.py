"""Reusable update-rule helpers for modern policies."""
from __future__ import annotations

import tensorflow as tf


class GradientNormAdaptiveLearningRate:
    """Adaptive step-size rule based on the global gradient norm.

    Implements alpha = sqrt(epsilon / (||g||_2^2 + eta)).
    """

    def __init__(self, epsilon: float, eta: float = 1e-8):
        epsilon = float(epsilon)
        eta = float(eta)
        if epsilon <= 0.0:
            raise ValueError("epsilon must be > 0")
        if eta <= 0.0:
            raise ValueError("eta must be > 0")
        self.epsilon = epsilon
        self.eta = eta

    def learning_rate(self, gradients):
        gradients = [g for g in gradients if g is not None]
        if gradients:
            grad_norm = tf.linalg.global_norm(gradients)
        else:
            grad_norm = tf.constant(0.0, dtype=tf.float32)
        dtype = grad_norm.dtype
        epsilon = tf.cast(self.epsilon, dtype)
        eta = tf.cast(self.eta, dtype)
        effective_lr = tf.sqrt(epsilon / (tf.square(grad_norm) + eta))
        return effective_lr, grad_norm
