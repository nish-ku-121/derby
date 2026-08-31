"""Shared policy interface for Derby policies."""
from __future__ import annotations

from abc import ABC
import os

import tensorflow as tf


# Quiet optional CPU driver warnings before TensorFlow is used by policy code.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class AbstractPolicy(ABC):
    def __init__(self, agent=None, is_tensorflow=False, discount_factor=0.99):
        super().__init__()
        # TODO: remove/replace this so that agents and policies do not point to each other.
        self.agent = agent
        self.is_tensorflow = is_tensorflow
        self.discount_factor = discount_factor

    def states_fold_type(self):
        raise NotImplementedError("Subclasses must implement states_fold_type()")

    def actions_fold_type(self):
        raise NotImplementedError("Subclasses must implement actions_fold_type()")

    def rewards_fold_type(self):
        raise NotImplementedError("Subclasses must implement rewards_fold_type()")

    def call(self, states):
        raise NotImplementedError("Subclasses must implement call()")

    def choose_actions(self, call_output):
        raise NotImplementedError("Subclasses must implement choose_actions()")

    def policy_loss(self, states, actions, rewards):
        raise NotImplementedError("Subclasses must implement policy_loss()")

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        pass

    def discount(self, rewards):
        """Compute discounted returns over the time axis."""
        if not tf.is_tensor(rewards):
            rewards = tf.convert_to_tensor(rewards)
        rewards = tf.cast(rewards, rewards.dtype if rewards.dtype.is_floating else tf.float32)
        gamma = tf.cast(self.discount_factor, rewards.dtype)
        rev = tf.reverse(rewards, axis=[1])
        rev_tm = tf.transpose(rev, [1, 0])
        init = tf.zeros_like(rev_tm[0])

        def scan_fn(acc, r):
            return r + gamma * acc

        discounted_rev_tm = tf.scan(scan_fn, rev_tm, initializer=init)
        discounted_rev = tf.transpose(discounted_rev_tm, [1, 0])
        return tf.reverse(discounted_rev, axis=[1])
