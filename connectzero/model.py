import json
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import optax


class ConnectZeroResidualBlock(eqx.Module):
    """
    A residual block for the Connect Four policy/value network.
    """

    conv1: eqx.nn.Conv2d
    batch_norm1: eqx.nn.BatchNorm
    conv2: eqx.nn.Conv2d
    batch_norm2: eqx.nn.BatchNorm

    def __init__(self, key):
        key1, key2 = jax.random.split(key)
        self.conv1 = eqx.nn.Conv2d(
            64, 64, kernel_size=(3, 3), stride=1, padding=1, key=key1, use_bias=False
        )
        self.batch_norm1 = eqx.nn.BatchNorm(input_size=64, axis_name="batch")
        self.conv2 = eqx.nn.Conv2d(
            64, 64, kernel_size=(3, 3), stride=1, padding=1, key=key2, use_bias=False
        )
        self.batch_norm2 = eqx.nn.BatchNorm(input_size=64, axis_name="batch")

    def __call__(
        self, x: jnp.ndarray, state: eqx.nn.State
    ) -> tuple[jnp.ndarray, eqx.nn.State]:
        skip = x
        x = self.conv1(x)
        x, state = self.batch_norm1(x, state)
        x = jax.nn.relu(x)
        x = self.conv2(x)
        x, state = self.batch_norm2(x, state)
        return jax.nn.relu(x + skip), state


class ConnectZeroStem(eqx.Module):
    """
    A stem for the Connect Four policy/value network.
    """

    conv1: eqx.nn.Conv2d
    batch_norm1: eqx.nn.BatchNorm

    def __init__(self, key):
        key, _ = jax.random.split(key)
        self.conv1 = eqx.nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, key=key, use_bias=False
        )
        self.batch_norm1 = eqx.nn.BatchNorm(input_size=64, axis_name="batch")

    def __call__(
        self, x: jnp.ndarray, state: eqx.nn.State
    ) -> tuple[jnp.ndarray, eqx.nn.State]:
        x = self.conv1(x)
        x, state = self.batch_norm1(x, state)
        return jax.nn.relu(x), state


class ConnectZeroPolicyHead(eqx.Module):
    """
    A policy head for the Connect Four policy/value network.
    """

    conv: eqx.nn.Conv2d
    batch_norm: eqx.nn.BatchNorm
    linear: eqx.nn.Linear

    def __init__(self, key):
        convkey, linearkey = jax.random.split(key)
        self.conv = eqx.nn.Conv2d(
            64, 2, kernel_size=1, stride=1, padding=0, key=convkey, use_bias=False
        )
        self.batch_norm = eqx.nn.BatchNorm(input_size=2, axis_name="batch")
        self.linear = eqx.nn.Linear(84, 7, key=linearkey)

    def __call__(
        self, x: jnp.ndarray, state: eqx.nn.State
    ) -> tuple[jnp.ndarray, eqx.nn.State]:
        x = self.conv(x)
        x, state = self.batch_norm(x, state)
        x = jax.nn.relu(x)
        x = jnp.reshape(x, (84,))
        x = self.linear(x)
        return x, state


class ConnectZeroValueHead(eqx.Module):
    """
    A value head for the Connect Four policy/value network.
    """

    conv: eqx.nn.Conv2d
    batch_norm: eqx.nn.BatchNorm
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    def __init__(self, key):
        convkey, linear1key, linear2key = jax.random.split(key, 3)
        self.conv = eqx.nn.Conv2d(
            64, 1, kernel_size=1, stride=1, padding=0, key=convkey, use_bias=False
        )
        self.batch_norm = eqx.nn.BatchNorm(input_size=1, axis_name="batch")
        self.linear1 = eqx.nn.Linear(42, 64, key=linear1key)
        self.linear2 = eqx.nn.Linear(64, 1, key=linear2key)

    def __call__(
        self, x: jnp.ndarray, state: eqx.nn.State
    ) -> tuple[jnp.ndarray, eqx.nn.State]:
        x = self.conv(x)
        x, state = self.batch_norm(x, state)
        x = jax.nn.relu(x)
        x = jnp.reshape(x, (42,))
        x = self.linear1(x)
        x = jax.nn.relu(x)
        x = self.linear2(x)
        x = jax.nn.tanh(x)
        return x, state


class ConnectZeroModel(eqx.Module):
    """
    A Connect Four policy/value network in the style of AlphaZero.

    Shapes:
    - Input: [3, 6, 7] (3 planes, 6 rows, 7 columns)
      One plane for each player (1 or 2)
      One plane for the empty space
    - Policy head: [7]
      One value for each possible action (0-6)
    - Value head: [1]
      Value estimate for the current player (-1 to 1)
    """

    stem: ConnectZeroStem
    blocks: list[ConnectZeroResidualBlock]
    policy_head: ConnectZeroPolicyHead
    value_head: ConnectZeroValueHead

    def __init__(self, key, num_blocks: int = 7, **kwargs):
        key, stemkey = jax.random.split(key)
        self.stem = ConnectZeroStem(stemkey)
        key, blockkeys = jax.random.split(key)
        blockkeys = jax.random.split(blockkeys, num_blocks)
        self.blocks = [ConnectZeroResidualBlock(blockkey) for blockkey in blockkeys]
        key, policyheadkey, valueheadkey = jax.random.split(key, 3)
        self.policy_head = ConnectZeroPolicyHead(policyheadkey)
        self.value_head = ConnectZeroValueHead(valueheadkey)

    def __call__(
        self, x: jnp.ndarray, state: eqx.nn.State
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], eqx.nn.State]:
        x, state = self.stem(x, state)

        for block in self.blocks:
            x, state = block(x, state)

        policy, state = self.policy_head(x, state)
        value, state = self.value_head(x, state)
        return (policy, value), state


def save(
    filename: str,
    hyperparams: dict,
    model: ConnectZeroModel,
    state: eqx.nn.State,
    opt_state: Optional[optax.OptState] = None,
):
    with open(filename, "wb") as f:
        hyperparams["has_opt_state"] = opt_state is not None
        hyperparam_str = json.dumps(hyperparams)
        f.write((hyperparam_str + "\n").encode())
        # Serialize model/state first, then opt_state separately.
        # This allows loading just model/state without needing to deserialize opt_state.
        eqx.tree_serialise_leaves(f, (model, state))
        if opt_state is not None:
            eqx.tree_serialise_leaves(f, opt_state)


def load(
    filename: str,
    opt_state: Optional[optax.OptState] = None,
    optimizer: Optional[optax.GradientTransformation] = None,
) -> tuple[ConnectZeroModel, eqx.nn.State, Optional[optax.OptState]]:
    with open(filename, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        has_opt_state = hyperparams.get("has_opt_state", False)

        model, state = eqx.nn.make_with_state(ConnectZeroModel)(
            key=jax.random.PRNGKey(0), **hyperparams
        )

        # Always deserialize model/state first (they're serialized separately from opt_state)
        (model, state) = eqx.tree_deserialise_leaves(f, (model, state))

        # Only deserialize opt_state if it exists in file AND we can reconstruct its structure
        if has_opt_state:
            if opt_state is None and optimizer is not None:
                opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

            if opt_state is not None:
                opt_state = eqx.tree_deserialise_leaves(f, opt_state)
                return model, state, opt_state
            # If has_opt_state but no optimizer provided, skip opt_state (inference mode)

        return model, state, None
