import os
import re

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from connectzero.game import create_dataloader
from connectzero.model import ConnectZeroModel
from connectzero.model import load as load_model
from connectzero.model import save as save_model

BATCH_SIZE = 8
WEIGHT_DECAY = 10e-4
MOMENTUM = 0.9


def compute_loss(
    model: ConnectZeroModel,
    state: eqx.nn.State,
    inputs: jnp.ndarray,
    policy_targets: jnp.ndarray,
    value_targets: jnp.ndarray,
) -> tuple[jnp.ndarray, eqx.nn.State]:
    batched_model = jax.vmap(
        model, axis_name="batch", in_axes=(0, None), out_axes=(0, None)
    )
    (pred_policy, pred_value), state = batched_model(inputs, state)

    policy_loss = optax.losses.softmax_cross_entropy(pred_policy, policy_targets).mean()
    value_loss = optax.losses.squared_error(pred_value, value_targets).mean()

    return policy_loss + value_loss, state


# We use @eqx.filter_jit to JIT compile this function, handling Equinox modules correctly
@eqx.filter_jit
def make_step(
    model: ConnectZeroModel,
    state: eqx.nn.State,
    opt_state: optax.OptState,
    batch: dict,
    optimizer: optax.Optimizer,
):
    """
    Performs a single optimization step.

    1. Calculate gradients using jax.value_and_grad on loss_fn.
    2. Update optimizer state and model parameters using optax.apply_updates.
    """
    inputs = batch["board_state"]
    policy_targets = batch["policy_target"]
    value_targets = batch["value_target"]
    (loss_value, state), grads = jax.value_and_grad(compute_loss, has_aux=True)(
        model, state, inputs, policy_targets, value_targets
    )
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, state, opt_state, loss_value


def train_loop(
    checkpoint_path: str,
    data_pattern: str,
    save_dir: str,
):
    """
    Main training loop.
    """
    model, state, opt_state = load_model(checkpoint_path)

    steps = 0
    if checkpoint_path:
        # Try to parse the step count from the filename
        # Expected format: checkpoint_{steps}_steps.eqx
        match = re.search(r"checkpoint_(\d+)_steps\.eqx", checkpoint_path)
        if match:
            steps = int(match.group(1))

    learning_rate_schedule = optax.linear_schedule(2e-1, 2e-4, 100_000)

    optimizer = optax.chain(
        optax.add_decayed_weights(WEIGHT_DECAY),
        optax.sgd(learning_rate_schedule, momentum=MOMENTUM),
    )
    if opt_state is None:
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    dataloader = create_dataloader(data_pattern)

    epoch_loss = 0.0

    for step, batch in enumerate(dataloader.iter(batch_size=BATCH_SIZE)):
        model, state, opt_state, loss = make_step(
            model, state, opt_state, batch, optimizer
        )
        epoch_loss += loss.item()
        steps += 1

        if steps % 10 == 0:
            print(f"Step {steps}, Loss: {loss.item():.4f}")

    save_path = os.path.join(save_dir, f"checkpoint_{steps}_steps.eqx")
    hyperparams = {"num_blocks": len(model.blocks)}

    save_model(save_path, hyperparams, model, state, opt_state)


if __name__ == "__main__":
    # Simple entry point
    train_loop(data_dir="./data", save_dir="./checkpoints")
