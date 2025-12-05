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


@eqx.filter_jit
def make_step(
    model: ConnectZeroModel,
    state: eqx.nn.State,
    opt_state: optax.OptState,
    batch: dict,
    optimizer: optax.GradientTransformation,
):
    inputs = batch["board_state"]
    policy_targets = batch["policy_target"]
    value_targets = batch["value_target"]
    (loss_value, state), grads = eqx.filter_value_and_grad(compute_loss, has_aux=True)(
        model, state, inputs, policy_targets, value_targets
    )
    updates, new_opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, state, new_opt_state, loss_value


def train_loop(
    checkpoint_path: str,
    data_pattern: str,
    save_dir: str,
    batch_size: int = 8,
):
    """
    Main training loop.
    """
    learning_rate_schedule = optax.linear_schedule(2e-1, 2e-4, 100_000)

    optimizer = optax.chain(
        optax.add_decayed_weights(WEIGHT_DECAY),
        optax.sgd(learning_rate_schedule, momentum=MOMENTUM),
    )

    model, state, opt_state = load_model(checkpoint_path, optimizer=optimizer)

    steps = 0
    if checkpoint_path:
        # Try to parse the step count from the filename
        # Expected format: checkpoint_{steps}_steps.eqx
        match = re.search(r"checkpoint_(\d+)_steps\.eqx", checkpoint_path)
        if match:
            steps = int(match.group(1))

    if opt_state is None:
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    dataloader = create_dataloader(data_pattern)

    epoch_loss = 0.0

    for step, batch in enumerate(dataloader.iter(batch_size=batch_size)):
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
