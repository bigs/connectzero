import glob
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

from connectzero.game import create_dataloader
from connectzero.model import ConnectZeroModel
from connectzero.model import load as load_model
from connectzero.model import save as save_model

BATCH_SIZE = 8
WEIGHT_DECAY = 10e-4
MOMENTUM = 0.9
LR_START = 2e-2
LR_END = 2e-5
LR_TRANSITION_STEPS = 100_000


def get_optimizer(scheduler_type: str = "linear") -> optax.GradientTransformation:
    if scheduler_type == "linear":
        learning_rate_schedule = optax.linear_schedule(
            LR_START, LR_END, LR_TRANSITION_STEPS
        )
    elif scheduler_type == "cosine":
        learning_rate_schedule = optax.warmup_cosine_decay_schedule(
            init_value=1e-4,
            peak_value=2e-2,
            warmup_steps=10_000,
            decay_steps=LR_TRANSITION_STEPS,
            end_value=1e-5,
        )
    else:
        raise ValueError(f"Invalid scheduler type: {scheduler_type}")

    return optax.chain(
        optax.add_decayed_weights(WEIGHT_DECAY),
        optax.sgd(learning_rate_schedule, momentum=MOMENTUM),
    )


def get_learning_rate(step: int) -> float:
    """Compute the learning rate at a given training step (matches `get_optimizer`)."""
    schedule = optax.linear_schedule(LR_START, LR_END, LR_TRANSITION_STEPS)
    return float(jax.device_get(schedule(step)))


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


def train_epoch(
    model: ConnectZeroModel,
    state: eqx.nn.State,
    opt_state: optax.OptState,
    data_files: list[str],
    batch_size: int,
    optimizer: optax.GradientTransformation,
    initial_step_count: int = 0,
) -> tuple[ConnectZeroModel, eqx.nn.State, optax.OptState, float, int]:
    dataloader = create_dataloader(data_files)

    if opt_state is None:
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    epoch_loss = 0.0
    steps_in_epoch = 0
    step_count = initial_step_count

    pbar = tqdm(
        dataloader.iter(batch_size=batch_size),
        desc="Training",
        unit="batch",
    )

    for batch in pbar:
        model, state, opt_state, loss = make_step(
            model, state, opt_state, batch, optimizer
        )
        epoch_loss += loss
        steps_in_epoch += 1
        step_count += 1

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            avg_loss=f"{(epoch_loss / steps_in_epoch).item():.4f}",
        )

    pbar.close()
    avg_loss = (epoch_loss / steps_in_epoch).item() if steps_in_epoch else 0.0
    print(
        f"Training complete: {steps_in_epoch} steps (total {step_count}), avg loss: {avg_loss:.4f}"
    )

    return model, state, opt_state, avg_loss, step_count


def train_loop(
    checkpoint_path: str,
    data_pattern: str,
    save_dir: str,
    batch_size: int = 8,
):
    """
    Legacy training entrypoint. Loads checkpoint, runs a single training epoch,
    and saves a new checkpoint.
    """
    optimizer = get_optimizer(scheduler_type="cosine")
    model, state, opt_state, steps = load_model(checkpoint_path, optimizer=optimizer)

    data_files = sorted(glob.glob(data_pattern))
    if not data_files:
        raise ValueError(f"No training data found matching pattern: {data_pattern}")

    model, state, opt_state, avg_loss, steps = train_epoch(
        model=model,
        state=state,
        opt_state=opt_state,
        data_files=data_files,
        batch_size=batch_size,
        optimizer=optimizer,
        initial_step_count=steps,
    )

    save_path = os.path.join(save_dir, f"checkpoint_{steps}_steps.eqx")
    hyperparams = {"num_blocks": len(model.blocks)}

    save_model(save_path, hyperparams, steps, model, state, opt_state)
    print(f"Saved checkpoint to {save_path} (steps={steps}, avg_loss={avg_loss:.4f})")
