from typing import NamedTuple

import jax
import jax.numpy as jnp
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, load_dataset
from jax import Array
from tqdm import tqdm


class TrainingSample(NamedTuple):
    """
    A training sample is a tuple of (board_state, policy_target, value_target).

    Usable in single game and batched contexts.
    """

    board_state: jnp.ndarray
    policy_target: jnp.ndarray
    value_target: Array


def play_move(
    board_state: jnp.ndarray, action: jnp.ndarray, player_id: int | jnp.ndarray
) -> jnp.ndarray:
    """
    Play the action (column) on the board for each item in the batch.
    Assumes 0=Top, 5=Bottom. Fills from 5 downwards.
    player_id can be a scalar or a batch array.
    """
    batch_range = jnp.arange(board_state.shape[0])
    selected_columns = board_state[batch_range, :, action]
    target_rows = jnp.sum(selected_columns == 0, axis=1) - 1

    return board_state.at[batch_range, target_rows, action].set(player_id)


def play_move_single(
    board_state: jnp.ndarray, action: Array, player_id: Array
) -> jnp.ndarray:
    """
    Play a single move on the board.

    Args:
        board_state: [6, 7] array of board state, dtype=int32.
        action: Array, dtype=int32. Column to play (0-6).
        player_id: Array, dtype=int32. Player ID (1 or 2).

    Returns:
        [6, 7] array of updated board state, dtype=int32.
    """
    updated_board_state = play_move(
        jnp.expand_dims(board_state, axis=0), jnp.expand_dims(action, axis=0), player_id
    )
    return jnp.squeeze(updated_board_state, axis=0)


def check_winner(board_state: jnp.ndarray, turn_count: jnp.ndarray) -> jnp.ndarray:
    """
    Check for a winner in the batch of boards.

    Args:
        board_state: [B, 6, 7] array (0=Empty, 1=P1, 2=P2)
        turn_count: [B] array of turn counts

    Returns:
        [B] array where:
        0 = Incomplete game
        1 = Player 1 won
        2 = Player 2 won
        3 = Draw
    """
    filters = jnp.array(
        [
            [[[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]],  # Horizontal
            [[[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]],  # Vertical
            [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]],  # Diag
            [[[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]],  # Anti-Diag
        ],
        dtype=jnp.int32,
    )

    player_one = jnp.where(board_state == 1, 1, 0)
    player_two = jnp.where(board_state == 2, 1, 0)

    input_tensor_one = jnp.expand_dims(player_one, axis=1)
    input_tensor_two = jnp.expand_dims(player_two, axis=1)

    one_output = jax.lax.conv_general_dilated(
        lhs=input_tensor_one,
        rhs=filters,
        window_strides=(1, 1),
        padding=[(0, 3), (0, 3)],
    )
    two_output = jax.lax.conv_general_dilated(
        lhs=input_tensor_two,
        rhs=filters,
        window_strides=(1, 1),
        padding=[(0, 3), (0, 3)],
    )
    one_win = jnp.any(one_output == 4, axis=(1, 2, 3))
    two_win = jnp.any(two_output == 4, axis=(1, 2, 3))
    winner = jnp.where(one_win, 1, jnp.where(two_win, 2, 0))
    return jnp.where((winner == 0) & (turn_count >= 42), 3, winner)


def check_winner_single(
    board_state: jnp.ndarray, turn_count: jnp.ndarray
) -> jnp.ndarray:
    winner = check_winner(
        jnp.expand_dims(board_state, axis=0), jnp.expand_dims(turn_count, axis=0)
    )
    return jnp.squeeze(winner, axis=0)


def trajectories_active(
    board_state: jnp.ndarray, turn_count: jnp.ndarray
) -> jnp.ndarray:
    """
    Check if any trajectory is still active.
    """
    winners = check_winner(board_state, turn_count)
    return winners == 0


def trajectory_is_active(board_state: jnp.ndarray, turn_count: Array) -> Array:
    """
    Check if a trajectory is still active (game not finished).

    Args:
        board_state: [6, 7] array of board state, dtype=int32.
        turn_count: Array, dtype=int32. Current turn count.

    Returns:
        Array, dtype=bool. True if the game is still active, False otherwise.
    """
    return check_winner_single(board_state, turn_count) == 0


def print_board_states(board_states: jnp.ndarray) -> None:
    """
    Pretty print a batch of Connect Four board states side-by-side.

    Args:
        board_states: [B, 6, 7] array of integers
                      0 = Empty
                      1 = Player 1
                      2 = Player 2
    """
    # Symbols mapping: 0 -> '.', 1 -> 'X', 2 -> 'O'
    symbols = {0: ".", 1: "X", 2: "O"}

    # Move from GPU/TPU to CPU and convert to standard numpy for easier iteration
    board_states_np = jax.device_get(board_states)
    batch_size = board_states_np.shape[0]
    rows = board_states_np.shape[1]

    # Print headers
    header_row = ""
    separator_row = ""
    for b in range(batch_size):
        header_row += f"Board {b:<16}"  # Align left with padding
        separator_row += "  0 1 2 3 4 5 6    "
    print(header_row)
    print(separator_row)

    # Print board rows
    for row in range(rows):
        line_str = ""
        for b in range(batch_size):
            row_content = " ".join(
                symbols[int(cell)] for cell in board_states_np[b, row]
            )
            line_str += f"| {row_content} |  "
        print(line_str)

    # Print bottom separator
    bottom_str = ""
    for b in range(batch_size):
        bottom_str += "  " + "- " * 7 + "   "
    print(bottom_str)
    print()


def print_board_state(board_state: jnp.ndarray) -> None:
    print_board_states(jnp.expand_dims(board_state, axis=0))


def to_model_input(board_state: jnp.ndarray, turn_count: jnp.ndarray) -> jnp.ndarray:
    """
    Convert a game state into the appropriate input for our policy/value network.

    The current player's tokens will always be in channel one, the other player's
    tokens will be in channel two, and the empty spaces will be in channel three.

    Args:
        board_state: [6, 7] array of board state, dtype=int32.
        turn_count: Array, dtype=int32. Current turn count.

    Returns:
        [3, 6, 7] array of model input, dtype=float32.
    """
    current_player = (turn_count % 2) + 1
    channel_one = jnp.where(board_state == current_player, 1, 0)
    channel_two = jnp.where(board_state == (current_player % 2) + 1, 1, 0)
    channel_three = jnp.where(board_state == 0, 1, 0)
    return jnp.stack([channel_one, channel_two, channel_three], axis=0).astype(
        jnp.float32
    )


to_model_input_batched = jax.vmap(to_model_input, in_axes=(0, 0))


def save_trajectories(samples: list[TrainingSample], filename: str):
    """
    Save a list of TrainingSample named tuples to a Parquet file.
    """
    # Convert list of NamedTuples to a dictionary of lists
    # We use jax.device_get to ensure we have numpy arrays on CPU

    # Verify lengths
    n = len(samples)
    if n == 0:
        return

    flat_data = {
        "board_state": [],
        "policy_target": [],
        "value_target": [],
    }

    for s in samples:
        b_state = jax.device_get(s.board_state)
        p_target = jax.device_get(s.policy_target)
        v_target = jax.device_get(s.value_target)

        # Check if batched (ndim=4 for board_state [B, 3, 6, 7])
        if b_state.ndim == 4:
            # Iterate over batch dimension
            B = b_state.shape[0]
            for i in range(B):
                flat_data["board_state"].append(b_state[i].flatten().tolist())
                flat_data["policy_target"].append(p_target[i].tolist())
                flat_data["value_target"].append(v_target[i].item())
        else:
            # Single game
            flat_data["board_state"].append(b_state.flatten().tolist())
            flat_data["policy_target"].append(p_target.tolist())
            flat_data["value_target"].append(v_target.item())

    # Create Arrow Table
    table = pa.Table.from_pydict(flat_data)

    # Write to Parquet
    pq.write_table(table, filename)
    tqdm.write(f"Saved {len(flat_data['board_state'])} samples to {filename}")


def load_trajectories(filename: str) -> list[TrainingSample]:
    """
    Load a list of TrainingSample named tuples from a Parquet file.
    """
    table = pq.read_table(filename)
    data = table.to_pydict()

    samples = []
    num_samples = len(data["board_state"])

    for i in range(num_samples):
        # Reconstruct board_state
        board_flat = data["board_state"][i]
        arr = jnp.array(board_flat, dtype=jnp.float32)

        if arr.size == 42:
            # This would be raw board state (int32), but we loaded as float
            # This case shouldn't happen if we always save model inputs [3, 6, 7]
            # But just in case, we handle it.
            board_state = arr.astype(jnp.int32).reshape(6, 7)
        elif arr.size == 126:
            board_state = arr.reshape(3, 6, 7)
        else:
            raise ValueError(f"Unexpected board state size: {arr.size}")

        # Reconstruct policy_target
        policy_target = jnp.array(data["policy_target"][i], dtype=jnp.float32)

        # Reconstruct value_target
        value_target = jnp.array([data["value_target"][i]], dtype=jnp.float32)

        samples.append(
            TrainingSample(
                board_state=board_state,
                policy_target=policy_target,
                value_target=value_target,
            )
        )

    return samples


def reshape_transforms(examples):
    """
    Preprocessing function to reshape flattened Parquet data back into
    spatial tensors for the model.
    """
    batch_size = len(examples["board_state"])

    boards = jnp.array(examples["board_state"], dtype=jnp.float32)
    boards = boards.reshape(batch_size, 3, 6, 7)

    policies = jnp.array(examples["policy_target"], dtype=jnp.float32)

    values = jnp.array(examples["value_target"], dtype=jnp.float32)
    values = values.reshape(batch_size, 1)

    return {"board_state": boards, "policy_target": policies, "value_target": values}


def get_parquet_row_count(filename: str) -> int:
    """
    Fast metadata-only row count for a Parquet file.

    """
    return int(pq.read_metadata(filename).num_rows)


def create_dataloader(data_files: list[str]) -> Dataset:
    ds = load_dataset("parquet", data_files=data_files, split="train")
    ds = ds.shuffle()
    ds = ds.with_transform(reshape_transforms)

    return ds
