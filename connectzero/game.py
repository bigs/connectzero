from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array


class TrainingSample(NamedTuple):
    """
    A training sample is a tuple of (board_state, policy_target, value_target, turn_count).

    Usable in single game and batched contexts.
    """

    board_state: jnp.ndarray
    policy_target: jnp.ndarray
    value_target: jax.ndarray
    turn_count: jax.ndarray


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
