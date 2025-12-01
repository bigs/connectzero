from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from game import check_winner, check_winner_single, play_move, play_move_single


class SimulateState(NamedTuple):
    key: jnp.ndarray
    board_state: jnp.ndarray
    turn_count: jnp.ndarray
    trajectory_active: jnp.ndarray
    winner: jnp.ndarray


def simulate_rollouts(
    key: jnp.ndarray,
    board_state: jnp.ndarray,
    turn_count: jnp.ndarray,
) -> jnp.ndarray:
    """
    Simulate random games from the given board states to completion.

    Args:
        key: JAX PRNG key for random moves.
        board_state: [B, 6, 7] array of board states.
        turn_count: [B] array of turn counts.

    Returns:
        [B] array of results (+1, -1, 0) from the perspective of the player
        who made the move leading to the leaf node.
    """

    def simulate_body(state: SimulateState) -> SimulateState:
        key, subkey = jax.random.split(state.key)

        legal_moves_mask = state.board_state[:, 0, :] == 0
        # All credit to Gemini for the Gumbel-Max trick
        # Makes it easy to sample from the legal moves
        logits = jnp.where(legal_moves_mask, 0.0, -jnp.inf)
        gumbel_noise = -jnp.log(-jnp.log(jax.random.uniform(subkey, logits.shape)))
        random_action = jnp.argmax(logits + gumbel_noise, axis=1)

        new_board_state = jnp.where(
            state.trajectory_active[:, None, None],
            play_move(
                state.board_state,
                random_action,
                (state.turn_count % 2) + 1,
            ),
            state.board_state,
        )

        new_turn_count = jnp.where(
            state.trajectory_active,
            state.turn_count + 1,
            state.turn_count,
        )

        winner = check_winner(new_board_state, new_turn_count)
        new_trajectory_active = state.trajectory_active & (winner == 0)

        return state._replace(
            key=key,
            board_state=new_board_state,
            turn_count=new_turn_count,
            trajectory_active=new_trajectory_active,
            winner=winner,
        )

    # Initial check
    winner = check_winner(board_state, turn_count)
    trajectory_active = winner == 0

    final_state = jax.lax.while_loop(
        cond_fun=lambda state: jnp.any(state.trajectory_active),
        body_fun=simulate_body,
        init_val=SimulateState(key, board_state, turn_count, trajectory_active, winner),
    )

    # Parent perspective. Who made this move? Subtle change in the logic
    # avoids returning the player who will move next, rather than the one
    # who just made the move.
    parent_player = 2 - (turn_count % 2)

    return jnp.where(
        final_state.winner == parent_player,
        1,
        jnp.where(final_state.winner == 3, 0, -1),
    )


class SimulateStateMCTS(NamedTuple):
    key: jnp.ndarray
    board_state: jnp.ndarray  # [6, 7], dtype=int32
    turn_count: Array  # dtype=int32
    trajectory_active: Array  # dtype=bool
    winner: Array  # dtype=int32


def simulate_rollouts_mcts(
    key: jnp.ndarray,
    board_state: jnp.ndarray,
    turn_count: Array,
) -> Array:
    """
    Simulate random games from the given board states to completion.

    Args:
        key: JAX PRNG key for random moves.
        board_state: [6, 7] array of board states, dtype=int32.
        turn_count: Array, dtype=int32. Current turn count.

    Returns:
        Array, dtype=int32. Results (+1, -1, 0) from the perspective of the player
        who made the move leading to the leaf node.
    """

    def simulate_body(state: SimulateStateMCTS) -> SimulateStateMCTS:
        key, subkey = jax.random.split(state.key)

        legal_moves_mask = state.board_state[0, :] == 0
        # All credit to Gemini for the Gumbel-Max trick
        # Makes it easy to sample from the legal moves
        logits = jnp.where(legal_moves_mask, 0.0, -jnp.inf)
        gumbel_noise = -jnp.log(-jnp.log(jax.random.uniform(subkey, logits.shape)))
        random_action = jnp.argmax(logits + gumbel_noise)

        new_board_state = play_move_single(
            state.board_state,
            random_action,
            (state.turn_count % 2) + 1,
        )
        new_turn_count = state.turn_count + 1
        winner = check_winner_single(new_board_state, new_turn_count)
        new_trajectory_active = state.trajectory_active & (winner == 0)

        return state._replace(
            key=key,
            board_state=new_board_state,
            turn_count=new_turn_count,
            trajectory_active=new_trajectory_active,
            winner=winner,
        )

    # Initial check
    winner = check_winner_single(board_state, turn_count)
    trajectory_active = winner == 0

    final_state = jax.lax.while_loop(
        cond_fun=lambda state: state.trajectory_active,
        body_fun=simulate_body,
        init_val=SimulateStateMCTS(
            key, board_state, turn_count, trajectory_active, winner
        ),
    )

    # Parent perspective. Who made this move? Subtle change in the logic
    # avoids returning the player who will move next, rather than the one
    # who just made the move.
    parent_player = 2 - (turn_count % 2)

    return jnp.where(
        final_state.winner == parent_player,
        1,
        jnp.where(final_state.winner == 3, 0, -1),
    )
