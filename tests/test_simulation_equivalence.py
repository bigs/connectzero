import jax
import jax.numpy as jnp
import pytest

from simulation import simulate_rollouts, simulate_rollouts_mcts


@pytest.fixture
def params():
    return {"B": 1}


def test_simulate_equivalence(params):
    B = params["B"]

    key = jax.random.PRNGKey(42)

    # Empty board
    board_state = jnp.zeros((6, 7), dtype=jnp.int32)
    batched_board = jnp.expand_dims(board_state, 0)

    turn_count = jnp.array(0, dtype=jnp.int32)
    batched_turn_count = jnp.expand_dims(turn_count, 0)

    # Run simulation
    # Note: We need to use the same key for both to get deterministic results
    # However, simulate_rollouts splits keys inside.
    # Let's see if they use the same random logic.
    # simulate_rollouts: key, subkey = jax.random.split(state.key)
    # simulate_rollouts_mcts: key, subkey = jax.random.split(state.key)
    # Logic seems identical.

    res_arena = simulate_rollouts(key, batched_board, batched_turn_count)
    res_mcts = simulate_rollouts_mcts(key, board_state, turn_count)

    assert res_arena[0] == res_mcts, "Simulation result mismatch"


def test_simulate_equivalence_midgame(params):
    B = params["B"]

    key = jax.random.PRNGKey(101)

    # Midgame board
    # 0 0 0 0 0 0 0
    # 0 0 0 0 0 0 0
    # 0 0 0 0 0 0 0
    # 0 0 1 2 0 0 0
    # 0 0 1 2 0 0 0
    # 0 0 1 2 0 0 0

    board_state = jnp.zeros((6, 7), dtype=jnp.int32)
    board_state = board_state.at[5, 2].set(1)
    board_state = board_state.at[4, 2].set(1)
    board_state = board_state.at[3, 2].set(1)

    board_state = board_state.at[5, 3].set(2)
    board_state = board_state.at[4, 3].set(2)
    board_state = board_state.at[3, 3].set(2)

    turn_count = jnp.array(6, dtype=jnp.int32)

    batched_board = jnp.expand_dims(board_state, 0)
    batched_turn_count = jnp.expand_dims(turn_count, 0)

    res_arena = simulate_rollouts(key, batched_board, batched_turn_count)
    res_mcts = simulate_rollouts_mcts(key, board_state, turn_count)

    assert res_arena[0] == res_mcts, "Simulation result mismatch (midgame)"


def test_simulate_equivalence_winning_state(params):
    # If the state is already winning, simulation should return result immediately
    B = params["B"]
    key = jax.random.PRNGKey(99)

    board_state = jnp.zeros((6, 7), dtype=jnp.int32)
    # Player 1 about to win or has won
    # Actually simulate_rollouts checks winner at start.

    # 1 1 1 1 ...
    board_state = board_state.at[5, 0].set(1)
    board_state = board_state.at[5, 1].set(1)
    board_state = board_state.at[5, 2].set(1)
    board_state = board_state.at[5, 3].set(1)

    turn_count = jnp.array(
        7, dtype=jnp.int32
    )  # P1 just played 4th move (total 7 moves roughly if alternating, but let's say P1 made the winning move)
    # Wait, if P1 made the move, turn count should be odd?
    # P1 moves at 0, 2, 4, 6.
    # If P1 just moved, turn count is 7 (0..6 played).
    # Previous player was (6 % 2) + 1 = 1. Correct.

    batched_board = jnp.expand_dims(board_state, 0)
    batched_turn_count = jnp.expand_dims(turn_count, 0)

    res_arena = simulate_rollouts(key, batched_board, batched_turn_count)
    res_mcts = simulate_rollouts_mcts(key, board_state, turn_count)

    # Should be 1 (Player 1 won)
    # But from perspective of player who made the move.
    # Player 1 made the move. So +1.

    assert res_arena[0] == res_mcts
    assert res_mcts == 1
