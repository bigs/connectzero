import jax
import jax.numpy as jnp
import pytest

from connectzero import batched, single
from connectzero.batched import BatchedSearchTree
from connectzero.single import SearchTree


@pytest.fixture
def params():
    return {"B": 1, "N": 100, "A": 7}


def assert_trees_equal(arena: BatchedSearchTree, mcts: SearchTree, batch_idx: int = 0):
    """Helper to compare an BatchedSearchTree (at a specific batch index) with an SearchTree."""

    # Compare structure
    assert jnp.array_equal(arena.children_index[batch_idx], mcts.children_index), (
        "children_index mismatch"
    )
    assert jnp.array_equal(arena.parents[batch_idx], mcts.parents), "parents mismatch"
    assert jnp.array_equal(
        arena.action_from_parent[batch_idx], mcts.action_from_parent
    ), "action_from_parent mismatch"

    # Compare statistics
    assert jnp.array_equal(arena.children_visits[batch_idx], mcts.children_visits), (
        "children_visits mismatch"
    )
    assert jnp.allclose(arena.children_values[batch_idx], mcts.children_values), (
        "children_values mismatch"
    )

    # Compare allocator state
    assert arena.next_node_index[batch_idx] == mcts.next_node_index, (
        "next_node_index mismatch"
    )
    assert arena.root_index[batch_idx] == mcts.root_index, "root_index mismatch"


def test_mcts_search_equivalence(params):
    B, N, A = params["B"], params["N"], params["A"]
    num_simulations = 10

    key = jax.random.PRNGKey(42)

    arena = BatchedSearchTree.init(B, N, A)
    mcts = SearchTree.init(N, A)

    # Empty board
    board_state = jnp.zeros((6, 7), dtype=jnp.int32)
    batched_board = jnp.expand_dims(board_state, 0)

    # Run MCTS Search
    next_arena, best_action_arena, new_board_arena = batched.run_mcts_search(
        arena, batched_board, num_simulations, key
    )

    next_mcts, best_action_mcts, new_board_mcts = single.run_mcts_search(
        mcts, board_state, num_simulations, key
    )

    # Verify outputs
    assert best_action_arena[0] == best_action_mcts, "Best action mismatch"
    assert jnp.array_equal(new_board_arena[0], new_board_mcts), (
        "New board state mismatch"
    )

    # Verify tree state
    # Note: run_mcts_search returns the ADVANCED tree (root moved to child)
    # So we check if the resulting trees are equivalent.
    assert_trees_equal(next_arena, next_mcts)

    # Let's also verify that they expanded the same way by checking the tree BEFORE advance?
    # We can't easily do that without modifying the function or stepping through.
    # But if the final trees are equal after N simulations and an Advance, it's very likely they were equal before.


def test_mcts_search_equivalence_midgame(params):
    B, N, A = params["B"], params["N"], params["A"]
    num_simulations = 20

    key = jax.random.PRNGKey(123)

    arena = BatchedSearchTree.init(B, N, A)
    mcts = SearchTree.init(N, A)

    # Midgame board
    board_state = jnp.zeros((6, 7), dtype=jnp.int32)
    board_state = board_state.at[5, 3].set(1)
    board_state = board_state.at[4, 3].set(2)
    board_state = board_state.at[5, 2].set(1)

    batched_board = jnp.expand_dims(board_state, 0)

    next_arena, best_action_arena, new_board_arena = batched.run_mcts_search(
        arena, batched_board, num_simulations, key
    )

    next_mcts, best_action_mcts, new_board_mcts = single.run_mcts_search(
        mcts, board_state, num_simulations, key
    )

    assert best_action_arena[0] == best_action_mcts, "Best action mismatch"
    assert_trees_equal(next_arena, next_mcts)
