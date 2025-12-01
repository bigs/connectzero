import jax.numpy as jnp
import pytest

from tree import (
    ArenaTree,
    MCTSTree,
    advance_mcts_tree,
    advance_tree,
    backpropagate,
    backpropagate_mcts_tree_result,
    expand_mcts_tree_node,
    expand_node,
)


@pytest.fixture
def tree_params():
    return {"B": 1, "N": 10, "A": 7}


def assert_trees_equal(arena: ArenaTree, mcts: MCTSTree, batch_idx: int = 0):
    """Helper to compare an ArenaTree (at a specific batch index) with an MCTSTree."""

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
    # Note: ArenaTree.next_node_index is [B], MCTSTree.next_node_index is scalar (array)
    assert arena.next_node_index[batch_idx] == mcts.next_node_index, (
        "next_node_index mismatch"
    )
    assert arena.root_index[batch_idx] == mcts.root_index, "root_index mismatch"


def test_init_equivalence(tree_params):
    B, N, A = tree_params["B"], tree_params["N"], tree_params["A"]

    arena = ArenaTree.init(B, N, A)
    mcts = MCTSTree.init(N, A)

    assert_trees_equal(arena, mcts)


def test_expand_equivalence(tree_params):
    B, N, A = tree_params["B"], tree_params["N"], tree_params["A"]

    arena = ArenaTree.init(B, N, A)
    mcts = MCTSTree.init(N, A)

    leaf_index = 0  # Root
    action = 3

    # Expand Arena
    # Arena expects batched inputs
    arena, arena_node_idx = expand_node(
        arena,
        jnp.array([leaf_index], dtype=jnp.int32),
        jnp.array([action], dtype=jnp.int32),
    )

    # Expand MCTS
    mcts, mcts_node_idx = expand_mcts_tree_node(
        mcts, jnp.array(leaf_index, dtype=jnp.int32), jnp.array(action, dtype=jnp.int32)
    )

    assert arena_node_idx[0] == mcts_node_idx, "New node index mismatch"
    assert_trees_equal(arena, mcts)

    # Expand another node from the newly created node
    new_leaf_index = mcts_node_idx
    action_2 = 4

    arena, arena_node_idx_2 = expand_node(
        arena,
        jnp.array([new_leaf_index], dtype=jnp.int32),
        jnp.array([action_2], dtype=jnp.int32),
    )

    mcts, mcts_node_idx_2 = expand_mcts_tree_node(
        mcts, new_leaf_index, jnp.array(action_2, dtype=jnp.int32)
    )

    assert arena_node_idx_2[0] == mcts_node_idx_2, "Second node index mismatch"
    assert_trees_equal(arena, mcts)


def test_backprop_equivalence(tree_params):
    B, N, A = tree_params["B"], tree_params["N"], tree_params["A"]

    arena = ArenaTree.init(B, N, A)
    mcts = MCTSTree.init(N, A)

    # Expand a node to backprop from
    leaf_index = 0
    action = 2

    arena, arena_child_idx = expand_node(
        arena,
        jnp.array([leaf_index], dtype=jnp.int32),
        jnp.array([action], dtype=jnp.int32),
    )
    mcts, mcts_child_idx = expand_mcts_tree_node(
        mcts, jnp.array(leaf_index, dtype=jnp.int32), jnp.array(action, dtype=jnp.int32)
    )

    result_val = 1

    # Backprop Arena
    arena = backpropagate(
        arena,
        arena_child_idx,  # Use the child index as start of backprop (it was just expanded)
        jnp.array([result_val], dtype=jnp.int32),
    )

    # Backprop MCTS
    # Note: backpropagate_mcts_tree_result takes initial_leaf_index.
    # If we just expanded, we usually backprop from the new leaf.
    mcts = backpropagate_mcts_tree_result(
        mcts, mcts_child_idx, jnp.array(result_val, dtype=jnp.int32)
    )

    assert_trees_equal(arena, mcts)


def test_advance_equivalence(tree_params):
    B, N, A = tree_params["B"], tree_params["N"], tree_params["A"]

    arena = ArenaTree.init(B, N, A)
    mcts = MCTSTree.init(N, A)

    # Expand a node so we can advance to it
    leaf_index = 0
    action = 5

    arena, _ = expand_node(
        arena,
        jnp.array([leaf_index], dtype=jnp.int32),
        jnp.array([action], dtype=jnp.int32),
    )
    mcts, _ = expand_mcts_tree_node(
        mcts, jnp.array(leaf_index, dtype=jnp.int32), jnp.array(action, dtype=jnp.int32)
    )

    # Advance Arena
    arena = advance_tree(arena, jnp.array([action], dtype=jnp.int32))

    # Advance MCTS
    mcts = advance_mcts_tree(mcts, jnp.array(action, dtype=jnp.int32))

    assert_trees_equal(arena, mcts)


def test_reset_on_invalid_advance_equivalence(tree_params):
    B, N, A = tree_params["B"], tree_params["N"], tree_params["A"]

    arena = ArenaTree.init(B, N, A)
    mcts = MCTSTree.init(N, A)

    # Try to advance to a node that hasn't been expanded (invalid move)
    action = 0

    # Advance Arena
    arena = advance_tree(arena, jnp.array([action], dtype=jnp.int32))

    # Advance MCTS
    mcts = advance_mcts_tree(mcts, jnp.array(action, dtype=jnp.int32))

    # Both should have reset
    assert_trees_equal(arena, mcts)

    # Check if it looks like a fresh tree (root index 0, next_node_index 1)
    assert mcts.root_index == 0
    assert mcts.next_node_index == 1
