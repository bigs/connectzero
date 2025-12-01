import jax.numpy as jnp
import pytest

from policies import select_leaf, select_leaf_mcts
from tree import (
    ArenaTree,
    MCTSTree,
    backpropagate,
    backpropagate_mcts_tree_result,
    expand_mcts_tree_node,
    expand_node,
)


@pytest.fixture
def params():
    return {"B": 1, "N": 20, "A": 7}


def assert_selection_equal(res_arena, res_mcts, batch_idx=0):
    """
    Compare SelectResult (Arena) and SelectResultMCTS.
    """
    # leaf_index
    assert res_arena.leaf_index[batch_idx] == res_mcts.leaf_index, "leaf_index mismatch"

    # action_to_expand
    assert res_arena.action_to_expand[batch_idx] == res_mcts.action_to_expand, (
        "action_to_expand mismatch"
    )

    # board_state
    assert jnp.array_equal(res_arena.board_state[batch_idx], res_mcts.board_state), (
        "board_state mismatch"
    )

    # turn_count
    assert res_arena.turn_count[batch_idx] == res_mcts.turn_count, "turn_count mismatch"

    # winner
    assert res_arena.winner[batch_idx] == res_mcts.winner, "winner mismatch"


def test_select_leaf_initial(params):
    B, N, A = params["B"], params["N"], params["A"]

    arena = ArenaTree.init(B, N, A)
    mcts = MCTSTree.init(N, A)

    # Empty board
    board_state = jnp.zeros((6, 7), dtype=jnp.int32)
    batched_board = jnp.expand_dims(board_state, 0)

    # Select on empty tree
    # Should pick first action because all UCBs are inf (visits=0)
    res_arena = select_leaf(arena, batched_board)
    res_mcts = select_leaf_mcts(mcts, board_state)

    assert_selection_equal(res_arena, res_mcts)

    # Expectation: Root is leaf, Action 0 is selected (argmax of infs)
    assert res_mcts.leaf_index == 0
    assert res_mcts.action_to_expand == 0


def test_select_leaf_after_expansion_and_backprop(params):
    B, N, A = params["B"], params["N"], params["A"]

    arena = ArenaTree.init(B, N, A)
    mcts = MCTSTree.init(N, A)

    board_state = jnp.zeros((6, 7), dtype=jnp.int32)
    batched_board = jnp.expand_dims(board_state, 0)

    # 1. Expand Root -> Action 0
    leaf_idx = 0
    action = 0

    arena, arena_node_idx = expand_node(
        arena,
        jnp.array([leaf_idx], dtype=jnp.int32),
        jnp.array([action], dtype=jnp.int32),
    )
    mcts, mcts_node_idx = expand_mcts_tree_node(
        mcts, jnp.array(leaf_idx, dtype=jnp.int32), jnp.array(action, dtype=jnp.int32)
    )

    # 2. Backpropagate a result to update stats for Action 0
    # This makes visits[0, 0] > 0, so UCB for 0 is finite.
    # Others are still inf.
    result = 1
    arena = backpropagate(arena, arena_node_idx, jnp.array([result], dtype=jnp.int32))
    mcts = backpropagate_mcts_tree_result(
        mcts, mcts_node_idx, jnp.array(result, dtype=jnp.int32)
    )

    # 3. Select again
    # Should avoid Action 0 (finite UCB) and pick Action 1 (infinite UCB)
    res_arena = select_leaf(arena, batched_board)
    res_mcts = select_leaf_mcts(mcts, board_state)

    assert_selection_equal(res_arena, res_mcts)
    assert res_mcts.leaf_index == 0
    assert res_mcts.action_to_expand == 1

    # 4. Expand Root -> Action 1
    action_1 = 1
    arena, arena_node_idx_2 = expand_node(
        arena,
        jnp.array([leaf_idx], dtype=jnp.int32),
        jnp.array([action_1], dtype=jnp.int32),
    )
    mcts, mcts_node_idx_2 = expand_mcts_tree_node(
        mcts, jnp.array(leaf_idx, dtype=jnp.int32), jnp.array(action_1, dtype=jnp.int32)
    )

    # Backprop Action 1 with a LOSS (-1) from perspective of parent?
    # Actually result passed to backprop is usually from perspective of the player who just played?
    # Or value of the node.
    # If we pass -1, parent edge value becomes -1.

    arena = backpropagate(arena, arena_node_idx_2, jnp.array([-1], dtype=jnp.int32))
    mcts = backpropagate_mcts_tree_result(
        mcts, mcts_node_idx_2, jnp.array(-1, dtype=jnp.int32)
    )

    # Now Action 0 has val +1, visits 1.
    # Action 1 has val -1, visits 1.
    # Other actions have visits 0 (inf).
    # Selection should pick Action 2 (next inf).
    res_arena = select_leaf(arena, batched_board)
    res_mcts = select_leaf_mcts(mcts, board_state)
    assert_selection_equal(res_arena, res_mcts)
    assert res_mcts.action_to_expand == 2


def test_select_leaf_traversal(params):
    """Test that selection traverses down already expanded nodes."""
    B, N, A = params["B"], params["N"], params["A"]

    arena = ArenaTree.init(B, N, A)
    mcts = MCTSTree.init(N, A)

    board_state = jnp.zeros((6, 7), dtype=jnp.int32)
    batched_board = jnp.expand_dims(board_state, 0)

    # We want to construct a scenario where selection goes Root -> Child -> Leaf
    # To do this, we need to populate visits for all children of Root so none are Inf.

    # 1. Expand ALL children of Root (0..6) and give them visits
    root_idx = 0

    # We need to keep track of node indices
    arena_child_indices = []
    mcts_child_indices = []

    for a in range(A):
        # Expand
        arena, a_idx = expand_node(
            arena,
            jnp.array([root_idx], dtype=jnp.int32),
            jnp.array([a], dtype=jnp.int32),
        )
        mcts, m_idx = expand_mcts_tree_node(
            mcts, jnp.array(root_idx, dtype=jnp.int32), jnp.array(a, dtype=jnp.int32)
        )

        arena_child_indices.append(a_idx)
        mcts_child_indices.append(m_idx)

        # Backprop to make visits > 0
        # Give Action 0 a very high value so it gets selected
        val = 100 if a == 0 else 0

        arena = backpropagate(arena, a_idx, jnp.array([val], dtype=jnp.int32))
        mcts = backpropagate_mcts_tree_result(
            mcts, m_idx, jnp.array(val, dtype=jnp.int32)
        )

    # 2. Verify that selection now picks Action 0 and traverses to Child 0
    # Child 0 is at index mcts_child_indices[0]
    # At Child 0, visits are 0, so it should pick Action 0 (first inf) from Child 0.

    res_arena = select_leaf(arena, batched_board)
    res_mcts = select_leaf_mcts(mcts, board_state)

    assert_selection_equal(res_arena, res_mcts)

    # Check traversal
    # Expected leaf is the child node created for action 0
    expected_leaf_mcts = mcts_child_indices[0]
    assert res_mcts.leaf_index == expected_leaf_mcts
    # Expected action is 0 (since that node is fresh, all children are unvisited)
    assert res_mcts.action_to_expand == 0

    # Check board state update
    # The board should have one move played (Action 0)
    # Player 1 played.
    assert res_mcts.board_state[5, 0] == 1  # Bottom left
    assert res_mcts.turn_count == 1
