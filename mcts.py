from typing import NamedTuple

import jax
import jax.numpy as jnp

from game import play_move, play_move_single
from policies import select_leaf, select_leaf_mcts
from simulation import simulate_rollouts, simulate_rollouts_mcts
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


class MCTSLoopState(NamedTuple):
    key: jnp.ndarray
    tree: ArenaTree


@jax.jit(static_argnames=["num_simulations"])
def run_mcts_search(
    tree: ArenaTree, board_state: jnp.ndarray, num_simulations: int, key: jnp.ndarray
) -> tuple[ArenaTree, jnp.ndarray, jnp.ndarray]:
    """
    Run MCTS search on the given tree and board state.
    """

    def mcts_step(i: int, state: MCTSLoopState) -> MCTSLoopState:
        key, tree = state.key, state.tree
        # Select
        select_result = select_leaf(tree, board_state)

        current_is_terminal = (select_result.winner != 0) | (
            select_result.turn_count >= 42
        )
        should_expand = (select_result.action_to_expand >= 0) & ~current_is_terminal

        # Expand
        expanded_tree, new_node_idx = expand_node(
            tree, select_result.leaf_index, select_result.action_to_expand
        )
        # Apply the expansion only if should_expand
        tree = jax.tree.map(
            lambda old, new: jnp.where(
                jnp.expand_dims(~should_expand, axis=tuple(range(1, new.ndim))),
                old,
                new,
            ),
            tree,
            expanded_tree,
        )

        # Simulate
        player_who_plays = (select_result.turn_count % 2) + 1
        prospective_board = play_move(
            select_result.board_state,
            select_result.action_to_expand,
            player_who_plays,
        )

        sim_board = jnp.where(
            should_expand[:, None, None], prospective_board, select_result.board_state
        )
        sim_turns = jnp.where(
            should_expand, select_result.turn_count + 1, select_result.turn_count
        )

        key, subkey = jax.random.split(key)
        results = simulate_rollouts(subkey, sim_board, sim_turns)

        # Backpropagate
        target_node_idx = jnp.where(
            should_expand, new_node_idx, select_result.leaf_index
        )
        tree = backpropagate(tree, target_node_idx, results)

        return MCTSLoopState(key=key, tree=tree)

    final_state: MCTSLoopState = jax.lax.fori_loop(
        0, num_simulations, mcts_step, MCTSLoopState(key=key, tree=tree)
    )
    batch_range = jnp.arange(tree.children_index.shape[0])
    root_visits = final_state.tree.children_visits[batch_range, tree.root_index, :]
    best_action = jnp.argmax(root_visits, axis=-1)
    turn_count = jnp.sum(jnp.where(board_state == 0, 0, 1), axis=(1, 2))
    player_who_plays = (turn_count % 2) + 1
    new_board_state = play_move(board_state, best_action, player_who_plays)

    next_tree = advance_tree(final_state.tree, best_action)

    return next_tree, best_action, new_board_state


class MCTSLoopStateMCTS(NamedTuple):
    key: jnp.ndarray
    tree: MCTSTree


@jax.jit(static_argnames=["num_simulations"])
def run_mcts_search_single(
    tree: MCTSTree, board_state: jnp.ndarray, num_simulations: int, key: jnp.ndarray
) -> tuple[MCTSTree, jnp.ndarray, jnp.ndarray]:
    """
    Run MCTS search on the given tree and board state (single game version).

    Args:
        tree: The MCTSTree.
        board_state: [6, 7] array, dtype=int32.
        num_simulations: Number of MCTS iterations.
        key: JAX PRNG key.

    Returns:
        tuple[MCTSTree, jnp.ndarray, jnp.ndarray]:
        - Updated MCTSTree (advanced to the chosen action).
        - The chosen action (int32).
        - The new board state (after chosen action).
    """

    def mcts_step(i: int, state: MCTSLoopStateMCTS) -> MCTSLoopStateMCTS:
        key, tree = state.key, state.tree
        # Select
        select_result = select_leaf_mcts(tree, board_state)

        current_is_terminal = (select_result.winner != 0) | (
            select_result.turn_count >= 42
        )
        should_expand = (select_result.action_to_expand >= 0) & ~current_is_terminal

        # Candidate expansion
        expanded_tree, new_node_idx = expand_mcts_tree_node(
            tree, select_result.leaf_index, select_result.action_to_expand
        )

        # Apply expansion conditionally
        tree = jax.tree.map(
            lambda old, new: jnp.where(should_expand, new, old), tree, expanded_tree
        )

        target_node_idx = jnp.where(
            should_expand, new_node_idx, select_result.leaf_index
        )

        # Simulate
        player_who_plays = (select_result.turn_count % 2) + 1
        prospective_board = play_move_single(
            select_result.board_state,
            select_result.action_to_expand,
            player_who_plays,
        )

        sim_board = jnp.where(
            should_expand, prospective_board, select_result.board_state
        )
        sim_turns = jnp.where(
            should_expand, select_result.turn_count + 1, select_result.turn_count
        )

        key, subkey = jax.random.split(key)
        results = simulate_rollouts_mcts(subkey, sim_board, sim_turns)

        # Backpropagate
        tree = backpropagate_mcts_tree_result(tree, target_node_idx, results)

        return MCTSLoopStateMCTS(key=key, tree=tree)

    final_state: MCTSLoopStateMCTS = jax.lax.fori_loop(
        0, num_simulations, mcts_step, MCTSLoopStateMCTS(key=key, tree=tree)
    )

    root_visits = final_state.tree.children_visits[final_state.tree.root_index, :]
    best_action = jnp.argmax(root_visits)

    turn_count = jnp.sum(jnp.where(board_state == 0, 0, 1))
    player_who_plays = (turn_count % 2) + 1
    new_board_state = play_move_single(board_state, best_action, player_who_plays)

    next_tree = advance_mcts_tree(final_state.tree, best_action)

    return next_tree, best_action, new_board_state
