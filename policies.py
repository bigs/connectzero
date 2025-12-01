from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from game import (
    check_winner,
    check_winner_single,
    play_move,
    play_move_single,
    trajectories_active,
    trajectory_is_active,
)
from tree import ArenaTree, MCTSTree


class SelectState(NamedTuple):
    """
    The state for the leaf selection jax loop.
    """

    current_node_index: jnp.ndarray  # [B], dtype=int32
    next_action: jnp.ndarray  # [B], dtype=int32
    trajectory_active: jnp.ndarray  # [B], dtype=bool
    board_state: jnp.ndarray  # [B, 6, 7], dtype=int32
    turn_count: jnp.ndarray  # [B], dtype=int32
    winner: jnp.ndarray  # [B], dtype=int32

    @classmethod
    def init(cls, B: int, board_state: jnp.ndarray, root_index: jnp.ndarray):
        turn_count = jnp.count_nonzero(board_state, axis=(1, 2))
        return cls(
            current_node_index=root_index,
            next_action=jnp.full((B,), -1, dtype=jnp.int32),
            trajectory_active=trajectories_active(board_state, turn_count),
            board_state=board_state,
            turn_count=turn_count,
            winner=check_winner(board_state, turn_count),
        )


class SelectResult(NamedTuple):
    leaf_index: jnp.ndarray  # [B], dtype=int32
    action_to_expand: jnp.ndarray  # [B], dtype=int32
    board_state: jnp.ndarray  # [B, 6, 7], dtype=int32
    turn_count: jnp.ndarray  # [B], dtype=int32
    winner: jnp.ndarray  # [B], dtype=int32


def select_leaf(tree: ArenaTree, board_state: jnp.ndarray) -> SelectResult:
    """
    Select a leaf node to expand.
    """

    c = jnp.sqrt(2)

    def compute_ucb_values(
        tree: ArenaTree,
        current_node_index: jnp.ndarray,
        board_state: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute the UCB values for each action.
        """
        visits = tree.children_visits[batch_range, current_node_index, :]
        total_values = tree.children_values[batch_range, current_node_index, :]
        safe_visits = jnp.maximum(visits, 1)
        q_values = jnp.where(visits > 0, total_values / safe_visits, 0.0)
        parent_visits = jnp.sum(visits, axis=1)
        safe_parent_visits = jnp.maximum(parent_visits, 1)
        exploration = c * jnp.sqrt(jnp.log(safe_parent_visits[:, None]) / safe_visits)
        ucb = jnp.where(visits == 0, jnp.inf, q_values + exploration)
        # Mask full columns
        ucb = jnp.where(jnp.any(board_state == 0, axis=1), ucb, -jnp.inf)

        return ucb

    # Don't compute this in every iteration, dummy
    batch_range = jnp.arange(tree.children_index.shape[0])

    def select_leaf_body(state: SelectState) -> SelectState:
        ucb_values = compute_ucb_values(
            tree, state.current_node_index, state.board_state
        )
        best_action = jnp.argmax(ucb_values, axis=1)

        child_indices = tree.children_index[
            batch_range, state.current_node_index, best_action
        ]
        child_exists = child_indices != -1

        # Update the next action for all active trajectories
        new_next_action = jnp.where(
            state.trajectory_active, best_action, state.next_action
        )

        # Advance the current node index only if the child exists and the trajectory is active
        new_current_node_index = jnp.where(
            state.trajectory_active & child_exists,
            child_indices,
            state.current_node_index,
        )

        prospective_board_state = play_move(
            state.board_state, best_action, (state.turn_count % 2) + 1
        )
        prospective_winner = check_winner(prospective_board_state, state.turn_count + 1)
        is_unfinished = prospective_winner == 0

        # Remain active only if we have not discovered a leaf node
        new_trajectory_active = state.trajectory_active & child_exists

        # Iterate
        new_board_state = jnp.where(
            # Only update when we keep traversing deeper
            new_trajectory_active[:, None, None],
            prospective_board_state,
            state.board_state,
        )
        new_turn_count = jnp.where(
            new_trajectory_active,
            state.turn_count + 1,
            state.turn_count,
        )

        # Reuse prospective_winner if we traversed, otherwise keep previous winner
        new_winner = jnp.where(new_trajectory_active, prospective_winner, state.winner)

        new_trajectory_active = new_trajectory_active & is_unfinished

        return state._replace(
            current_node_index=new_current_node_index,
            next_action=new_next_action,
            trajectory_active=new_trajectory_active,
            board_state=new_board_state,
            turn_count=new_turn_count,
            winner=new_winner,
        )

    final_state = jax.lax.while_loop(
        cond_fun=lambda state: jnp.any(state.trajectory_active),
        body_fun=select_leaf_body,
        init_val=SelectState.init(
            B=tree.children_index.shape[0],
            board_state=board_state,
            root_index=tree.root_index,
        ),
    )

    return SelectResult(
        leaf_index=final_state.current_node_index,
        action_to_expand=final_state.next_action,
        board_state=final_state.board_state,
        turn_count=final_state.turn_count,
        winner=final_state.winner,
    )


class SelectStateMCTS(NamedTuple):
    """
    The state for the leaf selection jax loop.
    """

    current_node_index: Array  # dtype=int32
    next_action: Array  # dtype=int32
    trajectory_active: Array  # dtype=bool
    board_state: jnp.ndarray  # [6, 7], dtype=int32
    turn_count: Array  # dtype=int32
    winner: Array  # dtype=int32

    @classmethod
    def init(cls, board_state: jnp.ndarray, root_index: jnp.ndarray):
        turn_count = jnp.count_nonzero(board_state)
        return cls(
            current_node_index=root_index,
            next_action=-1,
            trajectory_active=trajectory_is_active(board_state, turn_count),
            board_state=board_state,
            turn_count=turn_count,
            winner=check_winner_single(board_state, turn_count),
        )


class SelectResultMCTS(NamedTuple):
    leaf_index: Array  # dtype=int32
    action_to_expand: Array  # dtype=int32
    board_state: jnp.ndarray  # [6, 7], dtype=int32
    turn_count: Array  # dtype=int32
    winner: Array  # dtype=int32


def select_leaf_mcts(tree: MCTSTree, board_state: jnp.ndarray) -> SelectResultMCTS:
    """
    Select a leaf node to expand.
    """

    c = jnp.sqrt(2)

    def compute_ucb_values(
        tree: MCTSTree,
        current_node_index: Array,
        board_state: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute the UCB values for each action.

        Args:
            tree: The MCTS tree.
            current_node_index: Array, dtype=int32. Index of the current node.
            board_state: [6, 7] array of board state, dtype=int32.

        Returns:
            Array of UCB values for each action.
        """
        visits = tree.children_visits[current_node_index, :]
        total_values = tree.children_values[current_node_index, :]
        safe_visits = jnp.maximum(visits, 1)
        q_values = jnp.where(visits > 0, total_values / safe_visits, 0.0)
        parent_visits = jnp.sum(visits)  # scalar now
        safe_parent_visits = jnp.maximum(parent_visits, 1)
        exploration = c * jnp.sqrt(jnp.log(safe_parent_visits) / safe_visits)
        ucb = jnp.where(visits == 0, jnp.inf, q_values + exploration)
        # Mask full columns
        ucb = jnp.where(board_state[0, :] == 0, ucb, -jnp.inf)

        return ucb

    def select_leaf_body(state: SelectStateMCTS) -> SelectStateMCTS:
        ucb_values = compute_ucb_values(
            tree, state.current_node_index, state.board_state
        )
        best_action = jnp.argmax(ucb_values)

        child_index = tree.children_index[state.current_node_index, best_action]
        child_exists = child_index != -1

        # Advance the current node index only if the child exists
        new_current_node_index = jnp.where(
            child_exists,
            child_index,
            state.current_node_index,
        )

        new_board_state = play_move_single(
            state.board_state, best_action, (state.turn_count % 2) + 1
        )
        new_winner = check_winner_single(new_board_state, state.turn_count + 1)
        is_unfinished = new_winner == 0

        # Remain active only if we have not discovered a leaf node
        new_trajectory_active = child_exists & is_unfinished

        # Iterate
        new_turn_count = state.turn_count + 1

        return state._replace(
            current_node_index=new_current_node_index,
            next_action=best_action,
            trajectory_active=new_trajectory_active,
            board_state=new_board_state,
            turn_count=new_turn_count,
            winner=new_winner,
        )

    final_state = jax.lax.while_loop(
        cond_fun=lambda state: state.trajectory_active,
        body_fun=select_leaf_body,
        init_val=SelectStateMCTS.init(
            board_state=board_state,
            root_index=tree.root_index,
        ),
    )

    return SelectResultMCTS(
        leaf_index=final_state.current_node_index,
        action_to_expand=final_state.next_action,
        board_state=final_state.board_state,
        turn_count=final_state.turn_count,
        winner=final_state.winner,
    )
