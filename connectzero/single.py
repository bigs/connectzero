from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from connectzero.game import (
    TrainingSample,
    check_winner_single,
    play_move_single,
    trajectory_is_active,
)


class SearchTree(NamedTuple):
    """
    A flattened, static memory arena for a single MCTS tree.

    Shapes:
    N = Capacity (Max nodes per tree)
    A = Action Space (7 for Connect Four)
    """

    ### Graph of moves
    # Index of the child node. -1 indicates unexpanded/leaf.
    children_index: jnp.ndarray  # [N, A], dtype=int32

    # Index of the parent node. -1 for Root.
    parents: jnp.ndarray  # [N], dtype=int32

    # The action taken to get to this node from the parent.
    action_from_parent: jnp.ndarray  # [N], dtype=int32

    ### Statistics
    # Visit counts (N) for each action from this node.
    children_visits: jnp.ndarray  # [N, A], dtype=int32

    # Total accumulated value (W) for each action.
    # Divide by visits to get Q.
    children_values: jnp.ndarray  # [N, A], dtype=float32

    ### Allocator State
    # Tracks the next free slot in the 'N' dimension.
    next_node_index: Array  # dtype=int32

    # Track the floating root
    root_index: Array  # dtype=int32

    @classmethod
    def init(cls, N: int, A: int):
        """Initialize an empty tree with root at index 0."""
        return cls(
            children_index=jnp.full((N, A), -1, dtype=jnp.int32),
            parents=jnp.full((N), -1, dtype=jnp.int32),
            action_from_parent=jnp.full((N), -1, dtype=jnp.int32),
            children_visits=jnp.zeros((N, A), dtype=jnp.int32),
            children_values=jnp.zeros((N, A), dtype=jnp.float32),
            next_node_index=1,
            root_index=0,
        )


class SelectState(NamedTuple):
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


class SelectResult(NamedTuple):
    leaf_index: Array  # dtype=int32
    action_to_expand: Array  # dtype=int32
    board_state: jnp.ndarray  # [6, 7], dtype=int32
    turn_count: Array  # dtype=int32
    winner: Array  # dtype=int32


def select_leaf(tree: SearchTree, board_state: jnp.ndarray) -> SelectResult:
    """
    Select a leaf node to expand.
    """

    c = jnp.sqrt(2)

    def compute_ucb_values(
        tree: SearchTree,
        current_node_index: Array,
        board_state: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute the UCB values for each action.

        Args:
            tree: The SearchTree.
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

    def select_leaf_body(state: SelectState) -> SelectState:
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

        prospective_board_state = play_move_single(
            state.board_state, best_action, (state.turn_count % 2) + 1
        )
        new_board_state = jnp.where(
            child_exists,
            prospective_board_state,
            state.board_state,
        )

        prospective_winner = check_winner_single(
            prospective_board_state, state.turn_count + 1
        )
        new_winner = jnp.where(child_exists, prospective_winner, state.winner)

        is_unfinished = prospective_winner == 0

        # Remain active only if we have not discovered a leaf node
        new_trajectory_active = child_exists & is_unfinished

        # Iterate
        new_turn_count = jnp.where(
            child_exists,
            state.turn_count + 1,
            state.turn_count,
        )

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
        init_val=SelectState.init(
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


def expand_leaf(
    tree: SearchTree,
    leaf_index: Array,
    action_to_expand: Array,
) -> tuple[SearchTree, Array]:
    """
    Expand the tree at the given leaf index and action.

    Args:
        tree: The SearchTree to expand.
        leaf_index: Array, dtype=int32. Index of the leaf node to expand.
        action_to_expand: Array, dtype=int32. Action to expand from the leaf.

    Returns:
        A tuple containing:
        - The updated tree with the new node expanded.
        - Array, dtype=int32. The index of the newly created node.
    """
    next_node_index = tree.next_node_index
    next_next_node_index = next_node_index + 1
    next_children_index = tree.children_index.at[leaf_index, action_to_expand].set(
        next_node_index
    )
    next_parents = tree.parents.at[next_node_index].set(leaf_index)
    next_action_from_parent = tree.action_from_parent.at[next_node_index].set(
        action_to_expand
    )

    return tree._replace(
        next_node_index=next_next_node_index,
        children_index=next_children_index,
        parents=next_parents,
        action_from_parent=next_action_from_parent,
    ), next_node_index


class SimulateState(NamedTuple):
    key: jnp.ndarray
    board_state: jnp.ndarray  # [6, 7], dtype=int32
    turn_count: Array  # dtype=int32
    trajectory_active: Array  # dtype=bool
    winner: Array  # dtype=int32


def simulate_rollout(
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

    def simulate_body(state: SimulateState) -> SimulateState:
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


class BackpropState(NamedTuple):
    tree: SearchTree
    node_index: Array  # dtype=int32
    result: Array  # dtype=int32


def backpropagate(
    tree: SearchTree,
    initial_leaf_index: Array,
    result: Array,
) -> SearchTree:
    """
    Backpropagate the results from the leaf node up to the root.

    Args:
        tree: The SearchTree to update.
        initial_leaf_index: Array, dtype=int32. Index of the leaf node to start backpropagation from.
        result: Array, dtype=int32. Result value to backpropagate (+1, -1, or 0).

    Returns:
        The updated tree with backpropagated statistics.
    """

    def backprop_body(state: BackpropState) -> BackpropState:
        parent = state.tree.parents[state.node_index]
        action = state.tree.action_from_parent[state.node_index]

        # Update node stats on the parent's edge
        updated_children_visits = state.tree.children_visits.at[
            parent,
            action,
        ].add(1)
        updated_children_values = state.tree.children_values.at[
            parent,
            action,
        ].add(state.result)

        should_update = parent != -1

        next_tree = state.tree._replace(
            children_visits=jnp.where(
                should_update, updated_children_visits, state.tree.children_visits
            ),
            children_values=jnp.where(
                should_update, updated_children_values, state.tree.children_values
            ),
        )

        next_node_index = parent
        next_result = state.result * -1

        return BackpropState(
            tree=next_tree,
            node_index=next_node_index,
            result=next_result,
        )

    final_state = jax.lax.while_loop(
        cond_fun=lambda state: state.node_index != -1,
        body_fun=backprop_body,
        init_val=BackpropState(tree, initial_leaf_index, result),
    )

    return final_state.tree


def advance_search(tree: SearchTree, action: Array) -> SearchTree:
    """
    Advance the tree to the next root based on the action.
    If the action leads to an unexpanded node, reset the tree.

    Args:
        tree: The SearchTree to advance.
        action: Array, dtype=int32. The action to advance to.

    Returns:
        The updated tree with the new root, or a reset tree if the action was invalid.
    """
    next_root_index = tree.children_index[tree.root_index, action]

    # Check if the move is valid (node exists)
    valid_move = next_root_index != -1

    def _reset_search_tree(tree: SearchTree, _: Array) -> SearchTree:
        """
        Reset the tree to initial state.

        Args:
            tree: The SearchTree to reset.
            _: Array, dtype=int32. Unused parameter (for compatibility with jax.lax.cond).

        Returns:
            A new initialized tree.
        """
        return SearchTree.init(
            tree.children_index.shape[0],
            tree.children_index.shape[1],
        )

    def _reroot_search_tree(tree: SearchTree, node_index: Array) -> SearchTree:
        """
        Reroot the tree to the given node index.

        Args:
            tree: The SearchTree to reroot.
            node_index: Array, dtype=int32. Index of the new root node.

        Returns:
            The updated tree with the new root.
        """
        next_parents = tree.parents.at[node_index].set(-1)
        return tree._replace(
            root_index=node_index,
            parents=next_parents,
        )

    return jax.lax.cond(
        valid_move, _reroot_search_tree, _reset_search_tree, tree, next_root_index
    )


class MCTSLoopState(NamedTuple):
    key: jnp.ndarray
    tree: SearchTree


@jax.jit(static_argnames=["num_simulations"])
def run_mcts_search(
    tree: SearchTree, board_state: jnp.ndarray, num_simulations: int, key: jnp.ndarray
) -> tuple[SearchTree, jnp.ndarray, jnp.ndarray]:
    """
    Run MCTS search on the given tree and board state (single game version).

    Args:
        tree: The SearchTree.
        board_state: [6, 7] array, dtype=int32.
        num_simulations: Number of MCTS iterations.
        key: JAX PRNG key.

    Returns:
        tuple[SearchTree, jnp.ndarray, jnp.ndarray, TrainingSample]:
        - Updated SearchTree (advanced to the chosen action).
        - The chosen action (int32).
        - The new board state (after chosen action).
        - The training sample.
    """

    def mcts_step(i: int, state: MCTSLoopState) -> MCTSLoopState:
        key, tree = state.key, state.tree
        # Select
        select_result = select_leaf(tree, board_state)

        current_is_terminal = (select_result.winner != 0) | (
            select_result.turn_count >= 42
        )
        should_expand = (select_result.action_to_expand >= 0) & ~current_is_terminal

        # Candidate expansion
        expanded_tree, new_node_idx = expand_leaf(
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
        results = simulate_rollout(subkey, sim_board, sim_turns)

        # Backpropagate
        tree = backpropagate(tree, target_node_idx, results)

        return MCTSLoopState(key=key, tree=tree)

    final_state: MCTSLoopState = jax.lax.fori_loop(
        0, num_simulations, mcts_step, MCTSLoopState(key=key, tree=tree)
    )

    root_visits = final_state.tree.children_visits[final_state.tree.root_index, :]
    best_action = jnp.argmax(root_visits)

    turn_count = jnp.sum(jnp.where(board_state == 0, 0, 1))

    # Extract training data
    sample = extract_training_data(final_state.tree, board_state, turn_count)

    player_who_plays = (turn_count % 2) + 1
    new_board_state = play_move_single(board_state, best_action, player_who_plays)

    next_tree = advance_search(final_state.tree, best_action)

    return next_tree, best_action, new_board_state, sample


def extract_training_data(
    tree: SearchTree, board_state: jnp.ndarray, turn_count: jnp.ndarray
) -> TrainingSample:
    """
    Extract training data from the given tree and board state.

    Args:
        tree: The SearchTree.
        board_state: [6, 7] array, dtype=int32.
        turn_count: Array, dtype=int32. Current turn count.

    Returns:
        TrainingSample: The training sample.
    """
    root_visits = tree.children_visits[tree.root_index, :]
    total_visits = jnp.sum(root_visits)
    safe_total_visits = jnp.maximum(total_visits, 1)
    policy_target = root_visits / safe_total_visits
    value_target = jnp.sum(tree.children_values[tree.root_index, :]) / safe_total_visits
    return TrainingSample(board_state, policy_target, value_target, turn_count)
