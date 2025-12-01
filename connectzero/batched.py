from typing import NamedTuple

import jax
import jax.numpy as jnp

from connectzero.game import check_winner, play_move, trajectories_active


class BatchedSearchTree(NamedTuple):
    """
    A flattened, static memory arena for a batch of MCTS trees.

    Shapes:
    B = Batch Size (Number of parallel games)
    N = Capacity (Max nodes per tree)
    A = Action Space (7 for Connect Four)
    """

    ### Graph of moves
    # Index of the child node. -1 indicates unexpanded/leaf.
    children_index: jnp.ndarray  # [B, N, A], dtype=int32

    # Index of the parent node. -1 for Root.
    parents: jnp.ndarray  # [B, N], dtype=int32

    # The action taken to get to this node from the parent.
    action_from_parent: jnp.ndarray  # [B, N], dtype=int32

    ### Statistics
    # Visit counts (N) for each action from this node.
    children_visits: jnp.ndarray  # [B, N, A], dtype=int32

    # Total accumulated value (W) for each action.
    # Divide by visits to get Q.
    children_values: jnp.ndarray  # [B, N, A], dtype=float32

    ### Allocator State
    # Tracks the next free slot in the 'N' dimension for each game.
    next_node_index: jnp.ndarray  # [B], dtype=int32

    # Track the floating root
    root_index: jnp.ndarray  # [B], dtype=int32

    @classmethod
    def init(cls, B: int, N: int, A: int):
        """Initialize an empty tree with root at index 0."""
        return cls(
            children_index=jnp.full((B, N, A), -1, dtype=jnp.int32),
            parents=jnp.full((B, N), -1, dtype=jnp.int32),
            action_from_parent=jnp.full((B, N), -1, dtype=jnp.int32),
            children_visits=jnp.zeros((B, N, A), dtype=jnp.int32),
            children_values=jnp.zeros((B, N, A), dtype=jnp.float32),
            # Start at 1, because 0 is reserved for the Root
            next_node_index=jnp.ones((B,), dtype=jnp.int32),
            root_index=jnp.zeros((B,), dtype=jnp.int32),
        )


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


def select_leaf(tree: BatchedSearchTree, board_state: jnp.ndarray) -> SelectResult:
    """
    Select a leaf node to expand.
    """

    c = jnp.sqrt(2)

    def compute_ucb_values(
        tree: BatchedSearchTree,
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

    # Don't compute this in every iteration
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


def expand_leaf(
    tree: BatchedSearchTree,
    leaf_index: jnp.ndarray,
    action_to_expand: jnp.ndarray,
) -> tuple[BatchedSearchTree, jnp.ndarray]:
    """
    Expand the tree at the given leaf index and action.
    """
    batch_range = jnp.arange(tree.children_index.shape[0])
    new_node_idx = tree.next_node_index

    next_children_index = tree.children_index.at[
        batch_range, leaf_index, action_to_expand
    ].set(new_node_idx)

    next_parents = tree.parents.at[batch_range, new_node_idx].set(leaf_index)

    next_action_from_parent = tree.action_from_parent.at[batch_range, new_node_idx].set(
        action_to_expand
    )

    next_next_node_index = new_node_idx + 1

    new_tree = tree._replace(
        children_index=next_children_index,
        parents=next_parents,
        action_from_parent=next_action_from_parent,
        next_node_index=next_next_node_index,
    )
    return new_tree, new_node_idx


class SimulateState(NamedTuple):
    key: jnp.ndarray
    board_state: jnp.ndarray
    turn_count: jnp.ndarray
    trajectory_active: jnp.ndarray
    winner: jnp.ndarray


def simulate_rollout(
    key: jnp.ndarray,
    board_state: jnp.ndarray,
    turn_count: jnp.ndarray,
) -> jnp.ndarray:
    """
    Simulate random games from the given board states to completion.
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


class BackpropState(NamedTuple):
    tree: BatchedSearchTree
    node_index: jnp.ndarray
    result: jnp.ndarray


def backpropagate(
    tree: BatchedSearchTree,
    initial_leaf_index: jnp.ndarray,
    results: jnp.ndarray,
) -> BatchedSearchTree:
    """
    Backpropagate the results from the leaf node up to the root.
    """
    batch_range = jnp.arange(tree.children_index.shape[0])

    def backprop_body(state: BackpropState) -> BackpropState:
        parents = state.tree.parents[batch_range, state.node_index]
        actions = state.tree.action_from_parent[batch_range, state.node_index]

        # Update node stats on the parent's edge
        updated_children_visits = state.tree.children_visits.at[
            batch_range,
            parents,
            actions,
        ].add(1)
        updated_children_values = state.tree.children_values.at[
            batch_range,
            parents,
            actions,
        ].add(state.result)

        # Mask updates for finished threads (node_index == -1)
        should_update = (state.node_index != -1) & (parents != -1)

        next_tree = state.tree._replace(
            children_visits=jnp.where(
                should_update[:, None, None],
                updated_children_visits,
                state.tree.children_visits,
            ),
            children_values=jnp.where(
                should_update[:, None, None],
                updated_children_values,
                state.tree.children_values,
            ),
        )

        next_node_index = jnp.where(state.node_index != -1, parents, -1)
        next_result = state.result * -1

        return BackpropState(
            tree=next_tree,
            node_index=next_node_index,
            result=next_result,
        )

    final_state = jax.lax.while_loop(
        cond_fun=lambda state: jnp.any(state.node_index != -1),
        body_fun=backprop_body,
        init_val=BackpropState(tree, initial_leaf_index, results),
    )

    return final_state.tree


def advance_search(tree: BatchedSearchTree, action: jnp.ndarray) -> BatchedSearchTree:
    """
    Advance the tree to the next root based on the action.
    If the action leads to an unexpanded node, reset the tree.
    """
    batch_range = jnp.arange(tree.children_index.shape[0])
    next_root_index = tree.children_index[batch_range, tree.root_index, action]

    # Check if the move is valid (node exists)
    valid_move = next_root_index != -1

    # Prepare Advanced Tree
    # Clamp index to 0 to avoid index error, but the result will be ignored if invalid.
    safe_next_root_index = jnp.maximum(next_root_index, 0)

    next_parents = tree.parents.at[batch_range, safe_next_root_index].set(-1)
    advanced_tree = tree._replace(
        root_index=safe_next_root_index,
        parents=next_parents,
    )

    # Prepare Reset Tree
    reset_tree = BatchedSearchTree.init(
        tree.children_index.shape[0],
        tree.children_index.shape[1],
        tree.children_index.shape[2],
    )

    # Select between advanced and reset based on valid_move
    def select(adv, rst):
        # Broadcast valid_move to match data shape
        shape_diff = adv.ndim - valid_move.ndim
        mask = valid_move.reshape(valid_move.shape + (1,) * shape_diff)
        return jnp.where(mask, adv, rst)

    return jax.tree.map(select, advanced_tree, reset_tree)


class MCTSLoopState(NamedTuple):
    key: jnp.ndarray
    tree: BatchedSearchTree


@jax.jit(static_argnames=["num_simulations"])
def run_mcts_search(
    tree: BatchedSearchTree,
    board_state: jnp.ndarray,
    num_simulations: int,
    key: jnp.ndarray,
) -> tuple[BatchedSearchTree, jnp.ndarray, jnp.ndarray]:
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
        expanded_tree, new_node_idx = expand_leaf(
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
        results = simulate_rollout(subkey, sim_board, sim_turns)

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

    next_tree = advance_search(final_state.tree, best_action)

    return next_tree, best_action, new_board_state
