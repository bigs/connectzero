from typing import NamedTuple

import jax
import jax.numpy as jnp

# B = Batch size (e.g., 128 games)
B = 128
# N = Max simulations (e.g., 800 nodes)
N = 800
# A = Actions (7 columns)
A = 7


class ArenaTree(NamedTuple):
    """
    A flattened, static memory arena for MCTS trees.

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

    # --- Allocator State ---
    # Tracks the next free slot in the 'N' dimension for each game.
    next_node_index: jnp.ndarray  # [B], dtype=int32

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
    key: jnp.ndarray  # PRNG key

    @classmethod
    def init(cls, B: int, key: jnp.ndarray, board_state: jnp.ndarray):
        return cls(
            current_node_index=jnp.zeros((B,), dtype=jnp.int32),
            next_action=jnp.zeros((B,), dtype=jnp.int32),
            trajectory_active=jnp.ones((B,), dtype=jnp.bool_),
            board_state=board_state,
            turn_count=jnp.zeros((B,), dtype=jnp.int32),
            key=key,
        )


def play_move(
    board_state: jnp.ndarray, action: jnp.ndarray, player_id: int
) -> jnp.ndarray:
    """
    Play the action (column) on the board for each item in the batch.
    Assumes 0=Bottom, 5=Top. Fills from 0 upwards.
    """
    batch_range = jnp.arange(board_state.shape[0])
    selected_columns = board_state[batch_range, :, action]
    target_rows = jnp.argmax(selected_columns == 0, axis=1)

    return board_state.at[batch_range, target_rows, action].set(player_id)


def select_leaf(
    tree: ArenaTree, board_state: jnp.ndarray, key: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Select a leaf node to expand.
    """

    def compute_ucb_values(
        tree: ArenaTree, current_node_index: jnp.ndarray, key: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute the UCB values for each action.
        Currently returns random noise (uniform random [0, 1)) for each action.
        """
        batch_size = tree.children_index.shape[0]
        action_space_size = tree.children_index.shape[2]

        # Generate random values for each action in the batch
        random_values = jax.random.uniform(
            key, shape=(batch_size, action_space_size), minval=0.0, maxval=1.0
        )

        return random_values

    # Don't compute this in every iteration, dummy
    batch_range = jnp.arange(tree.children_index.shape[0])

    def select_leaf_body(state: SelectState) -> SelectState:
        key, subkey = jax.random.split(state.key)

        ucb_values = compute_ucb_values(tree, state.current_node_index, subkey)
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

        # Remain active only if we have not discovered a leaf node
        new_trajectory_active = state.trajectory_active & child_exists

        # Iterate
        new_board_state = jnp.where(
            # Only update board if we are taking a step deeper (child exists)
            new_trajectory_active[:, None, None],
            play_move(
                state.board_state, best_action, (state.turn_count % 2) + 1
            ),  # Player 1 or 2
            state.board_state,
        )

        new_turn_count = jnp.where(
            new_trajectory_active,
            state.turn_count + 1,
            state.turn_count,
        )

        return state._replace(
            current_node_index=new_current_node_index,
            next_action=new_next_action,
            trajectory_active=new_trajectory_active,
            board_state=new_board_state,
            turn_count=new_turn_count,
            key=key,
        )

    final_state = jax.lax.while_loop(
        cond_fun=lambda state: jnp.any(state.trajectory_active),
        body_fun=select_leaf_body,
        init_val=SelectState.init(
            B=tree.children_index.shape[0], key=key, board_state=board_state
        ),
    )

    return final_state.current_node_index, final_state.next_action


def expand_node(
    tree: ArenaTree,
    leaf_index: jnp.ndarray,
    action_to_expand: jnp.ndarray,
) -> ArenaTree:
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

    return tree._replace(
        children_index=next_children_index,
        parents=next_parents,
        action_from_parent=next_action_from_parent,
        next_node_index=next_next_node_index,
    )


def run_mcts_search(
    board_state: jnp.ndarray, num_simulations: int, key: jnp.ndarray
) -> tuple[ArenaTree, jnp.ndarray]:
    """
    Run MCTS search on the given tree and board state.
    """
    tree = ArenaTree.init(B=1, N=num_simulations + 1, A=7)
    for _ in range(num_simulations):
        key, subkey = jax.random.split(key)
        leaf_index, action_to_expand = select_leaf(tree, board_state, subkey)
        print(f"Leaf index: {leaf_index}")
        print(f"Action to expand: {action_to_expand}")
        tree = expand_node(tree, leaf_index, action_to_expand)

    # Stub return value
    return tree, action_to_expand


def main():
    # Initialize a board with two moves in the middle column
    starting_board_state = jnp.zeros((6, 7), dtype=jnp.int32)
    starting_board_state = starting_board_state.at[0, 3].set(1)
    starting_board_state = starting_board_state.at[1, 3].set(2)

    key = jax.random.PRNGKey(1)

    tree, board_at_leaf = run_mcts_search(
        board_state=starting_board_state, num_simulations=800, key=key
    )
    print(board_at_leaf)


if __name__ == "__main__":
    main()
