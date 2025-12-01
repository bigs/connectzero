from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array


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


class MCTSTree(NamedTuple):
    """
    A flattened, static memory arena for MCTS trees.

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


def expand_mcts_tree_node(
    tree: MCTSTree,
    leaf_index: Array,
    action_to_expand: Array,
) -> tuple[MCTSTree, Array]:
    """
    Expand the tree at the given leaf index and action.

    Args:
        tree: The MCTS tree to expand.
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


class MCTSBackpropState(NamedTuple):
    tree: MCTSTree
    node_index: Array  # dtype=int32
    result: Array  # dtype=int32


def backpropagate_mcts_tree_result(
    tree: MCTSTree,
    initial_leaf_index: Array,
    result: Array,
) -> MCTSTree:
    """
    Backpropagate the results from the leaf node up to the root.

    Args:
        tree: The MCTS tree to update.
        initial_leaf_index: Array, dtype=int32. Index of the leaf node to start backpropagation from.
        result: Array, dtype=int32. Result value to backpropagate (+1, -1, or 0).

    Returns:
        The updated tree with backpropagated statistics.
    """

    def backprop_body(state: MCTSBackpropState) -> MCTSBackpropState:
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

        return MCTSBackpropState(
            tree=next_tree,
            node_index=next_node_index,
            result=next_result,
        )

    final_state = jax.lax.while_loop(
        cond_fun=lambda state: state.node_index != -1,
        body_fun=backprop_body,
        init_val=MCTSBackpropState(tree, initial_leaf_index, result),
    )

    return final_state.tree


def advance_mcts_tree(tree: MCTSTree, action: Array) -> MCTSTree:
    """
    Advance the tree to the next root based on the action.
    If the action leads to an unexpanded node, reset the tree.

    Args:
        tree: The MCTS tree to advance.
        action: Array, dtype=int32. The action to advance to.

    Returns:
        The updated tree with the new root, or a reset tree if the action was invalid.
    """
    next_root_index = tree.children_index[tree.root_index, action]

    # Check if the move is valid (node exists)
    valid_move = next_root_index != -1

    def _reset_mcts_tree(tree: MCTSTree, _: Array) -> MCTSTree:
        """
        Reset the tree to initial state.

        Args:
            tree: The MCTS tree to reset.
            _: Array, dtype=int32. Unused parameter (for compatibility with jax.lax.cond).

        Returns:
            A new initialized tree.
        """
        return MCTSTree.init(
            tree.children_index.shape[0],
            tree.children_index.shape[1],
        )

    def _reroot_mcts_tree(tree: MCTSTree, node_index: Array) -> MCTSTree:
        """
        Reroot the tree to the given node index.

        Args:
            tree: The MCTS tree to reroot.
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
        valid_move, _reroot_mcts_tree, _reset_mcts_tree, tree, next_root_index
    )


def expand_node(
    tree: ArenaTree,
    leaf_index: jnp.ndarray,
    action_to_expand: jnp.ndarray,
) -> tuple[ArenaTree, jnp.ndarray]:
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


class BackpropState(NamedTuple):
    tree: ArenaTree
    node_index: jnp.ndarray
    result: jnp.ndarray


def backpropagate(
    tree: ArenaTree,
    initial_leaf_index: jnp.ndarray,
    results: jnp.ndarray,
) -> ArenaTree:
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


def advance_tree(tree: ArenaTree, action: jnp.ndarray) -> ArenaTree:
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
    reset_tree = ArenaTree.init(
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
