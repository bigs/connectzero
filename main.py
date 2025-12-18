import argparse
import glob
import json
import math
import os
import re
import time
from collections import deque
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from tqdm import tqdm

from connectzero import batched, single
from connectzero.game import (
    TrainingSample,
    check_winner,
    check_winner_single,
    get_parquet_row_count,
    play_move,
    play_move_single,
    print_board_state,
    print_board_states,
    save_trajectories,
)
from connectzero.model import ConnectZeroModel, load, save
from connectzero.train import get_learning_rate, get_optimizer, train_epoch, train_loop

jax.config.update("jax_default_matmul_precision", "tensorfloat32")


def process_single_game_history(
    history: list[TrainingSample],
    final_board_state: jnp.ndarray,
    final_turn_count: jnp.ndarray,
) -> list[TrainingSample]:
    """
    Process game history to set value targets based on game outcome.

    Uses negamax-style backwards propagation: the value target for each
    sample is set based on whether the player whose turn it was at that
    state eventually won, lost, or drew.

    Args:
        history: List of TrainingSample with zeroed value_target.
        final_board_state: [6, 7] array, the final board state.
        final_turn_count: The turn count at game end.

    Returns:
        List of TrainingSample with value_target set.
    """
    winner = int(check_winner_single(final_board_state, final_turn_count))
    start_turn = int(final_turn_count) - len(history)

    processed = []
    for i, sample in enumerate(history):
        turn_count = start_turn + i
        current_player = (turn_count % 2) + 1

        if winner == 3:  # Draw
            value = 0.0
        elif winner == current_player:  # Current player won
            value = 1.0
        else:  # Current player lost
            value = -1.0

        updated_sample = sample._replace(
            value_target=jnp.array([value], dtype=jnp.float32)
        )
        processed.append(updated_sample)

    return processed


def process_batched_game_history(
    history: list[TrainingSample] | TrainingSample,
    final_board_state: jnp.ndarray,
    final_turn_count: jnp.ndarray,
) -> list[TrainingSample]:
    """
    Process batched game history to set value targets based on game outcomes.

    Uses negamax-style backwards propagation: the value target for each
    sample is set based on whether the player whose turn it was at that
    state eventually won, lost, or drew. Filters out samples from games
    that had already finished (when batched games end at different times).

    Args:
        history: Either:
            - A list of TrainingSample where each sample is batched:
              board_state [B, 3, 6, 7], policy_target [B, 7], value_target [B].
            - A single stacked TrainingSample with leading time dimension:
              board_state [T, B, 3, 6, 7], policy_target [T, B, 7], value_target [T, B].
        final_board_state: [B, 6, 7] array, the final board states.
        final_turn_count: [B] array, the turn counts at game end.

    Returns:
        List containing a single TrainingSample with all valid samples
        batched together (shape [num_valid, ...]).
    """
    B = final_board_state.shape[0]  # Batch size
    if isinstance(history, TrainingSample):
        all_board_states = history.board_state
        all_policy_targets = history.policy_target
        T = all_board_states.shape[0]  # Number of time steps
    else:
        if not history:
            return []
        T = len(history)  # Number of time steps
        # Stack history into tensors [T, B, ...]
        all_board_states = jnp.stack(
            [s.board_state for s in history]
        )  # [T, B, 3, 6, 7]
        all_policy_targets = jnp.stack([s.policy_target for s in history])  # [T, B, 7]

    # Compute winners for all games
    winners = check_winner(final_board_state, final_turn_count)  # [B]

    # Create validity mask: sample at step t is valid only if t < final_turn_count[b]
    # This filters out samples collected after a game has already ended
    step_indices = jnp.arange(T)[:, None]  # [T, 1]
    valid_mask = step_indices < final_turn_count[None, :]  # [T, B]

    # Compute value targets vectorized
    # current_player at step t is (t % 2) + 1
    current_player = (step_indices % 2) + 1  # [T, 1]

    # Value is +1 if current player won, -1 if lost, 0 if draw
    # winners[b] == 3 means draw, winners[b] == current_player means win
    value_targets = jnp.where(
        winners[None, :] == 3,
        0.0,
        jnp.where(winners[None, :] == current_player, 1.0, -1.0),
    )  # [T, B]

    # Flatten [T, B, ...] -> [T*B, ...]
    flat_boards = all_board_states.reshape(T * B, 3, 6, 7)
    flat_policies = all_policy_targets.reshape(T * B, 7)
    flat_values = value_targets.reshape(T * B)
    flat_mask = valid_mask.reshape(T * B)

    # Move to CPU for filtering
    flat_boards = jax.device_get(flat_boards)
    flat_policies = jax.device_get(flat_policies)
    flat_values = jax.device_get(flat_values)
    flat_mask = jax.device_get(flat_mask)

    # Filter using boolean indexing (on CPU/numpy)
    valid_boards = flat_boards[flat_mask]  # [num_valid, 3, 6, 7]
    valid_policies = flat_policies[flat_mask]  # [num_valid, 7]
    valid_values = flat_values[flat_mask]  # [num_valid]

    # Return as a single batched TrainingSample
    return [
        TrainingSample(
            board_state=jnp.array(valid_boards, dtype=jnp.float32),
            policy_target=jnp.array(valid_policies, dtype=jnp.float32),
            value_target=jnp.array(valid_values, dtype=jnp.float32),
        )
    ]


class BatchedSelfPlayState(NamedTuple):
    key: jnp.ndarray
    tree: batched.BatchedSearchTree
    board_state: jnp.ndarray
    turn_count: jnp.ndarray
    t: jnp.ndarray
    history_board_state: jnp.ndarray
    history_policy_target: jnp.ndarray
    history_value_target: jnp.ndarray
    any_unfinished: jnp.ndarray


@eqx.filter_jit
def run_batched_selfplay_while_loop(
    key: jnp.ndarray,
    tree: batched.BatchedSearchTree,
    board_state: jnp.ndarray,
    num_simulations: int,
    model_tuple: tuple[ConnectZeroModel, eqx.nn.State] | None,
    temperature: float,
    temperature_depth: int,
    dirichlet_alpha: float,
    dirichlet_epsilon: float,
    max_moves: int = 42,
) -> tuple[
    jnp.ndarray,
    batched.BatchedSearchTree,
    jnp.ndarray,
    jnp.ndarray,
    TrainingSample,
    jnp.ndarray,
]:
    """
    Play a full batched self-play game using a JAX while_loop.

    Returns:
        key: Updated PRNG key
        tree: Final tree
        board_state: Final board state [B, 6, 7]
        turn_count: Final turn counts [B]
        history: Stacked TrainingSample with shapes:
            board_state [max_moves, B, 3, 6, 7]
            policy_target [max_moves, B, 7]
            value_target [max_moves, B]
        t: Number of executed moves (max over batch), scalar int32
    """
    turn_count = jnp.count_nonzero(board_state, axis=(1, 2))
    B = board_state.shape[0]

    history_board_state = jnp.zeros((max_moves, B, 3, 6, 7), dtype=jnp.float32)
    history_policy_target = jnp.zeros((max_moves, B, 7), dtype=jnp.float32)
    history_value_target = jnp.zeros((max_moves, B), dtype=jnp.float32)

    any_unfinished = jnp.any(check_winner(board_state, turn_count) == 0)

    init_state = BatchedSelfPlayState(
        key=key,
        tree=tree,
        board_state=board_state,
        turn_count=turn_count,
        t=jnp.array(0, dtype=jnp.int32),
        history_board_state=history_board_state,
        history_policy_target=history_policy_target,
        history_value_target=history_value_target,
        any_unfinished=any_unfinished,
    )

    def cond_fun(state: BatchedSelfPlayState) -> jnp.ndarray:
        return (state.t < max_moves) & state.any_unfinished

    def body_fun(state: BatchedSelfPlayState) -> BatchedSelfPlayState:
        key, subkey = jax.random.split(state.key)

        tree, _best_action, board_state, sample = batched.run_mcts_search(
            state.tree,
            state.board_state,
            num_simulations,
            jnp.sqrt(2),
            subkey,
            model_tuple,
            temperature,
            temperature_depth,
            dirichlet_alpha,
            dirichlet_epsilon,
        )

        turn_count = jnp.count_nonzero(board_state, axis=(1, 2))

        history_board_state = state.history_board_state.at[state.t].set(
            sample.board_state
        )
        history_policy_target = state.history_policy_target.at[state.t].set(
            sample.policy_target
        )
        history_value_target = state.history_value_target.at[state.t].set(
            sample.value_target
        )

        any_unfinished = jnp.any(check_winner(board_state, turn_count) == 0)

        return state._replace(
            key=key,
            tree=tree,
            board_state=board_state,
            turn_count=turn_count,
            t=state.t + 1,
            history_board_state=history_board_state,
            history_policy_target=history_policy_target,
            history_value_target=history_value_target,
            any_unfinished=any_unfinished,
        )

    final_state = jax.lax.while_loop(cond_fun, body_fun, init_state)

    history = TrainingSample(
        board_state=final_state.history_board_state,
        policy_target=final_state.history_policy_target,
        value_target=final_state.history_value_target,
    )

    return (
        final_state.key,
        final_state.tree,
        final_state.board_state,
        final_state.turn_count,
        history,
        final_state.t,
    )


run_search_vmap = jax.vmap(
    single.run_mcts_search, in_axes=(0, 0, None, None, 0, None, None, None, None, None)
)
advance_search_vmap = jax.vmap(single.advance_search, in_axes=(0, 0))


def handle_player_turn(
    tree: batched.BatchedSearchTree, board_state: jnp.ndarray, turn_count: jnp.ndarray
) -> tuple[batched.BatchedSearchTree, jnp.ndarray]:
    col = -1
    batch_size = board_state.shape[0]
    while col < 0 or col > 6:
        try:
            line = input(f"Your move (0-6) [Turn {int(turn_count[0])}]: ")
            col = int(line)
            if col < 0 or col > 6:
                print("Invalid column. Please enter 0-6.")
                continue
            # Check if column is full on ANY board
            # board_state is [B, 6, 7], we check if top row (0) has any non-zero
            if jnp.any(board_state[:, 0, col] != 0):
                print("Column is full on at least one board.")
                col = -1
        except ValueError:
            print("Invalid input.")

    # Broadcast action to batch size
    action = jnp.full((batch_size,), col, dtype=jnp.int32)

    # Advance tree first (using current root)
    tree = batched.advance_search(tree, action)

    # Play move
    player_who_plays = (turn_count % 2) + 1
    board_state = play_move(board_state, action, player_who_plays)

    print_board_states(board_state)
    print("Player played:", col)

    return tree, board_state


def handle_player_turn_single(
    tree: single.SearchTree, board_state: jnp.ndarray, turn_count: Array
) -> tuple[single.SearchTree, jnp.ndarray]:
    col = -1
    while col < 0 or col > 6:
        try:
            line = input(f"Your move (0-6) [Turn {int(turn_count)}]: ")
            col = int(line)
            if col < 0 or col > 6:
                print("Invalid column. Please enter 0-6.")
                continue
            if board_state[0, col] != 0:
                print("Column is full.")
                col = -1
        except ValueError:
            print("Invalid input.")

    action = jnp.array(col, dtype=jnp.int32)
    tree = single.advance_search(tree, action)
    player_who_plays = (turn_count % 2) + 1
    board_state = play_move_single(board_state, action, player_who_plays)

    print_board_state(board_state)
    print("Player played:", col)
    return tree, board_state


def get_unique_filename(directory: str) -> str:
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    timestamp = int(time.time())
    filename = f"trajectory_{timestamp}.parquet"
    full_path = os.path.join(directory, filename)

    while os.path.exists(full_path):
        timestamp += 1
        filename = f"trajectory_{timestamp}.parquet"
        full_path = os.path.join(directory, filename)

    return full_path


def find_latest_checkpoint(directory: str) -> str | None:
    """
    Find the checkpoint with the highest step count in the given directory.

    Looks for files matching the pattern checkpoint_{steps}_steps.eqx and
    returns the path to the one with the highest step count.

    Args:
        directory: Path to the checkpoint directory.

    Returns:
        Full path to the latest checkpoint, or None if no checkpoints found.
    """
    if not os.path.exists(directory):
        return None

    pattern = os.path.join(directory, "checkpoint_*_steps.eqx")
    checkpoint_files = glob.glob(pattern)

    if not checkpoint_files:
        return None

    # Parse step counts and find the maximum
    best_checkpoint = None
    best_steps = -1

    for filepath in checkpoint_files:
        filename = os.path.basename(filepath)
        match = re.match(r"checkpoint_(\d+)_steps\.eqx", filename)
        if match:
            steps = int(match.group(1))
            if steps > best_steps:
                best_steps = steps
                best_checkpoint = filepath

    return best_checkpoint


def get_next_data_subdir(data_dir: str) -> str:
    """
    Create and return the next sequential data subdirectory.

    Looks for existing directories matching the pattern setXXX and creates
    the next one in sequence (e.g., if set002 exists, creates set003).

    Args:
        data_dir: Base directory for training data.

    Returns:
        Full path to the newly created subdirectory.
    """
    os.makedirs(data_dir, exist_ok=True)

    # Find existing setXXX directories
    existing_sets = []
    for entry in os.listdir(data_dir):
        full_path = os.path.join(data_dir, entry)
        if os.path.isdir(full_path):
            match = re.match(r"set(\d+)", entry)
            if match:
                existing_sets.append(int(match.group(1)))

    # Determine next set number
    next_num = max(existing_sets) + 1 if existing_sets else 0

    # Create the new directory
    new_dir = os.path.join(data_dir, f"set{next_num:03d}")
    os.makedirs(new_dir, exist_ok=True)

    return new_dir


def select_training_files(data_dir: str, target_buffer_length: int) -> list[str]:
    """
    Select the newest parquet files until the target row count is met.
    """
    parquet_files = []
    for root, _, files in os.walk(data_dir):
        for filename in files:
            if filename.endswith(".parquet"):
                parquet_files.append(os.path.join(root, filename))

    # Sort newest-first by lexical order (trajectory_{timestamp}.parquet)
    parquet_files.sort(reverse=True)

    selected_files: list[str] = []
    total_rows = 0

    for path in parquet_files:
        row_count = get_parquet_row_count(path)
        if row_count <= 0:
            continue
        selected_files.append(path)
        total_rows += row_count
        if total_rows >= target_buffer_length:
            break

    return selected_files


def run_simulate(args, parser):
    key = jax.random.PRNGKey(args.seed)
    num_simulations = args.simulations

    model_tuple = None
    if args.puct:
        if args.checkpoint:
            print(f"Loading checkpoint from {args.checkpoint}")
            model, model_state, _, _ = load(args.checkpoint)
        else:
            print("Using PUCT with randomly initialized neural network")
            key, model_key = jax.random.split(key)
            model, model_state = eqx.nn.make_with_state(ConnectZeroModel)(model_key)

        model = eqx.nn.inference_mode(model)
        model_tuple = (model, model_state)
    elif args.checkpoint:
        parser.error("--checkpoint requires --puct to be set")

    for i in range(args.iterations):
        if args.iterations > 1:
            print(f"Starting iteration {i + 1}/{args.iterations}")

        replay_buffer = deque(maxlen=100000)
        game_history = []

        if args.single:
            if args.batch > 1:
                print("Running new engine in batch mode")
                board_state = jnp.zeros((args.batch, 6, 7), dtype=jnp.int32)
                print("Initial Board State:")
                print_board_states(board_state)

                turn_count = jnp.count_nonzero(board_state, axis=(1, 2))
                tree_prototype = single.SearchTree.init(N=num_simulations * 42 + 1, A=7)
                tree = jax.tree.map(
                    lambda x: jnp.stack([x] * args.batch), tree_prototype
                )

                # In --single mode with batching, we need to ensure randomness.
                # run_search_vmap uses the same key for all items in the batch if we pass a single key.
                # We need to pass a batch of keys.

                while jnp.any(check_winner(board_state, turn_count) == 0):
                    key, subkey = jax.random.split(key)
                    batch_keys = jax.random.split(subkey, args.batch)
                    # is_player_turn = is_interactive & (jnp.any(turn_count % 2 != 0))

                    tree, best_action, board_state, sample = run_search_vmap(
                        tree,
                        board_state,
                        num_simulations,
                        jnp.sqrt(2),
                        batch_keys,
                        model_tuple,
                        args.temperature,  # Pass temperature
                        args.temperature_depth,
                        0.3,  # dirichlet_alpha
                        0.25,  # dirichlet_epsilon
                    )

                    # Collect samples
                    # sample is a TrainingSample where each field is [B, ...]
                    # We convert to CPU immediately
                    game_history.append(jax.device_get(sample))

                    turn_count = jnp.count_nonzero(board_state, axis=(1, 2))
                    print_board_states(board_state)

                # Game Over
                print("Game Over")
                # Process game history to set value targets
                processed_history = process_batched_game_history(
                    game_history, board_state, turn_count
                )

                # Save to disk
                if args.out:
                    filename = get_unique_filename(args.out)
                    save_trajectories(processed_history, filename)

            else:
                # Single Game Mode
                print("Running in Single Game Mode")
                board_state = jnp.zeros((6, 7), dtype=jnp.int32)
                print("Initial Board State:")
                print_board_state(board_state)

                turn_count = jnp.count_nonzero(board_state)
                tree = single.SearchTree.init(N=num_simulations * 42 + 1, A=7)

                while check_winner_single(board_state, turn_count) == 0:
                    key, subkey = jax.random.split(key)
                    is_player_turn = args.interactive and (int(turn_count) % 2 != 0)

                    if is_player_turn:
                        tree, board_state = handle_player_turn_single(
                            tree, board_state, turn_count
                        )
                        turn_count = jnp.count_nonzero(board_state)
                    else:
                        tree, best_action, board_state, sample = single.run_mcts_search(
                            tree,
                            board_state,
                            num_simulations,
                            jnp.sqrt(2),
                            subkey,
                            model_tuple,
                            args.temperature,  # Pass temperature
                            args.temperature_depth,
                            0.3,  # dirichlet_alpha
                            0.25,  # dirichlet_epsilon
                        )
                        game_history.append(jax.device_get(sample))

                        turn_count = jnp.count_nonzero(board_state)
                        print_board_state(board_state)
                        print("Best Action:", best_action)

                # Game Over
                print("Game Over")
                processed_history = process_single_game_history(
                    game_history, board_state, turn_count
                )
                for s in processed_history:
                    replay_buffer.append(s)

                if args.out:
                    filename = get_unique_filename(args.out)
                    save_trajectories(list(replay_buffer), filename)

        else:
            # Batch Mode
            if args.interactive and args.batch > 1:
                print(
                    "Interactive mode only supports batch size 1. Setting batch size to 1."
                )
                args.batch = 1

            print(f"Running in Batch Mode (B={args.batch})")
            starting_board_state = jnp.zeros((args.batch, 6, 7), dtype=jnp.int32)
            print("Initial Board State:")
            print_board_states(starting_board_state)

            batch_size = starting_board_state.shape[0]
            board_state = starting_board_state

            turn_count = jnp.count_nonzero(board_state, axis=(1, 2))
            tree = batched.BatchedSearchTree.init(
                B=batch_size, N=num_simulations * 42 + 1, A=7
            )

            while jnp.any(check_winner(board_state, turn_count) == 0):
                key, subkey = jax.random.split(key)
                is_player_turn = args.interactive and (int(turn_count[0]) % 2 != 0)

                if is_player_turn:
                    tree, board_state = handle_player_turn(
                        tree, board_state, turn_count
                    )
                    turn_count = jnp.count_nonzero(board_state, axis=(1, 2))
                else:
                    tree, best_action, board_state, sample = batched.run_mcts_search(
                        tree,
                        board_state,
                        num_simulations,
                        jnp.sqrt(2),
                        subkey,
                        model_tuple,
                        args.temperature,
                        args.temperature_depth,
                        1.0,
                        0.25,
                    )
                    game_history.append(jax.device_get(sample))

                    turn_count = jnp.count_nonzero(board_state, axis=(1, 2))
                    print_board_states(board_state)
                    print("Best Action:", best_action)

            # Game Over
            print("Game Over")
            # Process game history to set value targets
            processed_history = process_batched_game_history(
                game_history, board_state, turn_count
            )

            if args.out:
                filename = get_unique_filename(args.out)
                save_trajectories(processed_history, filename)


def run_initialize(args):
    key = jax.random.PRNGKey(args.seed)

    # Default hyperparameters
    num_blocks = 7

    print(
        f"Initializing model with seed {args.seed} and {num_blocks} residual blocks..."
    )
    model, state = eqx.nn.make_with_state(ConnectZeroModel)(key, num_blocks=num_blocks)

    # Ensure directory exists
    dirname = os.path.dirname(args.path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    hyperparams = {
        "num_blocks": num_blocks,
    }
    save(args.path, hyperparams, 0, model, state)
    print(f"Model saved to {args.path}")


def run_train(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    train_loop(
        checkpoint_path=args.checkpoint_path,
        data_pattern=args.data_pattern,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
    )


def _resolve_checkpoint_input(path_or_name: str, default_dir: str) -> str:
    """
    Resolve an input checkpoint arg that might be a full path or just a filename.

    - If `path_or_name` includes a directory component (or is absolute), it is
      treated as a path as-is.
    - Otherwise, if it doesn't exist in CWD, fall back to `{default_dir}/{name}`.
    """
    if os.path.isabs(path_or_name) or os.path.dirname(path_or_name):
        return path_or_name
    if os.path.exists(path_or_name):
        return path_or_name
    return os.path.join(default_dir, path_or_name)


def _resolve_checkpoint_output(path_or_name: str, default_dir: str) -> str:
    """
    Resolve an output checkpoint arg that might be a full path or just a filename.

    - If `path_or_name` includes a directory component (or is absolute), it is
      treated as a path as-is.
    - Otherwise, it is written under `{default_dir}/{name}`.
    """
    if os.path.isabs(path_or_name) or os.path.dirname(path_or_name):
        return path_or_name
    return os.path.join(default_dir, path_or_name)


def run_init_optimizer(args, parser):
    default_checkpoint_dir = "./checkpoints"
    input_path = _resolve_checkpoint_input(
        args.input_checkpoint, default_checkpoint_dir
    )
    output_path = _resolve_checkpoint_output(
        args.output_checkpoint, default_checkpoint_dir
    )

    if not os.path.exists(input_path):
        parser.error(
            f"Input checkpoint not found: {args.input_checkpoint!r} "
            f"(resolved to {input_path})"
        )

    if os.path.abspath(input_path) == os.path.abspath(output_path):
        parser.error("Input and output checkpoint paths must be different.")

    # Load hyperparams header (we'll preserve it, but overwrite step/has_opt_state on save).
    with open(input_path, "rb") as f:
        header_line = f.readline().decode()
    try:
        hyperparams = json.loads(header_line)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Checkpoint header is not valid JSON. First line: {header_line!r}"
        ) from e

    # Load the checkpoint model + state. We intentionally skip deserializing the old
    # optimizer state; we're about to reinitialize it anyway.
    model, state, _old_opt_state, step = load(input_path)

    optimizer = get_optimizer(
        scheduler_type=args.scheduler_type, clip_grad_norm=args.clip_grad_norm
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Ensure output directory exists.
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    save(output_path, dict(hyperparams), step, model, state, opt_state)
    print(
        f"Reinitialized optimizer and saved checkpoint:\n"
        f"  input:  {input_path}\n"
        f"  output: {output_path}\n"
        f"  step: {step}\n"
        f"  scheduler_type: {args.scheduler_type}\n"
        f"  clip_grad_norm: {args.clip_grad_norm}"
    )


def _fmt_bytes(num_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(num_bytes)
    for u in units:
        if x < 1024.0 or u == units[-1]:
            return f"{x:.2f} {u}" if u != "B" else f"{int(x)} {u}"
        x /= 1024.0
    return f"{num_bytes} B"


def _summarize_opt_state(opt_state) -> dict:
    """Return a small, printable summary of an optax state pytree."""
    leaves = jax.tree_util.tree_leaves(opt_state)
    summary: dict[str, object] = {
        "type": type(opt_state).__name__,
        "num_leaves": len(leaves),
        "num_array_leaves": 0,
        "num_non_array_leaves": 0,
        "scalar_int_leaves": [],
        "array_dtypes": {},
    }

    for leaf in leaves:
        is_array = hasattr(leaf, "shape") and hasattr(leaf, "dtype")
        if not is_array:
            summary["num_non_array_leaves"] = int(summary["num_non_array_leaves"]) + 1
            continue

        summary["num_array_leaves"] = int(summary["num_array_leaves"]) + 1
        dtype = str(getattr(leaf, "dtype", "unknown"))
        array_dtypes = summary["array_dtypes"]
        assert isinstance(array_dtypes, dict)
        array_dtypes[dtype] = int(array_dtypes.get(dtype, 0)) + 1

        # Try to find a scalar int "count" leaf (common in some optax transforms).
        try:
            shape = tuple(leaf.shape)
        except Exception:
            shape = None
        if shape == ():
            try:
                kind = leaf.dtype.kind  # type: ignore[attr-defined]
            except Exception:
                kind = None
            if kind in ("i", "u"):
                try:
                    val = int(jax.device_get(leaf))
                    scalar_list = summary["scalar_int_leaves"]
                    assert isinstance(scalar_list, list)
                    scalar_list.append(val)
                except Exception:
                    pass

    return summary


def run_meta(args, parser):
    checkpoint_path = args.checkpoint_path
    if not os.path.exists(checkpoint_path):
        parser.error(f"Checkpoint not found: {checkpoint_path}")

    st = os.stat(checkpoint_path)
    print(f"checkpoint: {checkpoint_path}")
    print(f"size: {_fmt_bytes(st.st_size)}")
    print(f"mtime: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.st_mtime))}")

    # Read the JSON header line for fast metadata.
    with open(checkpoint_path, "rb") as f:
        header_line = f.readline().decode()
    try:
        hyperparams = json.loads(header_line)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Checkpoint header is not valid JSON. First line: {header_line!r}"
        ) from e

    has_opt_state = bool(hyperparams.get("has_opt_state", False))
    step = int(hyperparams.get("step", 0))
    print(f"step: {step}")
    print(f"learning_rate: {get_learning_rate(step):.10g}")
    print(f"has_opt_state: {has_opt_state}")

    # Print remaining hyperparams (excluding our metadata keys).
    extra = {k: v for k, v in hyperparams.items() if k not in ("step", "has_opt_state")}
    if extra:
        print("hyperparams:")
        for k in sorted(extra.keys()):
            print(f"  {k}: {extra[k]}")

    if not has_opt_state:
        return

    if args.skip_opt_state:
        print(
            "opt_state: present (skipped deserialization; re-run without --skip-opt-state to summarize)"
        )
        return

    optimizer = get_optimizer("cosine")
    _model, _state, opt_state, loaded_step = load(checkpoint_path, optimizer=optimizer)
    # `loaded_step` is the authoritative training step stored in the header.
    if loaded_step != step:
        print(f"note: header step={step} but deserialized step={loaded_step}")

    if opt_state is None:
        print(
            "opt_state: expected present, but could not be deserialized (missing optimizer structure?)"
        )
        return

    summary = _summarize_opt_state(opt_state)
    print("opt_state:")
    print(f"  type: {summary['type']}")
    print(
        f"  leaves: {summary['num_leaves']} (arrays={summary['num_array_leaves']}, non-arrays={summary['num_non_array_leaves']})"
    )
    print(f"  dtypes: {summary['array_dtypes']}")
    scalar_ints = summary.get("scalar_int_leaves", [])
    if isinstance(scalar_ints, list) and scalar_ints:
        uniq = sorted(set(int(x) for x in scalar_ints))
        # Keep it compact; these are usually just small counters.
        print(
            f"  scalar_int_leaves (possible counters): {uniq[:8]}{'...' if len(uniq) > 8 else ''}"
        )


def run_loop(args):
    """
    Run the combined self-play and training loop with a persistent model +
    optimizer state and a sliding window training buffer.
    """
    key = jax.random.PRNGKey(args.seed)
    num_simulations = args.simulations
    batch_size = args.batch

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    optimizer = get_optimizer()

    checkpoint_path = find_latest_checkpoint(args.checkpoint_dir)
    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path}")
        model, state, opt_state, steps = load(checkpoint_path, optimizer=optimizer)
    else:
        print("No checkpoint found; initializing new model.")
        key, model_key = jax.random.split(key)
        model, state = eqx.nn.make_with_state(ConnectZeroModel)(model_key)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
        steps = 0

    if opt_state is None:
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    loop_state_path = os.path.join(args.checkpoint_dir, "loop_state.json")
    loop_state = {}

    if os.path.exists(loop_state_path):
        try:
            with open(loop_state_path, "r") as f:
                loop_state = json.load(f)
            print(f"Found loop state: {loop_state}")
        except json.JSONDecodeError:
            print("Failed to load loop state file, starting fresh.")

    checkpoint_interval = loop_state.get("checkpoint_interval", 0)

    while True:
        # Determine if we are resuming a previous interval
        stored_games_played = loop_state.get("games_played", 0)
        stored_subdir = loop_state.get("data_subdir", None)

        is_resuming = False
        if stored_subdir and os.path.isdir(stored_subdir):
            if stored_games_played < args.games_per_checkpoint:
                is_resuming = True

        if is_resuming:
            data_subdir = stored_subdir
            games_played = stored_games_played
            print(
                f"\nResuming interval {checkpoint_interval} with {games_played} games played."
            )
            print(f"Saving trajectories to: {data_subdir}")
        else:
            checkpoint_interval += 1
            print(f"\n{'=' * 60}")
            print(f"Starting checkpoint interval {checkpoint_interval}")
            print(f"{'=' * 60}")

            data_subdir = get_next_data_subdir(args.data_dir)
            print(f"Saving trajectories to: {data_subdir}")
            games_played = 0

            # Initialize loop state
            loop_state = {
                "checkpoint_interval": checkpoint_interval,
                "data_subdir": data_subdir,
                "games_played": games_played,
            }
            with open(loop_state_path, "w") as f:
                json.dump(loop_state, f)

        # Track files written during this interval
        trajectory_files = []
        batches_done = 0
        mcts_steps = 0
        selfplay_start = time.perf_counter()
        batches_target = math.ceil(args.games_per_checkpoint / batch_size)

        def format_rate(rate: float, rate_label: str, inv_label: str) -> str:
            if rate <= 0:
                return f"{inv_label}=?"
            return (
                f"{rate_label}={rate:.2f}"
                if rate >= 1.0
                else f"{inv_label}={1.0 / rate:.2f}"
            )

        # Self-play loop until we reach games_per_checkpoint
        pbar = tqdm(
            total=batches_target,
            desc="Self-play",
            unit="batch",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
            initial=math.ceil(games_played / batch_size),
        )

        inference_model = eqx.tree_inference(model, value=True)
        model_tuple = (inference_model, state)

        while games_played < args.games_per_checkpoint:
            # Preserve previous behavior of consuming one split per batch
            key, _ = jax.random.split(key)

            # Initialize game state for this batch
            board_state = jnp.zeros((batch_size, 6, 7), dtype=jnp.int32)
            tree = batched.BatchedSearchTree.init(
                B=batch_size, N=num_simulations * 42 + 1, A=7
            )
            max_moves = 42  # 6x7 board

            # Play until all games in the batch are done (JAX while_loop)
            key, tree, board_state, turn_count, game_history, steps_taken = (
                run_batched_selfplay_while_loop(
                    key,
                    tree,
                    board_state,
                    num_simulations,
                    model_tuple,
                    args.temperature,
                    args.temperature_depth,
                    0.3,  # dirichlet_alpha
                    0.25,  # dirichlet_epsilon
                    max_moves,
                )
            )

            steps_taken_host = int(jax.device_get(steps_taken))
            mcts_steps += steps_taken_host

            elapsed = max(time.perf_counter() - selfplay_start, 1e-8)
            steps_per_sec = mcts_steps / elapsed
            batches_per_sec = batches_done / elapsed
            steps_display = format_rate(steps_per_sec, "steps/s", "sec/step")
            batches_display = format_rate(batches_per_sec, "batches/s", "sec/batch")
            pbar.set_postfix(
                batch=len(trajectory_files) + 1,
                moves=f"{steps_taken_host}/{max_moves}",
                steps=steps_display,
                batches=batches_display,
            )

            # Process game history to set value targets
            processed_history = process_batched_game_history(
                game_history, board_state, turn_count
            )

            # Save trajectories to disk
            filename = get_unique_filename(data_subdir)
            save_trajectories(processed_history, filename)
            trajectory_files.append(filename)

            remaining_games = args.games_per_checkpoint - games_played
            batch_increment = min(batch_size, max(remaining_games, 0))
            games_played += batch_increment
            batches_done += 1
            pbar.update(1)

            # Update loop state
            loop_state["games_played"] = games_played
            with open(loop_state_path, "w") as f:
                json.dump(loop_state, f)

        pbar.close()

        # Training phase with sliding window selection
        training_files = select_training_files(
            args.data_dir, args.training_buffer_length
        )
        if not training_files:
            print("No training data available; skipping training step.")
            continue

        print(
            f"\n--- Training on {len(training_files)} files "
            f"(target rows >= {args.training_buffer_length}) ---"
        )

        train_model = eqx.tree_inference(model, value=False)
        train_model, state, opt_state, loss, steps = train_epoch(
            model=train_model,
            state=state,
            opt_state=opt_state,
            data_files=training_files,
            batch_size=args.batch_size,
            optimizer=optimizer,
            initial_step_count=steps,
        )

        model = train_model

        save_path = os.path.join(args.checkpoint_dir, f"checkpoint_{steps}_steps.eqx")
        hyperparams = {"num_blocks": len(model.blocks)}
        save(save_path, hyperparams, steps, model, state, opt_state)
        checkpoint_path = save_path

        # Clear loop state as we finished this interval
        if os.path.exists(loop_state_path):
            os.remove(loop_state_path)
            loop_state = {}

        print(
            f"Checkpoint interval {checkpoint_interval} complete. "
            f"avg_loss={loss:.4f}, saved {save_path}"
        )


def main():
    parser = argparse.ArgumentParser(description="MCTS Connect Four")

    # Global arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=int(time.time()),
        help="Random seed for PRNG key (default: current timestamp)",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default=None,
        help="Directory path to save trajectories. If not set, data is not saved.",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        default=1,
        help="Batch size for simulation (default: 1)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    # Initialize subcommand
    initialize_parser = subparsers.add_parser(
        "initialize", help="Initialize a new model"
    )
    initialize_parser.add_argument(
        "path",
        type=str,
        help="Path to save the initialized model",
    )

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to existing checkpoint to load (or path to initialized model)",
    )
    train_parser.add_argument(
        "data_pattern",
        type=str,
        help="Glob pattern for training data (parquet files)",
    )
    train_parser.add_argument(
        "--save-dir",
        type=str,
        default="./checkpoints",
        help="Directory to save new checkpoints (default: ./checkpoints)",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size (default: 8)",
    )

    # Meta subcommand
    meta_parser = subparsers.add_parser(
        "meta", help="Print metadata about a checkpoint file"
    )
    meta_parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to a checkpoint (.eqx) file",
    )
    meta_parser.add_argument(
        "--skip-opt-state",
        action="store_true",
        help="Only read the JSON header; do not deserialize optimizer state",
    )

    # Reinitialize optimizer subcommand
    init_opt_parser = subparsers.add_parser(
        "init-optimizer",
        help="Load a checkpoint, reinitialize the optimizer state, and write a new checkpoint",
    )
    init_opt_parser.add_argument(
        "input_checkpoint",
        type=str,
        help="Input checkpoint path or name under ./checkpoints",
    )
    init_opt_parser.add_argument(
        "output_checkpoint",
        type=str,
        help="Output checkpoint path or name under ./checkpoints",
    )
    init_opt_parser.add_argument(
        "--scheduler-type",
        type=str,
        default="cosine",
        choices=["linear", "cosine"],
        help='Optimizer LR scheduler type (default: "cosine")',
    )
    init_opt_parser.add_argument(
        "--clip-grad-norm",
        type=float,
        default=1.0,
        help="Global gradient norm clipping (default: 1.0)",
    )

    # Simulate subcommand
    simulate_parser = subparsers.add_parser("simulate", help="Run MCTS simulation")
    simulate_parser.add_argument(
        "--interactive",
        action="store_true",
        help="Play interactively against the computer (Computer goes first)",
    )
    simulate_parser.add_argument(
        "--single",
        action="store_true",
        help="Use single-game optimized mode (ignores --batch)",
    )
    simulate_parser.add_argument(
        "--simulations",
        type=int,
        default=10000,
        help="Number of MCTS simulations per move (default: 10000)",
    )
    simulate_parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint file to load (requires --puct)",
    )
    simulate_parser.add_argument(
        "-p",
        "--puct",
        action="store_true",
        help="Use PUCT with a randomly initialized neural network",
    )

    simulate_parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for early moves (default: 1.0)",
    )
    simulate_parser.add_argument(
        "--temperature-depth",
        type=int,
        default=15,
        help="Number of moves to apply temperature sampling before switching to greedy (default: 15)",
    )
    simulate_parser.add_argument(
        "-n",
        "--iterations",
        type=int,
        default=1,
        help="Number of times to run the full self-play loop (default: 1)",
    )

    # Loop subcommand - combined self-play and training
    loop_parser = subparsers.add_parser(
        "loop", help="Run combined self-play and training loop"
    )
    loop_parser.add_argument(
        "-g",
        "--games-per-checkpoint",
        type=int,
        default=100,
        help="Number of games to play before training (default: 100)",
    )
    loop_parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Directory for loading/saving checkpoints (default: ./checkpoints)",
    )
    loop_parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory for trajectory data (default: ./data)",
    )
    loop_parser.add_argument(
        "--simulations",
        type=int,
        default=800,
        help="Number of MCTS simulations per move (default: 800)",
    )
    loop_parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for early moves (default: 1.0)",
    )
    loop_parser.add_argument(
        "--temperature-depth",
        type=int,
        default=15,
        help="Number of moves to apply temperature sampling before switching to greedy (default: 15)",
    )
    loop_parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size (default: 8)",
    )
    loop_parser.add_argument(
        "--training-buffer-length",
        type=int,
        default=5000,
        help="Target number of samples to keep in the sliding training buffer (default: 5000)",
    )

    args = parser.parse_args()

    if args.command == "simulate":
        run_simulate(args, parser)
    elif args.command == "initialize":
        run_initialize(args)
    elif args.command == "train":
        run_train(args)
    elif args.command == "meta":
        run_meta(args, parser)
    elif args.command == "init-optimizer":
        run_init_optimizer(args, parser)
    elif args.command == "loop":
        run_loop(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
