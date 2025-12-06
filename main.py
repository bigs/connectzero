import argparse
import os
import time
from collections import deque

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from connectzero import batched, single
from connectzero.game import (
    TrainingSample,
    check_winner,
    check_winner_single,
    play_move,
    play_move_single,
    print_board_state,
    print_board_states,
    save_trajectories,
)
from connectzero.model import ConnectZeroModel, load, save
from connectzero.train import train_loop


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
    history: list[TrainingSample],
    final_board_state: jnp.ndarray,
    final_turn_count: jnp.ndarray,
) -> list[TrainingSample]:
    """
    Process batched game history to set value targets based on game outcomes.

    Uses negamax-style backwards propagation: the value target for each
    sample is set based on whether the player whose turn it was at that
    state eventually won, lost, or drew. Processes each game in the batch
    independently.

    Args:
        history: List of TrainingSample with zeroed value_target.
                Each sample has batched dimensions: board_state [B, 3, 6, 7],
                policy_target [B, 7], value_target [B].
        final_board_state: [B, 6, 7] array, the final board states.
        final_turn_count: [B] array, the turn counts at game end.

    Returns:
        List of TrainingSample with value_target set.
    """
    batch_size = final_board_state.shape[0]
    winners = check_winner(final_board_state, final_turn_count)
    winners = jax.device_get(winners)  # Convert to numpy for indexing

    # Process each game in the batch
    processed_samples = []
    for i, sample in enumerate(history):
        # Extract batched components
        batched_board_state = jax.device_get(sample.board_state)  # [B, 3, 6, 7]
        batched_policy_target = jax.device_get(sample.policy_target)  # [B, 7]
        batched_value_target = []

        # Process each game in the batch
        # Sample i corresponds to turn_count = i (before move i+1)
        # For games that ended early, we still assign value targets based on final outcome
        for b in range(batch_size):
            # The sample at index i corresponds to turn_count = i
            # But if the game ended earlier, we use the final turn_count
            # to determine if this sample is valid (though we still assign value)
            turn_count_at_step = i
            current_player = (turn_count_at_step % 2) + 1
            winner = int(winners[b])

            if winner == 3:  # Draw
                value = 0.0
            elif winner == current_player:  # Current player won
                value = 1.0
            else:  # Current player lost
                value = -1.0

            batched_value_target.append(value)

        # Reconstruct the batched sample
        updated_sample = TrainingSample(
            board_state=jnp.array(batched_board_state, dtype=jnp.float32),
            policy_target=jnp.array(batched_policy_target, dtype=jnp.float32),
            value_target=jnp.array(batched_value_target, dtype=jnp.float32),
        )
        processed_samples.append(updated_sample)

    return processed_samples


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


def run_simulate(args, parser):
    key = jax.random.PRNGKey(args.seed)
    num_simulations = args.simulations

    model_tuple = None
    if args.puct:
        if args.checkpoint:
            print(f"Loading checkpoint from {args.checkpoint}")
            model, model_state, _ = load(args.checkpoint)
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
                        1.0,  # dirichlet_alpha
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
                # Add to replay buffer
                for s in processed_history:
                    replay_buffer.append(s)

                # Save to disk
                if args.out:
                    filename = get_unique_filename(args.out)
                    save_trajectories(list(replay_buffer), filename)

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
                            1.0,  # dirichlet_alpha
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
            for s in processed_history:
                replay_buffer.append(s)

            if args.out:
                filename = get_unique_filename(args.out)
                save_trajectories(list(replay_buffer), filename)


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
    save(args.path, hyperparams, model, state)
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

    args = parser.parse_args()

    if args.command == "simulate":
        run_simulate(args, parser)
    elif args.command == "initialize":
        run_initialize(args)
    elif args.command == "train":
        run_train(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
