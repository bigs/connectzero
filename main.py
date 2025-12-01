import argparse
import time

import jax
import jax.numpy as jnp
from jax import Array

from connectzero import batched, single
from connectzero.game import (
    check_winner,
    check_winner_single,
    play_move,
    play_move_single,
    print_board_state,
    print_board_states,
)

run_search_vmap = jax.vmap(single.run_mcts_search, in_axes=(0, 0, None, 0))
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


def main():
    parser = argparse.ArgumentParser(description="MCTS Connect Four")
    parser.add_argument(
        "--seed",
        type=int,
        default=int(time.time()),
        help="Random seed for PRNG key (default: current timestamp)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Play interactively against the computer (Computer goes first)",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        default=1,
        help="Batch size for simulation (default: 1)",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Use single-game optimized mode (ignores --batch)",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=10000,
        help="Number of MCTS simulations per move (default: 10000)",
    )
    args = parser.parse_args()

    key = jax.random.PRNGKey(args.seed)
    num_simulations = args.simulations

    if args.single:
        if args.batch > 1:
            print("Running new engine in batch mode")
            board_state = jnp.zeros((args.batch, 6, 7), dtype=jnp.int32)
            print("Initial Board State:")
            print_board_states(board_state)

            turn_count = jnp.count_nonzero(board_state, axis=(1, 2))
            tree_prototype = single.SearchTree.init(N=num_simulations * 42 + 1, A=7)
            tree = jax.tree.map(lambda x: jnp.stack([x] * args.batch), tree_prototype)

            while jnp.any(check_winner(board_state, turn_count) == 0):
                key, subkey = jax.random.split(key)
                batch_keys = jax.random.split(subkey, args.batch)
                # is_player_turn = is_interactive & (jnp.any(turn_count % 2 != 0))

                tree, best_action, board_state = run_search_vmap(
                    tree, board_state, num_simulations, batch_keys
                )
                turn_count = jnp.count_nonzero(board_state, axis=(1, 2))
                print_board_states(board_state)

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
                    tree, best_action, board_state = single.run_mcts_search(
                        tree, board_state, num_simulations, subkey
                    )
                    turn_count = jnp.count_nonzero(board_state)
                    print_board_state(board_state)
                    print("Best Action:", best_action)

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
                tree, board_state = handle_player_turn(tree, board_state, turn_count)
                turn_count = jnp.count_nonzero(board_state, axis=(1, 2))
            else:
                tree, best_action, board_state = batched.run_mcts_search(
                    tree, board_state, num_simulations, subkey
                )
                turn_count = jnp.count_nonzero(board_state, axis=(1, 2))
                print_board_states(board_state)
                print("Best Action:", best_action)


if __name__ == "__main__":
    main()
