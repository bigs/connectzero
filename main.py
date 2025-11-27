import argparse
import time
import jax
import jax.numpy as jnp

from game import play_move, check_winner, print_board_states
from tree import ArenaTree, advance_tree
from mcts import run_mcts_search


def handle_player_turn(
    tree: ArenaTree, board_state: jnp.ndarray, turn_count: jnp.ndarray
) -> tuple[ArenaTree, jnp.ndarray]:
    col = -1
    while col < 0 or col > 6:
        try:
            line = input(f"Your move (0-6) [Turn {int(turn_count[0])}]: ")
            col = int(line)
            if col < 0 or col > 6:
                print("Invalid column. Please enter 0-6.")
                continue
            # Check if column is full
            if board_state[0, 0, col] != 0:
                print("Column is full.")
                col = -1
        except ValueError:
            print("Invalid input.")

    action = jnp.array([col], dtype=jnp.int32)

    # Advance tree first (using current root)
    tree = advance_tree(tree, action)

    # Play move
    player_who_plays = (turn_count % 2) + 1
    board_state = play_move(board_state, action, player_who_plays)

    print_board_states(board_state)
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
    args = parser.parse_args()

    # Initialize a board with two moves in the middle column
    starting_board_state = jnp.zeros((6, 7), dtype=jnp.int32)
    starting_board_state = jnp.expand_dims(starting_board_state, axis=0)

    print("Initial Board State:")
    print_board_states(starting_board_state)

    key = jax.random.PRNGKey(args.seed)
    batch_size = starting_board_state.shape[0]
    num_simulations = 10000
    board_state = starting_board_state

    turn_count = jnp.count_nonzero(board_state, axis=(1, 2))
    # Node count is an upper bound based on simulations and turns per game
    tree = ArenaTree.init(B=batch_size, N=num_simulations * 42 + 1, A=7)

    while jnp.any(check_winner(board_state, turn_count) == 0):
        # Split key to ensure different random seeds for each search
        key, subkey = jax.random.split(key)

        # Check for player turn (Player 2)
        is_player_turn = args.interactive and (int(turn_count[0]) % 2 != 0)

        if is_player_turn:
            tree, board_state = handle_player_turn(tree, board_state, turn_count)
            turn_count = jnp.count_nonzero(board_state, axis=(1, 2))
        else:
            tree, best_action, board_state = run_mcts_search(
                tree, board_state, num_simulations, subkey
            )
            turn_count = jnp.count_nonzero(board_state, axis=(1, 2))
            print_board_states(board_state)
            print("Best Action:", best_action)


if __name__ == "__main__":
    main()
