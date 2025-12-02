# ConnectZero

The goal of this project is to recreate the results of AlphaZero and MuZero for the game of Connect Four, building up the code step by step.

## Phases

1. **Pure MCTS** (Completed): The first phase is to implement a pure MCTS agent. The MCTS algorithm is complex enough, that I wanted to address it in isolation before adding the network element. Since the search space is relatively small compared to other games, the level of play of the pure MCTS agent is actually decent.

2. **AlphaZero Style Policy/Value network** (In progress): The second phase is to implement a policy and value network that can be used to guide the MCTS search. Trained via self-play.
  - [x] Training training data export helper functions
  - [x] Implement policy/value (PV) network in Equinox
  - [ ] Implement training data serialization with Parquet or Huggingface Datasets
  - [ ] Implement PUCT with the PV network
  - [ ] Implement full training loop

3. **MuZero Style Policy/Value network** (Not started): The third phase is to implement a MuZero-style policy and value network that can be used to guide the MCTS search. Relies on latent board representation and is also trained via self-play.

## Tools / Implementation Details

I used jax to implement the MCTS algorithm, since it's highly parallelizable. I may yet use jax directly for the autograd of the policy and value networks, or I may use pytorch.

Everything has been implemented with batch-support from the start, since that will be important for the policy and value networks and scaling up self-play. I will not support distributed training. The reason I chose Connect Four was that it seemed quite tractable on consumer hardware.

## LLMs

Since this is a learning exercise, I've avoided using any agentic code generation for the core logic. Elements I've considered auxilliary (such as the ability to play against the engine as a human vs exclusiely self-play) are implemented using LLMs. Likewise, any visualizations and perhaps training harnesses will probably be made with LLM assistance.

I have, however, used LLMs extensively for learning the details of the algorithms and for periodic code review, especially as this is the first time I've used jax so extensively. Dang, they're useful.
