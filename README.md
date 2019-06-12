# AlphaZero_Connect4
implement AlphaZero on Connect4 using PyTorch and standard python libraries

# # How to play

* Run the MCTS_c4.py to generate self-play datasets. Note that for the first time, you will need to create and save a random, initialized alpha_net for loading.

* Run train_c4.py to train the alpha_net with the datasets.

* At predetermined checkpoints, run evaluator_c4.py to evaluate the trained net against the neural net from previous iteration. Saves the neural net that performs better.

* Repeat for next iteration.