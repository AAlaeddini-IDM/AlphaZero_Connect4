# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 09:55:20 2019

@author: WT
"""

import os
import numpy as np
import pickle
import encoder_decoder_c4 as ed
from visualize_board_c4 import view_board as vb
import matplotlib.pyplot as plt

data_path = "./evaluator_data/evaluator_trained1_iter6_vs_trained1_iter0/"
file = "evaluate_net_dataset_cpu0_15_2019-04-13_best"
filename = os.path.join(data_path,file)
with open(filename, 'rb') as fo:
    dataset = pickle.load(fo, encoding='bytes')

#last_move = np.argmax(dataset[-2][1])
#b = ed.decode_board(dataset[-2][0])

#b.drop_piece(last_move)
    
for i in range(len(dataset)-1):
    #board = ed.decode_board(dataset[i][0])
    board = ed.decode_board(dataset[i])
    fig = vb(board.current_board)
    #plt.savefig(os.path.join("C:/Users/WT/Desktop/Python_Projects/chess/chess_ai_py35updated/gamesimages/iter1/ex4/", \
    #                        f"{file}_{i}.png"))
    #plt.savefig(os.path.join("C:/Users/WT/Desktop/Python_Projects/chess/chess_ai_py35updated/src/evaluator_data/iter0/best_wins/ex1/", \
     #                       f"{file}_{i}.png"))
    plt.savefig(os.path.join("C:/Users/WT/Desktop/Python_Projects/Connect4/src/evaluator_data/evaluator_trained1_iter6_vs_trained1_iter0/1",\
                             f"{file}_{i}.png"))