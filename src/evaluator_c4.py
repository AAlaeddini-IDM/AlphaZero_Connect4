#!/usr/bin/env python

import os.path
import torch
import numpy as np
from alpha_net_c4 import ConnectNet
from connect_board import board as cboard
import encoder_decoder_c4 as ed
import copy
from MCTS_c4 import UCT_search, do_decode_n_move_pieces, get_policy
import pickle
import torch.multiprocessing as mp
import datetime

def save_as_pickle(filename, data):
    completeName = os.path.join("./evaluator_data/",\
                                filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)

class arena():
    def __init__(self,current_cnet,best_cnet):
        self.current = current_cnet
        self.best = best_cnet
    
    def play_round(self):
        if np.random.uniform(0,1) <= 0.5:
            white = self.current; black = self.best; w = "current"; b = "best"
        else:
            white = self.best; black = self.current; w = "best"; b = "current"
        current_board = cboard()
        checkmate = False
        dataset = []
        value = 0; t = 0.1
        while checkmate == False and current_board.actions() != []:
            dataset.append(copy.deepcopy(ed.encode_board(current_board)))
            print(current_board.current_board); print(" ")
            if current_board.player == 0:
                root = UCT_search(current_board,777,white,t)
                policy = get_policy(root, t); print(policy, "white = %s" %(str(w)))
            elif current_board.player == 1:
                root = UCT_search(current_board,777,black,t)
                policy = get_policy(root, t); print(policy, "black = %s" %(str(b)))
            current_board = do_decode_n_move_pieces(current_board,\
                                                    np.random.choice(np.array([0,1,2,3,4,5,6]), \
                                                                     p = policy)) # decode move and move piece(s)
            if current_board.check_winner() == True: # someone wins
                if current_board.player == 0: # black wins
                    value = -1
                elif current_board.player == 1: # white wins
                    value = 1
                checkmate = True
        dataset.append(ed.encode_board(current_board))
        if value == -1:
            dataset.append(f"{b} as black wins")
            return b, dataset
        elif value == 1:
            dataset.append(f"{w} as white wins")
            return w, dataset
        else:
            dataset.append("Nobody wins")
            return None, dataset
    
    def evaluate(self, num_games,cpu):
        current_wins = 0
        for i in range(num_games):
            winner, dataset = self.play_round(); print("%s wins!" % winner)
            if winner == "current":
                current_wins += 1
            save_as_pickle("evaluate_net_dataset_cpu%i_%i_%s_%s" % (cpu,i,datetime.datetime.today().strftime("%Y-%m-%d"),\
                                                                     str(winner)),dataset)
        print("Current_net wins ratio: %.5f" % current_wins/num_games)
        #if current_wins/num_games > 0.55: # saves current net as best net if it wins > 55 % games
        #    torch.save({'state_dict': self.current.state_dict()}, os.path.join("./model_data/",\
        #                                "best_net.pth.tar"))

def fork_process(arena_obj,num_games,cpu): # make arena picklable
    arena_obj.evaluate(num_games,cpu)

if __name__=="__main__":
    multiprocessing = 0
    current_net="c4_current_net_trained2_iter0.pth.tar"; best_net="current_net_trained1_iter2.pth.tar"
    current_net_filename = os.path.join("./model_data/",\
                                    current_net)
    best_net_filename = os.path.join("./model_data/",\
                                    best_net)
    current_cnet = ConnectNet()
    best_cnet = ConnectNet()
    cuda = torch.cuda.is_available()
    if cuda:
        current_cnet.cuda()
        best_cnet.cuda()
    
    if multiprocessing == 1:
        mp.set_start_method("spawn",force=True)
        
        current_cnet.share_memory(); best_cnet.share_memory()
        current_cnet.eval(); best_cnet.eval()
        
        checkpoint = torch.load(current_net_filename)
        current_cnet.load_state_dict(checkpoint['state_dict'])
        checkpoint = torch.load(best_net_filename)
        best_cnet.load_state_dict(checkpoint['state_dict'])
         
        processes = []
        for i in range(6):
            p = mp.Process(target=fork_process,args=(arena(current_cnet,best_cnet),10,i))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
            
    elif multiprocessing == 0:
        current_cnet.eval(); best_cnet.eval()
        checkpoint = torch.load(current_net_filename)
        current_cnet.load_state_dict(checkpoint['state_dict'])
        checkpoint = torch.load(best_net_filename)
        best_cnet.load_state_dict(checkpoint['state_dict'])
        arena1 = arena(current_cnet=current_cnet,best_cnet=best_cnet)
        arena1.evaluate(num_games=100, cpu=0)
        
        