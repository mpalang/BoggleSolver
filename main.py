# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 20:48:32 2026

@author: morit
"""
import numpy as np
from pathlib import Path
import sys
import pickle
from Dictionary.utils import Trie,TrieNode,buildTrie,BoggleSolver
# from BoardRecognizer.Read import read_board,get_board
# from BoardRecognizer.auxiliary import load_image,load_templates,show


trie_file = Path(__file__).parent/'Dictionary'/'trie.pkl'
if not trie_file.exists():
    buildTrie()    
with open(trie_file,'rb') as f:
    print('load trie')
    trie = pickle.load(f)    

# paths = list(Path('board_recognizing/TrainingImages').glob('*.jpg'))
# images = [load_image(path) for path in paths]
# templates = load_templates()

#%%
board = np.array([['T','T','F','R'],
                  ['E','A','A','H'],
                  ['H','Qu','R','D'],
                  ['O','L','E','A']
                    ])

# board = get_board(images[0]) 
# board = read_board(board,templates)
bs = BoggleSolver(board)
valid_words = bs.solve(trie)
valid_words = [word for word in bs.solve(trie) if len(word)>=5]
print(valid_words)
    