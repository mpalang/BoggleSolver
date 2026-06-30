# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 20:48:32 2026

@author: morit
"""
import numpy as np
from pathlib import Path
import sys
import pickle
from utils import Trie,TrieNode,buildTrie,BoggleSolver

if str(Path(__file__).parent) not in sys.path:
    sys.path.append(str(Path(__file__).parent))
    
from utils import buildTrie

trie_file = Path(__file__).parent/'trie.pkl'
if not trie_file.exists():
    buildTrie()    
with open(trie_file,'rb') as f:
    print('load trie')
    trie = pickle.load(f)    


Board = [['L','A','K','L'],
         ['G','O','I','W'],
         ['T','N','H','D'],
         ['U','A','T','Qu']
         ]

bs = BoggleSolver(Board)
valid_words = [word for word in bs.solve(trie) if len(word)>3]
    