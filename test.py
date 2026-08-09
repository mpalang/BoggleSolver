# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 18:46:15 2026

@author: morit

"""
from pathlib import Path
import numpy as np
import sys
if str(Path(__file__).parent) not in sys.path:
    sys.path.append(str(Path(__file__).parent))
    
from board_recognizing.auxiliary import load_image,show,timer,load_templates,make_bw,canon,rotate_image
from board_recognizing.boggle_reader import BoggleReader

import pickle

with open(Path('Dictionary/trie.pkl'),'rb') as f:
    trie = pickle.load(f)

#%%

paths = list(Path('board_recognizing/TrainingImages').glob('*.jpg'))
images = [load_image(path) for path in paths]

image = images[0]

#%%
boggles = []
for image in images:
    try:
        boggles.append(BoggleReader(image))
    except Exception as e:
        print(e)
    

#%%

b = boggles[5]
show(b.board)
print(b.letters)
print(b.valid_words)



