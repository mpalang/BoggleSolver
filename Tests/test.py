# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 18:46:15 2026

@author: morit

"""

from pathlib import Path
# import numpy as np
# import sys
import os
os.chdir(r'C:\Users\morit\OneDrive\Anwendungen\BoggleSolver')
    
from boggle_solver.utils import load_image,show,timer,make_bw,canon,rotate_image
from boggle_solver.BoggleSolver import BoggleSolver

paths = list(Path('boggle_solver/TrainingImages').glob('*.jpg'))
image = load_image(paths[11])

#%%

boggle = BoggleSolver(image)

show(boggle.board)
print(boggle.letters)
print(boggle.valid_words)



