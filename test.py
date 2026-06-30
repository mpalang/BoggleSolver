# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 23:46:58 2026

@author: morit
"""
from pathlib import Path

with open(Path(__file__).parent/'words_alpha.txt') as words_file:
    valid_words = set(words_file.read().upper().split())