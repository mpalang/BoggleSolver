# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 21:39:17 2026

@author: moritzpalang
"""
from pathlib import Path
import pickle
import warnings

from utils.paths import DICTIONARIES


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False
        
  
class Trie:
    def __init__(self):
        """This class constructs a dictionary trie."""
        self.root = TrieNode()


    def insert(self, word):
        node = self.root
        for n,letter in enumerate(word):
            node = node.children.setdefault(letter, TrieNode())
        node.is_word = True
        
        
    def contains(self, word):
        node = self.root
        word = word.upper()
        for letter in word:
            if letter not in node.children:
                return False
            node = node.children[letter]
        return node.is_word

    
    def is_prefix(self, prefix):
        node = self.root
        prefix = prefix.upper()
        for letter in prefix:
            if letter not in node.children:
                return False
            node = node.children[letter]
        return True

    
    def words_with_prefix(self,prefix):
        node = self.root
        prefix = prefix.upper()
        for letter in prefix:
            if letter not in node.children:
                return []
            node = node.children[letter]
        results = []
    
        def dfs(n, path):
            if n.is_word:
                results.append(path)
            for ch, child in n.children.items():
                dfs(child, path + ch)
        dfs(node, prefix)
        return results



def build_trie(dictionary: Path | str = 'small'):
    """Builds a new trie. Requires a valid dictionary.txt file with words separated by linebreaks."""

    if dictionary == 'large' or dictionary == '400k':
        filepath = DICTIONARIES / "words_400k.txt"
    elif dictionary == 'small' or dictionary == '180k':
        filepath = DICTIONARIES / "words_180k.txt"
    elif isinstance(dictionary,(str, Path)):
        filepath = dictionary

    if not filepath.exists():
        raise FileNotFoundError(f'Dictionary file:{filepath} not found.')
    with open(filepath) as words_file:
        valid_words = set(words_file.read().upper().split())
            
    #bulding trie
    trie = Trie()
    for word in valid_words:
        trie.insert(word)
    
    #saving trie
    filename = f"trie_{Path(dictionary).stem}.pkl"
    with open(DICTIONARIES / filename, "wb") as f:
        pickle.dump(trie, f)

    return trie



def load_trie(dictionary: str = "small"):
    path = DICTIONARIES / f"trie_{dictionary}.pkl"
    if not path.exists():
        warnings.warn('No trie file found. Creating one.')
        trie = build_trie(dictionary) # saves trie to 'trie.pkl'
    else:
        with open(path,"rb") as f:
            trie = pickle.load(f)
    return trie
