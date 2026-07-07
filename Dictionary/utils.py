# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 21:39:17 2026

@author: moritzpalang
"""
from pathlib import Path
import pickle

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
        for letter in word:
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
        
def buildTrie():
    print('loading words')
    with open(Path(__file__).parent/'words_180k.txt') as words_file:
        valid_words = set(words_file.read().upper().split())
    
    print('bulding trie')
    trie = Trie()
    for word in valid_words:
        trie.insert(word)
    
    print('saving trie')
    with open(Path(__file__).parent/"trie.pkl", "wb") as f:
        pickle.dump(trie, f)
        
    print('done')


class BoggleSolver:
    def __init__(self,board):
        self.board = board
        
    def solve(self,trie):
        results = set()

        ROWS,COLS = self.board.shape
        
        for r in range(ROWS):
            for c in range(COLS):
                self.dfs(r, c, trie.root, "", set(), results)
        
        results = sorted(results, key=lambda x: (x,len(x)))
        self.valid_words = results
        return results
        
    def dfs(self,r, c, node, word, visited, results):
        DIRS = [(-1,-1),(-1,0),(0,-1),(0,1),(1,0),(1,1),(-1,1),(1,-1)]
        ROWS,COLS = self.board.shape
        
        letter = self.board[r,c]
        if letter not in node.children:
            return
    
        node = node.children[letter]
        word += letter
    
        if node.is_word:
            results.add(word)
    
        visited.add((r, c))
    
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
    
            if (0 <= nr < ROWS and
                0 <= nc < COLS and
                (nr, nc) not in visited):
    
                self.dfs(nr, nc, node, word, visited, results)
    
        visited.remove((r, c))

# if __name__=='__main__':
#     # buildTrie()
#     with open(Path(__file__).parent/'words_alpha.txt') as words_file:
#         valid_words = set(words_file.read().upper().split())
#     trie = Trie()
#     for word in valid_words:
#         trie.insert(word)