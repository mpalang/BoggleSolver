from numpy import typing as nt
import numpy as np
from pathlib import Path
from solving.trie import load_trie

class WordFinder:
    def __init__(self, board: nt.NDArray[str] | list[str] | None = None, dictionary: Path | str = 'small'):
        self.board = self.process_input(board)
        self.trie = load_trie(dictionary = dictionary) 
        if self.board is not None and self.trie is not None:
            self.solve()


    def process_input(self, board):
        if isinstance(board, list):
            if len(board) != 16:
                raise ValueError(f"If board is passed as list it requires 16 elements.")
            board = np.array(board).reshape(4,4)
        elif isinstance(board, np.ndarray):
            board = board
            if board.shape != (4,4):
                raise ValueError(f"If board is passed as array it requires shape (4x4) but {board.shape} was given.")
        else:
            board = None
        return board
        
    def solve(self, board: nt.NDArray[str]| list | None = None):
        board = self.process_input(board)
        if board is not None:
            self.board = board

        results = set()
        ROWS,COLS = self.board.shape

        for r in range(ROWS):
            for c in range(COLS):
                self.dfs(r, c, self.trie.root, "", set(), results)
        
        results = sorted(results, key=lambda x: (x,len(x)))

        self.valid_words = results
        return results

        
    def dfs(self,r, c, node, word, visited, results):
        DIRS = [(-1,-1),(-1,0),(0,-1),(0,1),(1,0),(1,1),(-1,1),(1,-1)]
        ROWS,COLS = self.board.shape
        
        letter = self.board[r,c].upper()
        if letter == 'QU':
            if 'Q' not in node.children:
                return
            word += 'Q'
            node = node.children['Q']
            if 'U' not in node.children:
                return
            word += 'U'
            node = node.children['U']
        
        else:
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