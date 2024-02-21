''' 
Adopted from: https://rhettinger.github.io/puzzle.html
Author: Raymond Hettinger
'''

from collections import deque
from sys import intern
import re

class Puzzle:

    # default starting position
    pos = ""

     # ending position used by isgoal()
    goal = ""

    def __init__(self, pos=None):
        if pos: self.pos = pos

    # returns a string representation of the position for printing the object
    def __repr__(self):
        return repr(self.pos)

    # returns a string representation after adjusting for symmetry
    def canonical(self):
        return repr(self)

    def isgoal(self):
        return self.pos == self.goal

    # returns list of objects of this class
    def __iter__(self):
        if 0: yield self

    def solve(pos, depthFirst=False):
        queue = deque([pos])
        trail = {intern(pos.canonical()): None}
        solution = deque()
        load = queue.append if depthFirst else queue.appendleft

        while not pos.isgoal():
            for m in pos:
                c = m.canonical()
                if c in trail:
                    continue
                trail[intern(c)] = pos
                load(m)
            pos = queue.pop()

        while pos:
            solution.appendleft(pos)
            pos = trail[pos.canonical()]

        return list(solution)
        