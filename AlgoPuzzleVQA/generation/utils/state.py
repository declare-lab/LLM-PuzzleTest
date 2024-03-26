'''
Adopted from: https://github.com/Pariasrz/N-Puzzle-solver-with-Search-Algorithms
'''

class State:    
    greedy_evaluation = None
    AStar_evaluation = None
    heuristic = None
    
    def __init__(self, n, state, parent, direction, depth, cost):
        self.state = state
        self.parent = parent
        self.direction = direction
        self.depth = depth

        if parent:
            self.cost = parent.cost + cost
        else:
            self.cost = cost

    # remove illegal moves for a given state
    @staticmethod
    def available_moves(x, n): 
        moves = ["Left", "Right", "Up", "Down"]
        if x % n == 0:
            moves.remove("Left")
        if x % n == n-1:
            moves.remove("Right")
        if x - n < 0:
            moves.remove("Up")
        if x + n > n*n - 1:
            moves.remove("Down")
        return moves


    # produces children of a given state
    def expand(self , n): 
        x = self.state.index(0)
        moves = self.available_moves(x, n)
        
        children = []
        for direction in moves:
            temp = self.state.copy()
            if direction == "Left":
                temp[x], temp[x - 1] = temp[x - 1], temp[x]
            elif direction == "Right":
                temp[x], temp[x + 1] = temp[x + 1], temp[x]
            elif direction == "Up":
                temp[x], temp[x - n] = temp[x - n], temp[x]
            elif direction == "Down":
                temp[x], temp[x + n] = temp[x + n], temp[x]

            # depth should be changed as children are produced
            children.append(State(n, temp, self, direction, self.depth + 1, 1))
        return children

