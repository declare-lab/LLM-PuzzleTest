import os
import json
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def solveNQueens(n):
    # Solutions for board size of n * n
    def DFS(queens, xy_dif, xy_sum):
        p = len(queens)
        if p==n:
            result.append(queens)
            return None
        for q in range(n):
            if q not in queens and p-q not in xy_dif and p+q not in xy_sum: 
                DFS(queens+[q], xy_dif+[p-q], xy_sum+[p+q])  
    result = []
    DFS([],[],[])
    positions = []
    for sol in result:
        positions.append([(i, index) for i, index in enumerate(sol)])

    random.shuffle(positions)
    return positions


def chessboard(fig, ax, size):
    # Draw chessboard
    chessboard = np.zeros((size, size, 3))

    color1 = [255, 230, 179]
    color2 = [255, 195, 57]
    
    color1 = [c/255 for c in color1]
    color2 = [c/255 for c in color2]

    chessboard[::2, ::2] = color1
    chessboard[1::2, 1::2] = color1
    chessboard[1::2, ::2] = color2
    chessboard[::2, 1::2] = color2 
            
    ax.imshow(chessboard, interpolation="nearest")
    ax.axis("off")
    return chessboard


def add_queen(ax, coord):
    # Add a queen to the board
    img = plt.imread("templates/n_queens/queen.png")
    imagebox = OffsetImage(img, zoom=0.15)
    ab = AnnotationBbox(imagebox, coord, frameon=False)
    ax.add_artist(ab)


if __name__ == "__main__":

    os.makedirs("data/images/n_queens")

    positions_8 = solveNQueens(8)
    positions_9 = solveNQueens(9)
    positions_10 = solveNQueens(10)

    data, question_index, num_instances = [], 0, 100

    for j in tqdm(range(num_instances)):
        if j < 30:
            x, size, sol = "an", 8, positions_8[j]
        elif j < 65:
            x, size, sol = "a", 9, positions_9[j]
        else:
            x, size, sol = "a", 10, positions_10[j]

        fig, ax = plt.subplots(figsize=(8, 8))
        board = chessboard(fig, ax, size)
        random.shuffle(sol)
        given, missing = sol[:-2], sol[-2:]

        given.sort(key = lambda x: x[1])
        missing.sort(key = lambda x: x[1])

        answer = np.abs(missing[0][0] - missing[1][0]) + np.abs(missing[0][1] - missing[1][1])

        for pos in given:
            add_queen(ax, pos)

        fname = f"data/images/n_queens/n_queens_{question_index:04}.jpg"
        plt.savefig(fname, bbox_inches="tight", dpi=200)
        plt.close(fig)

        question = f"You are given {x} {size} * {size} chessboard. The Manhattan distance between two squares in a chessboard is equal to the " + \
               f"minimal number of orthogonal King moves between these squares on the otherwise empty board. The objective is to place {size} " + \
               f"chess queens on this board so that no two queens threaten each other; i.e. no two queens share the same row, column, or " + \
               f"diagonal. {size-2} queens have already been placed in some of the squares of the board, as shown in the image. Suppose you " + \
               f"pick two squares to place the two remaining queen pieces in a way that fulfills the objective. What is the " + \
               f"Manhattan distance between these two squares?"

        example = {
            "image": fname[5:], "question": question, "answer": int(answer),
            "solution": {
                "rows": int(size), "columns": int(size),
                "given_queen_positions": [(c, r) for r, c in given],
                "missing_queen_positions": [(c, r) for r, c in missing],
            }
        }

        data.append(example)
        question_index += 1


    with open("data/n_queens.json", "w") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")

