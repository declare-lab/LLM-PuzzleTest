import os
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


def mutilated_chessboard(fig, ax, rows, cols):
    chessboard = np.zeros((rows, cols, 3))
    
    color1 = [255, 195, 57]
    color2 = [255, 230, 179]

    color1 = [round(c/255, 2) for c in color1]
    color2 = [round(c/255, 2) for c in color2]

    chessboard[::2, ::2] = color1
    chessboard[1::2, 1::2] = color1
    chessboard[1::2, ::2] = color2
    chessboard[::2, 1::2] = color2 

    if rows * cols % 2 == 0:
        remove = np.random.choice(range(0, rows*cols - 1), 2, replace=False)
        remove.sort()
    else:
        remove = np.random.choice(range(0, rows*cols - 1), 1)

    removed_positions = []
    for cell in remove:
        chessboard[cell // cols, cell % cols] = [1, 1, 1]
        removed_positions.append([int(cell // cols), int(cell % cols)])

    colors = [0, 0]
    for cell in chessboard.reshape(-1, 3):
        if np.array_equal(cell, color1):
            colors[0] += 1
        elif np.array_equal(cell, color2):
            colors[1] += 1
            
    ax.imshow(chessboard, interpolation="nearest")
    ax.axis("off")

    return chessboard, colors, color1, color2, removed_positions


def create_question(rows, cols, num_removed):
    if num_removed == 1:
        remove = ["One of the squares", "cell"]
    elif num_removed == 2:
        remove = ["Two of the squares", "cells"]

    cells = rows * cols
    dominoes = (cells - num_removed) // 2
    
    question = f"The checkerboard shown in the image was originally of {rows} * {cols} in dimension having a total of " + \
    f"{cells} squares. It uses two colours of squares, one light yellow and one dark yellow, in a " + \
    f"chequered pattern. {remove[0]} have been removed from the board in the position of the white coloured " + \
    f"{remove[1]}, as shown in the image. You have {dominoes} dominoes of size 2 * 1. You can use them as is or you " + \
    f"can rotate them to use as a 1 * 2 domino. Is it possible to place all the {dominoes} dominoes in the " + \
    f"checkerboard to exactly cover all the remaining {cells - num_removed} squares? Answer Yes or No."

    return question


if __name__ == "__main__":

    os.makedirs("data/images/board_tile")

    data, question_index, num_instances = [], 0, 100
    progress_bar = tqdm(range(num_instances))
    solution_set = []

    while question_index < num_instances:

        rows, cols = random.choice(range(4, 10)), random.choice(range(5, 10))
        fig, ax = plt.subplots(figsize=(7, 7))
        board, colors, color1, color2, removed_positions = mutilated_chessboard(fig, ax, rows, cols)

        solution = {
            "board": board.tolist(),
            "rows": len(board), "columns": len(board[0]),
            "removed_positions": removed_positions,
            "color1": color1, "squares_of_color1": colors[0],
            "color2": color2, "squares_of_color2": colors[1],
        }

        if solution not in solution_set:
            solution_set.append(solution)
        
            fname = f"data/images/board_tile/board_tile_{question_index:04}.jpg"
            plt.savefig(fname, bbox_inches="tight", dpi=300)
            plt.close(fig)

            question = create_question(rows, cols, len(removed_positions))
            if colors[0] == colors[1]:
                answer = "Yes"
            else:
                answer = "No"

            example = {
                "image": fname[5:], "question": question, "answer": answer,
                "solution": solution
            }

            data.append(example)
            question_index += 1
            progress_bar.update(1)


    with open("data/board_tile.json", "w") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")

