import os
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt


def generate_moves(state):
    moves = []
    empty_index = state.index("_")
    
    # Move green to the right if possible
    if empty_index > 0 and state[empty_index - 1] == "G":
        moves.append(state[:empty_index - 1] + "_" + "G" + state[empty_index + 1:])
    
    # Move red to the left if possible
    if empty_index < len(state) - 1 and state[empty_index + 1] == "R":
        moves.append(state[:empty_index] + "R" + "_" + state[empty_index + 2:])
    
    # Jump green to the right if possible
    if empty_index > 1 and state[empty_index - 2] == "G":
        moves.append(state[:empty_index - 2] + "_" + state[empty_index - 1] + "G" + state[empty_index + 1:])
    
    # Jump red to the left if possible
    if empty_index < len(state) - 2 and state[empty_index + 2] == "R":
        moves.append(state[:empty_index] + "R" + state[empty_index + 1] + "_" + state[empty_index + 3:])

    return moves


def bfs_solve(start_state, goal_state):
    queue = deque([(start_state, [start_state])])
    visited = set([start_state])
    while queue:
        current_state, path = queue.popleft()
        if current_state == goal_state:
            return path
        for next_state in generate_moves(current_state):
            if next_state not in visited:
                visited.add(next_state)
                queue.append((next_state, path + [next_state]))

    return []


def draw_checkers(fig, ax, start, end):
    color_mapper = {"G": "green", "R": "red"}

    assert len(start) == len(end)
    grid_length = len(start)
    
    if grid_length == 3:
        x_max = 0.965
    elif grid_length == 4:
        x_max = 0.97
    elif grid_length in [5, 6]:
        x_max = 0.98
    elif grid_length in [7, 8]:
        x_max = 0.985
    elif grid_length == 9:
        x_max = 0.9875
    elif grid_length in [10, 11]:
        x_max = 0.99
    
    # Set the axis limits with extra space for separation
    ax.set_xlim(0, grid_length+0.1)
    ax.set_ylim(-0.1, 3.5)  # Increased height to 3 for space between rows
    
    # Draw vertical gridlines, discontinuing them in the vertical space between rows
    for x in range(grid_length + 1):
        ax.plot([x, x], [0, 1], color="grey", linestyle="-", linewidth=1)  # Bottom row
        ax.plot([x, x], [2, 3], color="grey", linestyle="-", linewidth=1)  # Top row
    
    # Draw horizontal gridlines for two rows, with space in between
    ax.axhline(1, 0, x_max, color="grey", linestyle="-", linewidth=1)  # Bottom row
    ax.axhline(0, 0, x_max, color="grey", linestyle="-", linewidth=1)  # Bottom row
    
    ax.axhline(2, 0, x_max, color="grey", linestyle="-", linewidth=1)  # Top row
    ax.axhline(3, 0, x_max, color="grey", linestyle="-", linewidth=1)  # Top row
    
    
    ax.axhline(1.5, 0, 0.99, color="grey", linestyle="dotted", linewidth=1)
    
    # Remove axis labels and ticks
    ax.axis("off")
    
    # Adjust checker positions to account for the new row positions
    start_checkers_positions = [(2, i, start[i]) for i in range(grid_length)]  # Adjusted to top row
    end_checkers_positions = [(0, i, end[i]) for i in range(grid_length)]  # Adjusted to bottom row
    
    # Draw black checkers
    for checker in start_checkers_positions:
        if checker[2] in ["G", "R"]:
            circle = plt.Circle((checker[1] + 0.5, checker[0] + 0.5), 0.4, color=color_mapper[checker[2]])
            ax.add_patch(circle)
    
    # Draw red checkers
    for checker in end_checkers_positions:
        if checker[2] in ["G", "R"]:
            circle = plt.Circle((checker[1] + 0.5, checker[0] + 0.5), 0.4, color=color_mapper[checker[2]])
            ax.add_patch(circle)
    
    
    ax.text(
        grid_length/2, 3.25, "Starting Configuration", 
        ha="center", va="bottom", fontsize=13, color="black"
    )
    
    ax.text(
        grid_length/2, -0.5, "Ending Configuration", 
        ha="center", va="bottom", fontsize=13, color="black"
    )


if __name__ == "__main__":

    os.makedirs("data/images/checker_move")

    data, question_index, num_instances = [], 0, 100
    progress_bar = tqdm(range(num_instances))

    # Generate candidate positions
    candidates1, candidates2 = [], []
    for count1 in range(1, 6):
        for count2 in range(1, 6):
            if count1 + count2 > 2:
                start_state = f"{'G' * count1}_{'R' * count2}"
                goal_state = f"{'R' * count2}_{'G' * count1}"
                solution = bfs_solve(start_state, goal_state)
                assert len(solution) - 1 == count1 * count2 + count1 + count2
        
                candidates1.append((start_state, goal_state, len(solution) - 1, solution))
        
                for i, intermediate1 in enumerate(solution):
                    for j, intermediate2 in enumerate(solution):
                        if intermediate1 != intermediate2 and i < j and i + 5 >= j:
                            candidates2.append((intermediate1, intermediate2, j - i, solution[i:j+1]))

    random.shuffle(candidates2)
    final_pool = candidates1 + candidates2
    final_pool = final_pool[:num_instances]


    # Write data
    while question_index < num_instances:
        start, end, answer, moves = final_pool[question_index]

        grid_length = len(start)
        num_green, num_red = start.count("G"), start.count("R")

        assert answer == len(moves) - 1
        assert start.count("G") == end.count("G") and start.count("R") == end.count("R")
        
        fig, ax = plt.subplots(figsize=(grid_length, 3.75))
        draw_checkers(fig, ax, start, end)

        fname = f"data/images/checker_move/checker_move_{question_index:04}.jpg"
        plt.savefig(fname, bbox_inches="tight", dpi=200)
        plt.close(fig)

        question = f"A checker game is being played on a grid of {grid_length} squares with {num_green} green and {num_red} " + \
                   f"red checkers. Initially, the checkers are arranged as shown in the starting configuration with the " + \
                   f"{grid_length - 1} checkers occupying {grid_length - 1} squares and one unoccupied square. Green checkers only move " + \
                   f"rightward and red checkers only move leftward. Every move is either i) a slide to the adjacent empty square, or " + \
                   f"ii) a jump over one position to an empty square, provided the checker being jumped over is of a different colour. " + \
                   f"Each square can accommodate a maximum of one checker at any time. How many moves are required to reach the " + \
                   f"ending configuration from the starting configuration following the specified rules?"

        example = {
            "image": fname[5:], "question": question, "answer": int(answer),
            "solution": {
                "start_position": start, "end_position": end,
                "green_checkers": num_green, "red_checkers": num_red,
                "moves": moves
            }
        }
        
        data.append(example)
        question_index += 1
        progress_bar.update(1)


    with open("data/checker_move.json", "w") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")

