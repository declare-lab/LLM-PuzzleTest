import os
import re
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
from utils.puzzle import Puzzle


class WoodSlide(Puzzle):
    """ WoodSlide
    This sliding block puzzle has 9 blocks of varying sizes:
    one 2x2, four 1x2, two 2x1, and two 1x1. The blocks are
    on a 5x4 grid with two empty 1x1 spaces. Starting from
    the position shown, slide the blocks around until the
    2x2 is in the lower left:

        1122
        1133
        45
        6788
        6799
    """

    pos = "11221133450067886799"
    goal = re.compile( r"................1..." )

    def isgoal(self):
        return self.goal.search(self.pos) != None

    def __repr__(self):
        ans = "\n"
        pos = self.pos.replace("0", ".")
        for i in [0, 4, 8, 12, 16]:
            ans = ans + pos[i:i+4] + "\n"
        return ans

    xlat = str.maketrans("38975","22264")

    def canonical(self):
        return self.pos.translate(self.xlat)

    block = { (0,-4), (1,-4), (2,-4), (3,-4),
              (16,4), (17,4), (18,4), (19,4),
              (0,-1), (4,-1), (8,-1), (12,-1), (16,-1),
              (3,1), (7,1), (11,1), (15,1), (19,1) }

    def __iter__(self):
        dsone = self.pos.find("0")
        dstwo = self.pos.find("0", dsone+1)
        for dest in [dsone, dstwo]:
            for adj in [-4, -1, 1, 4]:
                if (dest, adj) in self.block: continue
                piece = self.pos[dest+adj]
                if piece == "0": continue
                newmove = self.pos.replace(piece, "0")
                for i in range(20):
                    if 0 <= i+adj < 20 and self.pos[i+adj]==piece:
                        newmove = newmove[:i] + piece + newmove[i+1:]
                if newmove.count("0") != 2: continue
                yield WoodSlide(newmove)


def klotski_board(fig, ax, frame_start, frame_end):
    
    grid_start = [list(map(int, row)) for row in frame_start.split("\n")]
    grid_end = [list(map(int, row)) for row in frame_end.split("\n")]
    
    # Determine grid size
    rows = len(grid_start)
    cols = len(grid_start[0]) if rows > 0 else 0

    x_offset, y_offset = 0.25, 0.25
    
    ax.set_xlim(0, 2 * cols + 1.5)
    ax.set_ylim(0, rows + 5 * y_offset)
    ax.set_aspect("equal")
    
    # Invert y-axis to match the row indices
    ax.invert_yaxis()
    ax.axis("off")
    
    # Assign colors to each unique block
    wooden_colors = [
        "#8B4513", "#A0522D", "#C19A6B", "#CD853F", "#D2B48C", 
        "#DEB887", "#F4A460", "#8B7355", "#A67B5B", "#6E4C34"
    ]

    # Function to check if a cell is part of the current block
    def is_part_of_block(grid, x, y, block_val):
        return 0 <= x < cols and 0 <= y < rows and grid[y][x] == block_val

    # Draw each block as a single rectangle
    for y in range(rows):
        for x in range(cols):
            val_start, val_end = grid_start[y][x], grid_end[y][x]
            if val_start != 0:
                # Check if the cell has already been processed
                if (x == 0 or grid_start[y][x-1] != val_start) and (y == 0 or grid_start[y-1][x] != val_start):
                    # Find the width and height of the block
                    width = height = 1
                    while is_part_of_block(grid_start, x + width, y, val_start):
                        width += 1
                    while is_part_of_block(grid_start, x, y + height, val_start):
                        height += 1

                    # Create a rectangle for the block
                    rect = patches.Rectangle(
                        (x+x_offset, y+y_offset), width, height, linewidth=1.75, edgecolor="white",
                        facecolor=wooden_colors[val_start]
                    )
                    ax.add_patch(rect)

                    # Mark the cells of the block as processed
                    for dx in range(width):
                        for dy in range(height):
                            grid_start[y+dy][x+dx] = -1  # Mark as processed

            if val_end != 0:
                # Check if the cell has already been processed
                if (x == 0 or grid_end[y][x-1] != val_end) and (y == 0 or grid_end[y-1][x] != val_end):
                    # Find the width and height of the block
                    width = height = 1
                    while is_part_of_block(grid_end, x + width, y, val_end):
                        width += 1
                    while is_part_of_block(grid_end, x, y + height, val_end):
                        height += 1

                    # Create a rectangle for the block
                    rect = patches.Rectangle(
                        (x + 5.2, y+y_offset), width, height, linewidth=1.75, edgecolor="white",
                        facecolor=wooden_colors[val_end]
                    )
                    ax.add_patch(rect)

                    # Mark the cells of the block as processed
                    for dx in range(width):
                        for dy in range(height):
                            grid_end[y+dy][x+dx] = -1  # Mark as processed
                    
    border1 = patches.Rectangle(
        (x_offset/3, y_offset/2), cols + 0.3, rows + y_offset * 1.25, 
        linewidth=1, edgecolor="black", facecolor='none'
    )
    border2 = patches.Rectangle(
        (5 + x_offset/4, y_offset/2), cols + 0.3, rows + y_offset *  1.25, 
        linewidth=1, edgecolor="black", facecolor='none'
    )
    ax.add_patch(border1)
    ax.add_patch(border2)
    
    ax.text(
        2.25, 6, "Starting Configuration", 
        ha="center", va="bottom", fontsize=13, color="black"
    )

    ax.text(
        7.25, 6, "Ending Configuration", 
        ha="center", va="bottom", fontsize=13, color="black"
    )
    
    # Set labels and title
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.tight_layout() 


if __name__ == "__main__":

    os.makedirs("data/images/wood_slide")

    # Create data pool
    solution = WoodSlide().solve()
    positions = []
    for step in solution:
        current = step.pos
        positions.append("\n".join([current[i:i+4] for i in range(0, len(current), 4)]))

    max_moves_away = 5
    instance_pool = set()
    for i in range(len(positions)):
        for j in range(max(0, i - max_moves_away), min(len(positions), i + max_moves_away)):
            if j != i:
                index1, index2 = min(i, j), max(i, j)
                instance_pool.add((positions[index1], positions[index2], index2-index1))
                instance_pool.add((positions[index2], positions[index1], index2-index1))
                
    instance_pool = list(instance_pool)
    random.shuffle(instance_pool)

    # Write data
    data, question_index, num_instances = [], 0, 100

    for j in tqdm(range(num_instances)):

        frame_start, frame_end, answer = instance_pool[j]
        fig, ax = plt.subplots(figsize=(8, 12))
        klotski_board(fig, ax, frame_start, frame_end)

        fname = f"data/images/wood_slide/wood_slide_{question_index:04}.jpg"
        plt.savefig(fname, bbox_inches="tight", dpi=300)
        plt.close(fig)

        question = f"Consider a sliding block puzzle of grid size 5 * 4 units. It has 9 wooden blocks of varying sizes: one 2 * 2, " + \
               f"four 1 * 2, two 2 * 1, and two 1 * 1. The gird also has two empty 1 * 1 spaces. The blocks cannot be removed " + \
               f"from the grid, and may only be slid horizontally and vertically within its boundary. " + \
               f"A move is defined as selecting a block that is slideable, and moving it by 1 unit either horizontally or " + \
               f"vertically, whichever is possible. The image shows the starting and ending configurations of the puzzle grid. " + \
               f"The wooden blocks are shown in various shades of brown and the empty spaces are shown in white. What is the minimum " + \
               f"number of moves required to reach the ending configuration from the starting configuration?"

        index1 = positions.index(frame_start)
        index2 = positions.index(frame_end)

        if index2 > index1:
            moves = positions[index1: index2+1]
        else:
            moves = positions[index2: index1+1][::-1]

        
        example = {
            "image": fname[5:], "question": question, "answer": int(answer),
            "solution": {
                "start_position": frame_start, "end_position": frame_end,
                "moves": moves
            }
        }
        
        data.append(example)
        question_index += 1


    with open("data/wood_slide.json", "w") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")
