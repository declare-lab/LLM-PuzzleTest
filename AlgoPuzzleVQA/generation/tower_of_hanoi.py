import os
import json
import copy
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt


def tower_of_hanoi_basic(source, auxiliary, destination, numOfDisk, disks, results):
    if numOfDisk > 0:
        # recursively calling for moving the n-1 disk from source to auxiliary using destination. 
        tower_of_hanoi_basic(source, destination, auxiliary, numOfDisk-1, disks, results)
        
        plate = disks[source].pop()
        disks[destination].append(plate)
        disks["moves"] += 1
        results.append(copy.deepcopy(disks))
        
        # recursively asking to move remaining disk from auxiliary to destination using source.
        tower_of_hanoi_basic(auxiliary, source, destination, numOfDisk-1, disks, results)



mapping = {0: "A", 1: "B", 2: "C"}
def tower_of_hanoi_arbitary(diskPositions, largestToMove, destination, disks, results):
    for disk in range(largestToMove, len(diskPositions)):

        source = diskPositions[disk]         
        if source != destination:

            # sum of the peg numbers is 3, so to find the other one ...
            otherPeg = 3 - destination - source

            # before we can move disk, we have get the smaller ones out of the way
            tower_of_hanoi_arbitary(diskPositions, disk+1, otherPeg, disks, results)

            diskPositions[disk] = destination

            plate = disks[mapping[source]].pop(0)
            disks[mapping[destination]].insert(0, plate)
            disks["moves"] += 1
            results.append(copy.deepcopy(disks))

            # now we can put the smaller ones in the right place
            tower_of_hanoi_arbitary(diskPositions, disk+1, destination, disks, results)

            break;


def modify_results_arbitary_tower(results):
    start = results[0]
    num_disks = len(start["A"] + start["B"] + start["C"])
    mapping = {k: num_disks - k for k in range(num_disks)}
    new_results = []
    for item in results:
        new_results.append(
            {
                "moves": item["moves"],
                "A": [mapping[disk] for disk in item["A"][::-1]],
                "B": [mapping[disk] for disk in item["B"][::-1]],
                "C": [mapping[disk] for disk in item["C"][::-1]]
            }
        )

    return new_results


def draw_tower_of_hanoi(fig, ax, start_positions, end_positions):
    """
    Draw the Tower of Hanoi puzzle with start and end configurations.
    """
    # Basic settings
    disk_height, peg_width, base_height = 1, 0.5, 0.3
    n_disks = sum(len(peg) for peg in start_positions)
    peg_height = n_disks * disk_height * 1.3
    min_disk_width, size_diff = 2.5, 1.75
    max_disk_width = min_disk_width + size_diff * (n_disks - 1)
    
    # Color and style settings
    peg_color = "#654321"
    disk_colors = ["#ffdab3", "#ffb566", "#ff901a", "#cc6900", "#993300", "#cc0000"]

    # Space between pegs and width of base
    space_between_pegs = 1.25 * max_disk_width
    base_width = 4 * space_between_pegs

    if n_disks == 6:
        line_y = peg_height + disk_height + 1
        offset, base_offset_start, base_offset_end = 11.8, 0.05, 0.05
    elif n_disks == 5:
        line_y = peg_height + disk_height + 1.25
        offset, base_offset_start, base_offset_end = 11.8, 0.04, 0.05
    elif n_disks == 4:
        line_y = peg_height + disk_height + 1.1
        offset, base_offset_start, base_offset_end = 10, 0.03, 0.05
    elif n_disks == 3:
        line_y = peg_height + disk_height + 1
        offset, base_offset_start, base_offset_end = 8, 0.02, 0.02

    # Base in end position
    ax.add_patch(
        plt.Rectangle(
            (-max_disk_width, -base_height-base_offset_end), base_width, base_height, 
            color="#654321", linewidth=1, zorder=1
        )
    )

    # Pegs in end position
    for i in range(3):
        peg_x_position = i * space_between_pegs
        ax.add_patch(
            plt.Rectangle(
                (peg_x_position - peg_width / 2, 0), peg_width, peg_height, 
                color=peg_color, linewidth=1, zorder=1
            )
        )

    # Discs in end position
    for i, peg in enumerate(end_positions):
        peg_x_position = i * space_between_pegs
        for j, disk in enumerate(peg):
            disk_width = min_disk_width + size_diff * (disk - 1)
            disk_color = disk_colors[disk - 1]
            ax.add_patch(
                plt.Rectangle(
                    (peg_x_position - disk_width / 2, - 0.02 + j * disk_height), disk_width, disk_height - 0.02, 
                    color=disk_color, linewidth=1, zorder=1
                )
            )

    ax.axhline(y=line_y, color="grey", linewidth=0.5)

    # Base in start position
    ax.add_patch(
        plt.Rectangle(
            (-max_disk_width, -base_height -base_offset_start + offset), base_width, base_height, 
            color="#654321", linewidth=1, zorder=1
        )
    )
    
    # Pegs in start position
    for i in range(3):
        peg_x_position = i * space_between_pegs
        ax.add_patch(
            plt.Rectangle(
                (peg_x_position - peg_width / 2, offset), peg_width, peg_height, 
                color=peg_color, linewidth=2, zorder=1
            )
        )

    # Discs in start position
    for i, peg in enumerate(start_positions):
        peg_x_position = i * space_between_pegs
        for j, disk in enumerate(peg):
            disk_width = min_disk_width + size_diff * (disk - 1)
            disk_color = disk_colors[disk - 1]
            ax.add_patch(
                plt.Rectangle(
                    (peg_x_position - disk_width / 2, offset - 0.02 + j * disk_height), disk_width, disk_height - 0.02, 
                    color=disk_color, linewidth=1, zorder=1
                )
            )

    # Setting the limits and removing axes for better visual
    ax.set_xlim(-0.6 * max_disk_width, 3.1 * max_disk_width)
    ax.set_ylim(-3, offset + peg_height + disk_height + 2)
    ax.axis("off")
    ax.set_aspect("equal", adjustable="box")
        
    ax.text(
        space_between_pegs, peg_height + offset + disk_height + 0.5, "Starting Configuration", 
        ha="center", va="bottom", fontsize=16, color="black"
    )
    ax.text(
        space_between_pegs, -1.75, "Ending Configuration", 
        ha="center", va="top", fontsize=16, color="black"
    )

    return fig, ax



if __name__ == "__main__":

    os.makedirs("data/images/tower_of_hanoi")

    data, question_index, num_instances = [], 0, 100
    progress_bar = tqdm(range(num_instances))
    solution_set = []

    while question_index < num_instances:
        num_disks = np.random.choice([3, 4, 5, 6], p=[0.13, 0.21, 0.33, 0.33])
        answer = np.random.choice([1, 2, 3, 4, 5, 6], p=[0.18, 0.18, 0.17, 0.16, 0.16, 0.15])
        
        if question_index < 50:
            results = []
            disks = {
                "moves": 0,
                "A": list(range(num_disks, 0, -1)), "B": [], "C": []
            }
            results.append(copy.deepcopy(disks))
            tower_of_hanoi_basic("A", "B", "C", num_disks, disks, results)

        else:
            results = []
            disk_indices, start = [random.randint(0, 2) for _ in range(num_disks)], [[], [], []]

            for k in range(num_disks - 1, -1, -1):
                start[disk_indices[k]].append(k)
            
            disks = {
                "moves": 0,
                "A": start[0], "B": start[1], "C": start[2]
            }
            results.append(copy.deepcopy(disks))
            tower_of_hanoi_arbitary(disk_indices, 0, 2, disks, results)
            results = modify_results_arbitary_tower(results)

        start = random.choice(results)
        start_index = start["moves"]

        try:
            if start_index < answer:
                end = results[start_index + answer]
            elif start_index + answer >= len(results):
                end = results[start_index - answer]
            else:
                end = random.choice([results[start_index + answer], results[start_index - answer]])
        
            start_positions = [start[key] for key in ["A", "B", "C"]]
            end_positions = [end[key] for key in ["A", "B", "C"]]

            moves = []
            index = min(start["moves"], end["moves"])
            for k in range(index, index + answer + 1):
                elem = {
                    "position": [results[k]["A"], results[k]["B"], results[k]["C"]],
                    "num_moves": int(np.abs(results[k]["moves"] - start["moves"]))
                }
                moves.append(elem)
        
            if start["moves"] > end["moves"]:
                moves = moves[::-1]
            
            solution = {
                "start_position": start_positions, "end_position": end_positions,
                "moves": moves
            }

            # Ensure no repetition of data
            if solution not in solution_set:
                solution_set.append(solution)
            
                # Draw the figure
                fig, ax = plt.subplots(figsize=(12, 12))
                fig, ax = draw_tower_of_hanoi(fig, ax, start_positions, end_positions)
            
                fname = f"data/images/tower_of_hanoi/tower_of_hanoi_{question_index:04}.jpg"
                plt.savefig(fname, bbox_inches="tight", dpi=200)
                plt.close(fig)

                question = f"You are playing a Tower of Hanoi game with 3 rods and {num_disks} disks of various diameters, which can slide onto " + \
                   f"any rod. You are given the starting and ending configuration of the game as shown in the top and the bottom of the image, " + \
                   f"respectively. The game has the following rules: i) Only one disk may be moved at a time; ii) Each move consists of taking " + \
                   f"the upper disk from one of the stacks and placing it on top of another stack or on an empty rod; and iii) No disk can be " + \
                   f"placed on top of a disk that is smaller than it. What is the minimum number of moves required to go from the starting to " + \
                   f"the ending configuration?"
            
                example = {
                    "image": fname[5:], "question": question, "answer": int(answer),
                    "solution": solution
                }
            
                data.append(example)
                question_index += 1
                progress_bar.update(1)
            
        except IndexError:
            continue


    with open("data/tower_of_hanoi.json", "w") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")

