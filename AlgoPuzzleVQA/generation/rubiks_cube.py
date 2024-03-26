import os
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt

import magiccube
from magiccube.cube_base import Color, CubeException, Face
from magiccube.solver.basic.basic_solver import BasicSolver


mapper = {
    "Y": "yellow", "B": "blue", "R": "red", "G": "green", "O": "orange", "W": "grey"
}
face_ids = ["left", "right", "down", "up", "back", "front"]


def cube_faces(cube):
    faces = {}
    for k, id_ in enumerate(face_ids):
        face = cube.get_face(Face(k))
    
        flat_face = [mapper[str(cell)[-1]] for row in face for cell in row]
        faces[id_] = flat_face

    return faces


def unflatten_face(faces):
    new_faces = {}
    for k, v in faces.items():
        new_faces[k] = [
            [v[0], v[1], v[2]],
            [v[3], v[4], v[5]],
            [v[6], v[7], v[8]]
        ]

    return new_faces


def plot_cube(fig, axs, cube_faces):
    plt.subplots_adjust(wspace=8, hspace=5)

    # colours for each face
    colours = {
        "grey": [200, 200, 200],
        "yellow": [255, 255, 0],
        "green": [0, 150, 0],
        "blue": [0, 127, 255],
        "orange": [255, 143, 50],
        "red": [255, 0, 0]
    }

    # Function to plot a single face with gridlines and label
    def plot_face_with_gridlines_and_label(ax, face_id, label):
        
        grid = np.array([colours[cell] for cell in cube_faces[face_id]]).reshape(3, 3, -1)        
        ax.imshow(grid)

        # Add gridlines
        for x in range(1, 3):
            ax.axhline(y=x - 0.5, color="black", linewidth=0.075)
            ax.axvline(x=x - 0.5, color="black", linewidth=0.075)

        # Add label
        ax.text(1, -0.75, label, ha="center", va="center", fontsize=11, color="black")

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    # Hide all axes initially
    for ax in axs.ravel():
        ax.axis("off")

    # Plot each face on the appropriate subplot with gridlines and label
    plot_face_with_gridlines_and_label(axs[0, 1], "up", "Up")        # Top face
    plot_face_with_gridlines_and_label(axs[1, 0], "left", "Left")    # Left face
    plot_face_with_gridlines_and_label(axs[1, 1], "front", "Front")  # Front face
    plot_face_with_gridlines_and_label(axs[1, 2], "right", "Right")  # Right face
    plot_face_with_gridlines_and_label(axs[1, 3], "back", "Back")    # Back face
    plot_face_with_gridlines_and_label(axs[2, 1], "down", "Down")    # Bottom face

    plt.tight_layout()


if __name__ == "__main__":

    os.makedirs("data/images/rubiks_cube")

    data, question_index, num_instances = [], 0, 100
    progress_bar = tqdm(range(num_instances))
    solution_set = []

    while question_index < num_instances:

        colours = ["Y", "R", "G", "O", "B", "W"]
        random.shuffle(colours)
        colour_string = ""
        for c in colours:
            colour_string += c  * 9

        cube = magiccube.Cube(3, colour_string)

        if question_index >= 40:
            initial_move_seq = []
            for k in range(20):
                initial_move_seq.append(random.choice(["F", "B", "R", "L", "U", "D"]) + random.choice(["1", "2", "3"]))
            initial_move_seq = " ".join(initial_move_seq)
            cube.rotate(initial_move_seq)
        
        initial_faces = cube_faces(cube)

        move_seq = []
        num_moves = random.choice([1, 2, 3])
        for k in range(num_moves):
            n = random.choice(["1", "2", "3"])
            dir = random.choice(["F", "B", "R", "L", "U", "D"])
            if n != "1":
                dir = dir + n
            move_seq.append(dir)
        move_seq = " ".join(move_seq)

        fig, axs = plt.subplots(3, 4, figsize=(7, 6))
        plot_cube(fig, axs, initial_faces)

        cube.rotate(move_seq)
        final_faces = cube_faces(cube)
        
        query_face = random.choice(face_ids)
        if random.uniform(0, 1) < 0.75:
           query_colour = random.choice(list(set(final_faces[query_face])))
        else:
            query_colour = random.choice(list(mapper.values()))
        answer = final_faces[query_face].count(query_colour)

        question = f"A 3 * 3 Rubik's Cube has six different coloured panels: red, green, blue, yellow, orange, and grey. The initial " + \
               f"state of the cube in terms of the different colour positions in its six faces is shown in the image. To represent " + \
               f"the movements of the cube we use six letters: U for Up, D for Down, L for Left, R for Right, F for Front, B for Back. " + \
               f"These letters are used in sequence where you need to perform each letter in the sequence from left to right. Each " + \
               f"letter tells you to move that face clockwise by 90 degrees. A number 'n' immediately after a letter denotes " + \
               f"that you need to move that face clockwise by 90 * n degrees. For example, 'U R3' would mean rotating the up face 90 " + \
               f"degrees clockwise and then rotating the right face 270 degrees clockwise. You perform the move sequence '{move_seq}' " + \
               f"starting from the state shown in the image. What would be the number of small 1 * 1 {query_colour} squares in the " + \
               f"{query_face} face after completing the move sequence?"


        solution = {
            "moves": move_seq, "query_face": query_face, "query_colour": query_colour,
            "start_position": unflatten_face(initial_faces), "end_position": unflatten_face(final_faces)
        }

        if solution not in solution_set:
            solution_set.append(solution)

            fname = f"data/images/rubiks_cube/rubiks_cube_{question_index:04}.jpg"
            plt.savefig(fname, bbox_inches="tight", dpi=300)
            plt.close(fig)

            example = {
                "image": fname[5:], "question": question, "answer": int(answer),
                "solution": solution
            }

            data.append(example)
            question_index += 1
            progress_bar.update(1)


    with open("data/rubiks_cube.json", "w") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")

