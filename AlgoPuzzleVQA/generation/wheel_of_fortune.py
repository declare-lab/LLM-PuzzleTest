import os
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


def get_segment_angles(segments=8):

    if segments in [6, 8]:
        minimum = 30
        pool = [15, 30, 45, 60]

    elif segments == 10:
        minimum = 20
        pool = [10, 25, 40, 55]
        
    chosen = []
    while sum(chosen) != 360:
        chosen = [minimum] * segments
        available = 360 - sum(chosen)
    
        for i in range(segments):
            value = min(random.choice(pool), available)
            chosen[i] += value
            available -= value

    random.shuffle(chosen)

    return chosen
    

def wheel(fig, ax, segments, equal_segments=False):

    all_colors = [
        "orangered", "silver", "yellow", "royalblue", "forestgreen", "darkorange", "magenta", "cyan", "lime", "blanchedalmond"
    ]

    all_prizes = [
        "Vacation", "Pizza", "Ice Cream", "Car", "Yacht", "Money", "Jewelry", "Laptop", "Watch", "Chocolate", "Camera"
    ]
    
    segment_colors = random.sample(all_colors, segments)
    segment_prizes = random.sample(all_prizes, segments)
    if equal_segments:
        segment_angles = [360 / segments] * segments
    else:
        segment_angles = get_segment_angles(segments)
    
    # Remove grid and ticks
    ax.grid(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Remove the polar labels
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
    offset = 45 * np.pi / 180 
    arrow_at = random.choice(list(range(8)))

    # Draw the segments of the wheel
    boundaries = []
    for i in range(segments):
        theta1 = sum(segment_angles[:i]) * np.pi / 180 + offset
        theta2 = sum(segment_angles[:i+1]) * np.pi / 180 + offset
        bound = [theta1 *  180 / np.pi, theta2 *  180 / np.pi]
        bound = [round(item, 1) for item in bound]
        if i == 0:
            boundaries += bound
        else:
            boundaries.append(bound[-1])
            
        ax.bar((theta1 + theta2) / 2, 1, width=theta2-theta1, bottom=0.05, color=segment_colors[i])
    
        theta_mid = (theta1 + theta2) / 2 
        ax.text(
            theta_mid, 0.65, segment_prizes[i], horizontalalignment="center", verticalalignment="center", 
            rotation=theta_mid * 180/np.pi, rotation_mode="anchor", fontsize=33
        )
        
    # Draw the central circle
    center_circle = plt.Circle((0, 0), 0.1, transform=ax.transData._b, color="white", zorder=0)
    ax.add_artist(center_circle)
    
    # Add the arrow at the top
    ax.annotate(
        "", xy=(arrow_at * offset, 1.05), xytext=(arrow_at * offset, 1.25), 
        arrowprops=dict(facecolor="brown", shrink=0, width=10, headwidth=20)
    )
    
    return boundaries[:-1], segment_angles, segment_colors, segment_prizes, arrow_at * 45


def get_question(segments, direction, mode):
    question = \
        f"A fortune wheel has {segments} segments of different colour. The initial position of the wheel is shown in the figure. " + \
        f"Each segment is associated with a prize as shown in the embedded text within the segment. " + \
        f"The axis of rotation of the wheel passes through its center and is perpendicular to the surface of the wheel. " \
        f"You spin the wheel {direction} and it {mode} before stopping. "  + \
        f"You are going to win the prize for the segment that now falls in front of the brown arrow. " + \
        f"What is your prize?"

    return question


def rotate(boundaries, colors, prizes, angle):
    
    new_boundaries = [(b + angle) % 360 for b in boundaries]
    least = new_boundaries.index(min(new_boundaries))

    new_boundaries = new_boundaries[least:] + new_boundaries[:least]
    new_colors = colors[least:] + colors[:least]
    new_prizes = prizes[least:] + prizes[:least]
    
    return new_boundaries, new_colors, new_prizes


def get_random_rotation(direction=1, mode=1):
    if mode == 1:
        rotations = random.choice(list(range(1, 6))) + random.choice(np.array(list(range(0, 20)))/20)
        angles = int(np.round(direction * 360 * (rotations % 1)))
        return rotations, angles
    else:
        angles = direction * random.choice(list(range(90, 1815, 15)))
        return angles


def create_instance(boundaries, segment_angles, segment_colors, segment_prizes, arrow_at):

    x, y = random.uniform(0, 1), random.choice([-1, 1])
    if y == 1:
        direction = "counterclockwise"
    else:
        direction = "clockwise"
        
    if x < 0.5:
        full_rotations, angle_rotated = get_random_rotation(y, 1)
        while arrow_at in rotate(boundaries, segment_colors, segment_prizes, angle_rotated)[0]:
            full_rotations, angle_rotated = get_random_rotation(y, 1)
        mode = f"makes {full_rotations} full rotations"

    else:
        angle_rotated = get_random_rotation(y, 2)
        while arrow_at in rotate(boundaries, segment_colors, segment_prizes, angle_rotated)[0]:
            angle_rotated = get_random_rotation(y, 2)
        mode = f"rotates {np.abs(angle_rotated)} degrees"
    
    question = get_question(len(segment_prizes), direction, mode)
    new_boundaries, new_colors, new_prizes = rotate(boundaries, segment_colors, segment_prizes, angle_rotated)

    index = -1
    for item in new_boundaries:
        if arrow_at < item:
            break
        index += 1

    answer = new_prizes[index]

    instance = {
        "question": question, "answer": answer,
        "solution": {
            "angles": segment_angles, "arrow_at": arrow_at, "effective_rotation": (angle_rotated % 360) * y, "direction": direction,
            "original_colors": segment_colors, "original_boundaries": boundaries, "original_prizes": segment_prizes,
            "rotated_colors": new_colors, "rotated_boundaries": new_boundaries, "rotated_prizes": new_prizes,
        }
    }
    
    return instance


if __name__ == "__main__":

    os.makedirs("data/images/wheel_of_fortune")

    data, question_index, num_instances = [], 0, 100
    progress_bar = tqdm(range(num_instances))
    solution_set = []

    while question_index < num_instances:
        if question_index < 50:
            equal_segments = True
        else:
            equal_segments = False

        if question_index % 50 < 15:
            num_segments = 6
        elif question_index % 50 < 35:
            num_segments = 8
        else:
            num_segments = 10

        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={"projection": "polar"})
        boundaries, segment_angles, segment_colors, segment_prizes, arrow_at = wheel(fig, ax, num_segments, equal_segments=equal_segments)
        instance = create_instance(boundaries, segment_angles, segment_colors, segment_prizes, arrow_at)

        # Ensure no repetition of data
        if instance["solution"] not in solution_set:
            solution_set.append(instance["solution"])

            fname = f"data/images/wheel_of_fortune/wheel_of_fortune_{question_index:04}.jpg"
            plt.savefig(fname, bbox_inches="tight", dpi=300)
            plt.close(fig)

            instance = {**{"image": fname[5:]}, **instance}

            data.append(instance)
            question_index += 1
            progress_bar.update(1)

    with open("data/wheel_of_fortune.json", "w") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")

