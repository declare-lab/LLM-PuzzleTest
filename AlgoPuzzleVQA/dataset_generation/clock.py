import os
import json
import math
import random
import numpy as np
import datetime as dt
from tqdm import tqdm
import matplotlib.pyplot as plt


def clock(fig, ax, time_str):
    # Parse the time
    hours, minutes = map(int, time_str.split(":"))

    # Draw the clock face
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal", "box")
    ax.axis("off")

    # Draw a circular boundary with a nice edge for the clock
    clock_circle = plt.Circle((0, 0), 1, color="black", fill=False, linewidth=2)
    ax.add_artist(clock_circle)

    # Draw the clock numbers in a stylish font
    for hour in range(1, 13):
        x = 0.85 * math.sin(math.radians(hour * 30))
        y = 0.85 * math.cos(math.radians(hour * 30))
        ax.text(x, y, str(hour), ha="center", va="center", fontsize=16, fontweight="bold", color="darkslategray")

    # Calculate hand angles
    hour_angle = (hours % 12 + minutes / 60) * 30  # 360 degrees / 12 hours
    minute_angle = minutes * 6  # 360 degrees / 60 minutes

    # Draw the hands in a more elegant style
    # Hour hand
    ax.plot([0, 0.5 * math.sin(math.radians(hour_angle))],
            [0, 0.5 * math.cos(math.radians(hour_angle))], color="saddlebrown", linewidth=7, solid_capstyle="round")
    
    # Minute hand
    ax.plot([0, 0.8 * math.sin(math.radians(minute_angle))],
            [0, 0.8 * math.cos(math.radians(minute_angle))], color="seagreen", linewidth=4, solid_capstyle="round")

    # Add a small circle at the center
    center_circle = plt.Circle((0, 0), 0.05, color="black", fill=True)
    ax.add_artist(center_circle)

    # Add tick marks for minutes
    for i in range(60):
        angle = math.radians(i * 6)
        x_start = math.sin(angle)
        y_start = math.cos(angle)
        if i % 5 == 0:
            x_end = 0.93 * x_start
            y_end = 0.93 * y_start
            ax.plot([x_start, x_end], [y_start, y_end], color="grey", linewidth=2.5)
        else:
            x_end = 0.95 * x_start
            y_end = 0.95 * y_start
            ax.plot([x_start, x_end], [y_start, y_end], color="grey", linewidth=1)            


def compute_time(current_hour, current_minute, delta_hour, delta_minute):
    x = dt.time(hour=current_hour, minute=current_minute)
    delta = dt.timedelta(hours=delta_hour, minutes=delta_minute)
    result = (dt.datetime.combine(dt.date(5,5,5), x) + delta).time()
    
    return [result.hour % 12, result.minute]


def create_question(mode, delta_hour=0, delta_minute=0):

    names = [
        "Emily", "Jacob", "Hannah", "Michael", "Madison", "Matthew", "Ashley", "Joshua", "Sarah", "Chris",
        "Alexis", "Nicholas", "Samantha", "Andrew", "Jessica", "Joseph", "Elizabeth", "Daniel", "Taylor", "Tyler"
    ]

    if mode == 0:
        return "The clock is a standard analog clock without the seconds hand. What is the time shown in the clock?"

    person = random.choice(names)
    if delta_hour == 0:
        marker = f"{delta_minute} minutes"
    elif delta_hour == 1:
        marker = f"{delta_hour} hour {delta_minute} minutes"
    elif delta_hour > 1:
        marker = f"{delta_hour} hours {delta_minute} minutes"
    
    if mode == 1:
        question = f"{person} came to an event {marker} ago. The current time is shown on the clock. " + \
        f"The clock is a standard analog clock without the seconds hand. What was the time when {person} came to the event?"

    elif mode == 2:
        question = f"{person}'s event is going to start in {marker}. The current time is shown on the clock. " + \
        f"The clock is a standard analog clock without the seconds hand. What will be the time when the event starts?"

    return question


if __name__ == "__main__":

    os.makedirs("data/images/clock")

    data, question_index, num_instances = [], 0, 100
    progress_bar = tqdm(range(num_instances))
    solution_set = []

    while question_index < num_instances:
        current_hour = random.randint(1, 12)
        current_minute = random.randint(0, 59)

        delta_hour = random.randint(0, 3)
        delta_minute = random.randint(1, 59)

        mode = random.randint(1, 2)

        question = create_question(mode, delta_hour, delta_minute)
        solution = {
            "current_hour": current_hour, "current_minute": current_minute
        }

        current_time = f"{current_hour}:{current_minute:02d}"
        
        if mode == 0:
            answer = current_time
        
        elif mode == 1:
            result = compute_time(current_hour, current_minute, - delta_hour, - delta_minute)
            answer = f"{result[0]}:{result[1]:02d}"
            solution["subtract_hour"] = delta_hour
            solution["subtract_minute"] = delta_minute
        
        elif mode == 2:
            result = compute_time(current_hour, current_minute, + delta_hour, + delta_minute)
            answer = f"{result[0]}:{result[1]:02d}"
            solution["add_hour"] = delta_hour
            solution["add_minute"] = delta_minute

        if solution not in solution_set:
            solution_set.append(solution)
            fig, ax = plt.subplots(figsize=(8, 8))
            clock(fig, ax, current_time)
        
            fname = f"data/images/clock/clock_{question_index:04}.jpg"
            plt.savefig(fname, bbox_inches="tight", dpi=500)
            plt.close(fig)
        
            example = {
                "image": fname[5:], "question": question, "answer": answer,
                "solution": solution
            }
        
            data.append(example)
            question_index += 1
            progress_bar.update(1)


    with open("data/clock.json", "w") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")
