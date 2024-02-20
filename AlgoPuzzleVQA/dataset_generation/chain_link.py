import os
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont


def steps(x, cut_time=1, join_time=2):
    total_cuts = 0
    result = [x[:]]
    
    while x[0] < sum(x[1:]):
        i = 1
        while x[i] == 0:
            i += 1
        x[0] += 1
        x[i] -= 1
        if i != 1:
            x[i-1] += 1
        total_cuts += 1

        result.append(x[:])

    solution = {
        "steps": result, "cuts": total_cuts, "joins": x[0],
        "each_cut_time": cut_time, "each_join_time": join_time,
        "total_time": total_cuts * cut_time + x[0] * join_time
    }
    
    return solution


def get_xy_for_chain(top_x, top_y, width=100, height=60):
    top = (top_x, top_y)
    bottom = (top_x + width, top_y + height)

    return top, bottom

def get_dimension_for_chain(total_chains_x, total_chains_y, rows, cols, width=100, height=60, offset_x=50, offset_y=50):
    x = int((0.5 + total_chains_x/2) * width) + cols * offset_x
    y = int((0.5 + total_chains_y/2) * height) + rows * offset_y

    return x, y


def create_image(instance):
    # grid_dim_y, grid_dim_x = 250, 300
    pad_offset_x, pad_offset_y = 75, 75
    link_offset_x, link_offset_y = 45, 30
    outline = (255, 200, 0)
    opening_width, width, height = 20, 100, 60
    radius = width // 4

    total_pieces = sum(instance[0]) + sum(instance[1])
    if total_pieces < 10:
        rows, cols  = 3, 3
    elif total_pieces < 13:
        rows, cols = 3, 4
    elif total_pieces < 17:
        rows, cols = 4, 4
    
    flat_chains = [-1] * instance[0][0]
    for length, nums in enumerate(instance[1]):
        flat_chains += [length + 1] * nums
    
    grid_offset = len(flat_chains) % (rows * cols)
    if grid_offset != 0:
        grid_offset = rows * cols - grid_offset
    grid = np.array(list(np.abs(flat_chains)) + [0] * grid_offset).reshape(rows, cols)
    
    max_chains_each_row, max_chains_each_col = np.amax(grid, 1), np.amax(grid, 0)

    total_chains_x = sum(max_chains_each_col)
    total_chains_y  = sum(max_chains_each_row)
    
    image_width, image_height = get_dimension_for_chain(
        total_chains_x, total_chains_y, rows, cols, offset_x=pad_offset_x, offset_y=pad_offset_y
    )
    
    result = Image.new("RGB", (image_width + pad_offset_x, image_height + pad_offset_y), color=(255, 255, 255))
    draw = ImageDraw.Draw(result)
    
    for k, chain_length in enumerate(flat_chains):
        row_index = k // cols
        col_index = k % cols
    
        top_x, top_y = get_dimension_for_chain(
            sum(max_chains_each_col[:col_index]), sum(max_chains_each_row[:row_index]), row_index, col_index,
            offset_x=pad_offset_x, offset_y=pad_offset_y
        )

        if chain_length == -1:
            top_left, bottom_right = get_xy_for_chain(top_x, top_y, width, height * 1.1)
            left, top = top_left
            right, bottom = bottom_right
        
            # Draw the four corners
            draw.arc([(left, top), (left + radius * 2, top + radius * 2)], 180, 270, fill=outline, width=3)
            draw.arc([(right - radius * 2, top), (right, top + radius * 2)], 270, 360, fill=outline, width=3)
            draw.arc([(right - radius * 2, bottom - radius * 2), (right, bottom)], 0, 90, fill=outline, width=3)
            draw.arc([(left, bottom - radius * 2), (left + radius * 2, bottom)], 90, 180, fill=outline, width=3)
        
            # Draw the sides
            draw.line([(left + radius, top), (right - radius, top)], fill=outline, width=3)  # Top
            draw.line([(left + radius, bottom), (right - radius, bottom)], fill=outline, width=3)  # Bottom
            draw.line([(left, top + radius), (left, bottom - radius - opening_width)], fill=outline, width=3)  # Left
            draw.line([(right, top + radius), (right, bottom - radius)], fill=outline, width=3)  # Right
            
        else:
            for j in range(chain_length):
                top, bottom = get_xy_for_chain(top_x + j * link_offset_x, top_y + j * link_offset_y, width, height)
                draw.rounded_rectangle((top, bottom), radius, outline=outline, width=3)
            
    return result


def create_question(instance, cut_time, join_time):

    total_pieces = sum(instance[0]) + sum(instance[1])
    segments = [-1] * instance[0][0]
    
    for length, nums in enumerate(instance[1]):
        segments += [length + 1] * nums
    
    length = sum(np.abs(segments))
    
    question = f"Alice has {len(segments)} segments of chains of different lengths as shown in the image. " + \
    f"The total length of all the segments combined is {length} pieces. " + \
    f"She has a saw machine with which a closed piece can be cut opened. She also has a welding machine with which an open piece " + \
    f"can be closed. Each cut takes {cut_time} minutes and each welding takes {join_time} minutes. "
    
    if segments.count(-1) == 1:
        question += f"Initially, she has {segments.count(-1)} segment with 1 open piece as shown in the image. All the other pieces " + \
        "are closed. "
    
    elif segments.count(-1) > 1:
        question += f"Initially, she has {segments.count(-1)} segments each with 1 open piece as shown in the image. All the other pieces " + \
        "are closed. "
    
    else:
        question += f"Initially, there are no open pieces in any of the segments. "
    
    question += f"She now wants to make the longest possible necklace using all the available {length} pieces. Each piece in the necklace " + \
                f"would be connected to exactly two other pieces. This would require cutting open some pieces and then joining all the " + \
                f"resulting segments together. What is the minimum time in which she can create the necklace?"
    
    return question


if __name__ == "__main__":

    os.makedirs("data/images/chain_link", exist_ok=True)
            
    data, question_index, num_instances = [], 0, 100

    samples_set, samples = [], []
    while len(samples) < num_instances:
        x = [random.randint(0, 4) for j in range(4)] + [random.randint(0, 3)] + [random.randint(0, 2)]
        if sum(x) > 7 and sum(x) < 17 and x not in samples_set:
            samples_set.append(x)
            samples.append([[x[0]], x[1:]])

    for instance in tqdm(samples):
        cut_time = random.choice([2, 3, 4, 5])
        join_time = random.choice([2, 3, 4, 5])
        
        solution = steps(instance[0] + instance[1], cut_time, join_time)
        question = create_question(instance, cut_time, join_time)

        answer = solution["total_time"]
        
        instance_image = create_image(instance)
        fname = f"data/images/chain_link/chain_link_{question_index:04}.jpg"
        instance_image.save(fname, dpi=(1000, 1000))
        
        example = {
            "image": fname[5:], "question": question, "answer": answer,
            "solution": solution
        }

        data.append(example)
        question_index += 1

    with open("data/chain_link.json", "w") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")
