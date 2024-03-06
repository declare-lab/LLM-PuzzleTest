import os
import copy
import json
import heapq
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageOps, ImageDraw, ImageFont


def minSwaps(nums):
    keyToIndex = dict([(nums[i], i) for i in range(len(nums))])
    heap = nums[::]
    heapq.heapify(heap)

    swaps = 0
    for i in range(len(nums)):
        smallest = heapq.heappop(heap)

       # check if previous swappes made the array sorted
        if nums[i] != smallest:
            currNum = nums[i]
            nums[i], nums[keyToIndex[smallest]] = nums[keyToIndex[smallest]], nums[i]
            keyToIndex[smallest], keyToIndex[currNum] = keyToIndex[currNum], keyToIndex[smallest]
            swaps += 1

    return swaps


def transition(start, end, cols):
    red_gradient = int(float(end[0] - start[0]) / (cols - 1))
    green_gradient = int(float(end[1] - start[1]) / (cols - 1))
    blue_gradient = int(float(end[2] - start[2]) / (cols - 1))
    gradient = np.array([red_gradient, green_gradient, blue_gradient])
    
    grid = [start]
    for j in range(cols-2):
        next = start + gradient * (j+1)
        grid.append(next)
    grid.append(end)

    grid = np.array(grid, dtype=np.uint8).reshape(-1, cols, 3)

    return grid, gradient
    

def create_grid_random(cols=5, fixed="", size=(250, 250)):
    start = np.random.randint(50, 200, 3)
    diff = np.random.randint(-120, 120, 3)
    end = np.clip(start + diff, 0, 255)
    grid, gradient = transition(start, end, cols)
    im = Image.fromarray(grid).resize(size, resample=Image.NEAREST)

    return im, grid, gradient


def create_constrained_grid_1d(start, end, cols=5, size=(250, 250)):        
    start, end = np.array(start), np.array(end)
    grid, gradient = transition(start, end, cols)
    im = Image.fromarray(grid).resize(size, resample=Image.NEAREST)

    return im, grid, gradient


def create_constrained_grid_2d(start_top, end_top, start_bottom, end_bottom, rows=5, cols=5, size=(250, 250), fix_corners=False):    
    full_grid = np.empty((rows, cols, 3), dtype=np.uint8)
    row_top = transition(start_top, end_top, cols)[0][0]
    row_bottom = transition(start_bottom, end_bottom, cols)[0][0]
    
    full_grid[0] = row_top
    full_grid[rows-1] = row_bottom
    row_top, row_bottom = row_top.tolist(), row_bottom.tolist()
    
    for j in range(cols):
        full_grid[1:-1, j, :] = transition(row_top[j], row_bottom[j], rows)[0][0, 1: -1, :]

    im = Image.fromarray(full_grid).resize(size, resample=Image.NEAREST)

    return im, full_grid


def padding(img, expected_size):
    desired_size = expected_size
    delta_width = desired_size[0] - img.size[0]
    delta_height = desired_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, pad_height)
    
    return ImageOps.expand(img, padding, fill=(255, 255, 255))


def final_image(image1, image2):
    font = ImageFont.truetype("fonts/OpenSans-Regular.ttf", 90)
    background = Image.open("templates/hue_template.jpg")
    
    width, height = 2860, 1400
    background = padding(background, (width, height))
    copy = background.copy()
    
    left = 175
    right = width - 1024 - left
    text_left = left + 465
    text_right = text_left + 1490
    text_height = 1185
    
    copy.paste(image1, (left, 125)) 
    copy.paste(image2, (right, 125)) 
    
    ImageDraw.Draw(copy).text((text_left, text_height), "(A)", (0, 0, 0), font=font)
    ImageDraw.Draw(copy).text((text_right, text_height), "(B)", (0, 0, 0), font=font)
    
    return copy


def shuffle(full_grid, size, to_shuffle, shuffle_corners=True):

    rows, cols = full_grid.shape[:2]
    corners = [0, cols-1, cols * (rows - 1), rows * cols - 1]
    pool_indices = []
    for i in range(rows):
        for j in range(cols):
            index = i * cols + j 
            if shuffle_corners:
                pool_indices.append(index)
            elif index not in corners:
                pool_indices.append(index)

    to_shuffle = min(to_shuffle, len(pool_indices))
    selected_indices_shuffled = random.sample(pool_indices, to_shuffle)
    selected_indices = sorted(copy.deepcopy(selected_indices_shuffled))

    order = list(range(rows * cols))
    full_grid_reshaped = full_grid.reshape(-1, 3)
    shuffled_grid = copy.deepcopy(full_grid_reshaped)

    out_of_position = 0
    for i, j in zip(selected_indices, selected_indices_shuffled):
        shuffled_grid[j] = full_grid_reshaped[i]
        order[j] = i
        if i != j:
            out_of_position += 1

    swaps = minSwaps(copy.deepcopy(order))
    shuffled_grid = shuffled_grid.reshape(rows, cols, 3)
    shuffled_im = Image.fromarray(shuffled_grid).resize(size, resample=Image.NEAREST)

    return shuffled_im, shuffled_grid, swaps, out_of_position, order


colors_1d = [
    [[56, 52, 128], [156, 52, 128]], [[128, 52, 236], [128, 132, 236]], [[236, 52, 128], [236, 132, 208]], 
    [[52, 128, 236], [132, 208, 236]], [[236, 128, 52], [236, 208, 132]], [[128, 52, 236], [208, 132, 236]], 
    [[116, 52, 128], [156, 92, 168]], [[52, 128, 116], [92, 168, 156]], [[128, 116, 52], [168, 156, 92]], 
    [[142, 163, 173], [142, 255, 198]], [[163, 173, 142], [255, 198, 142]], [[163, 142, 173], [255, 142, 198]], 
    [[56, 152, 28], [56, 252, 128]], [[56, 28, 152], [56, 128, 252]], [[152, 56, 28], [252, 56, 128]], 
    [[255, 255, 0], [255, 132, 88]], [[0, 255, 255], [88, 132, 255]],  [[77, 77, 255], [42, 144, 255]], 
    [[255, 77, 77], [255, 144, 42]], [[255, 105, 180], [255, 221, 186]], [[49, 115, 215], [154, 92, 251]], 
    [[215, 49, 115], [251, 154, 92]], [[66, 201, 56], [2, 130, 201]], [[3, 132, 146], [4, 85, 173]],  
    [[146, 132, 3], [173, 85, 4]], [[3, 146, 132], [4, 113, 205]], [[146, 132, 3], [113, 205, 4]], 
    [[132, 3, 146], [205, 50, 113]], [[255, 182, 185], [97, 192, 191]],  [[246, 114, 128], [53, 92, 125]], 
    [[255, 111, 60], [255, 201, 60]], [[111, 60, 255], [201, 60, 255]], [[60, 255, 111], [60, 155, 201]],  
    [[255, 60, 111], [155, 60, 201]]
]


colors_2d = [
    [[236, 52, 128], [236, 132, 208], [116, 52, 128], [156, 92, 168]], [[236, 52, 128], [236, 132, 208], [142, 163, 173], [142, 255, 198]], 
    [[255, 255, 0], [255, 102, 178], [0, 205, 225], [60, 89, 233]], [[61, 88, 156], [0, 224, 52], [255, 51, 174], [255, 255, 26]], 
    [[255, 77, 77], [103, 42, 144], [255, 255, 26], [0, 210, 208]], [[236, 52, 128], [128, 52, 236], [236, 132, 128], [128, 132, 236]], 
    [[60, 255, 111], [201, 60, 255], [60, 155, 201], [111, 60, 255]], [[60, 255, 111], [255, 201, 60], [60, 155, 201], [255, 111, 60]], 
    [[255, 51, 204], [75, 40, 153], [255, 255, 0], [51, 204, 255]], [[0, 200, 162], [75, 40, 193], [255, 255, 50], [255, 51, 194]], 
    [[52, 128, 236], [236, 128, 52], [132, 208, 236], [236, 208, 132]], [[128, 52, 236], [236, 128, 52], [208, 132, 236], [236, 208, 132]], 
    [[52, 128, 236], [255, 111, 60], [132, 208, 236], [255, 201, 60]], [[255, 105, 180], [52, 128, 116], [255, 221, 186], [92, 168, 156]], 
    [[56, 52, 128], [156, 52, 128], [163, 142, 173], [255, 142, 198]], [[128, 52, 236], [128, 132, 236], [236, 128, 52], [236, 208, 132]], 
    [[128, 52, 236], [128, 132, 236], [255, 105, 180], [255, 221, 186]], [[236, 52, 128], [236, 132, 208], [88, 132, 255], [0, 255, 255]], 
    [[52, 128, 236], [132, 208, 236], [215, 49, 115], [251, 154, 92]], [[236, 128, 52], [236, 208, 132], [77, 77, 255], [42, 144, 255]], 
    [[128, 52, 236], [208, 132, 236], [255, 111, 60], [255, 201, 60]], [[52, 128, 116], [92, 168, 156], [111, 60, 255], [201, 60, 255]], 
    [[142, 163, 173], [142, 255, 198], [255, 102, 178], [255, 182, 0]], [[142, 163, 173], [142, 255, 198], [77, 77, 255], [42, 144, 255]], 
    [[163, 173, 142], [255, 198, 142], [128, 52, 236], [128, 132, 236]], [[163, 173, 142], [255, 198, 142], [88, 132, 255], [0, 255, 255]], 
    [[163, 173, 142], [255, 198, 142], [108, 67, 94], [203, 115, 88]], [[163, 173, 142], [255, 198, 142], [4, 113, 205], [3, 146, 132]], 
    [[163, 142, 173], [255, 142, 198], [88, 132, 255], [0, 255, 255]], [[163, 142, 173], [255, 142, 198], [108, 67, 94], [203, 115, 88]], 
    [[56, 28, 152], [56, 128, 252], [255, 132, 88], [255, 255, 0]], [[56, 28, 152], [56, 128, 252], [255, 105, 180], [255, 221, 186]], 
    [[152, 56, 28], [252, 56, 128], [97, 192, 191], [255, 182, 185]], [[255, 255, 0], [255, 132, 88], [42, 144, 255], [77, 77, 255]], 
    [[0, 255, 255], [88, 132, 255], [255, 201, 60], [255, 111, 60]], [[255, 182, 0], [255, 102, 178], [49, 115, 215], [154, 92, 251]], 
    [[255, 182, 0], [255, 102, 178], [203, 115, 88], [108, 67, 94]], [[77, 77, 255], [42, 144, 255], [255, 105, 180], [255, 221, 186]], 
    [[77, 77, 255], [42, 144, 255], [255, 111, 60], [255, 201, 60]], [[255, 77, 77], [255, 144, 42], [2, 130, 201], [66, 201, 56]], 
    [[255, 77, 77], [255, 144, 42], [111, 60, 255], [201, 60, 255]], [[249, 209, 226], [115, 119, 49], [251, 154, 92], [215, 49, 115]], 
    [[249, 209, 226], [115, 119, 49], [3, 146, 132], [4, 113, 205]], [[255, 105, 180], [255, 221, 186], [108, 67, 94], [203, 115, 88]], 
    [[255, 105, 180], [255, 221, 186], [111, 60, 255], [201, 60, 255]], [[239, 172, 1], [42, 131, 86], [201, 60, 255], [111, 60, 255]], 
    [[49, 115, 215], [154, 92, 251], [255, 201, 60], [255, 111, 60]], [[108, 67, 94], [203, 115, 88], [97, 192, 191], [255, 182, 185]], 
    [[66, 201, 56], [2, 130, 201], [246, 114, 128], [53, 92, 125]], [[3, 146, 132], [4, 113, 205], [113, 205, 4], [146, 132, 3]], 
    [[3, 146, 132], [4, 113, 205], [255, 201, 60], [255, 111, 60]], [[255, 111, 60], [255, 201, 60], [111, 60, 255], [201, 60, 255]], 
    [[255, 111, 60], [255, 201, 60], [60, 155, 201], [60, 255, 111]], [[255, 111, 60], [255, 201, 60], [155, 60, 201], [255, 60, 111]]
]


def swap_question(rows, cols):
    question = \
    f"A {rows} * {cols} board consists of {rows * cols} different coloured tiles. A random state of the board is shown in (A). " + \
    f"The ideal state of the board is shown in (B). A swap consists of selecting any two tiles on the board and switching their positions. " + \
    f"What is the minimum number of swaps required to restore the ideal state of the board from (A)?"
    
    return question


def out_of_position_question(rows, cols, mode=1):
    if mode == 1:
        query = "How many tiles are in their ideal position in state (B)?"
    elif mode == 2:
        query = "How many tiles are out of their ideal position in state (B)?"
        
    question = \
    f"A {rows} * {cols} board consists of {rows * cols} different coloured tiles. The ideal state of the board is shown in (A). " + \
    f"A random state of the board is shown in (B). {query}"
    
    return question


if __name__ == "__main__":

    os.makedirs("data/images/colour_hue")

    data, question_index, num_instances = [], 0, 100
    progress_bar = tqdm(range(num_instances))
    solution_set = []

    size = (1024, 1024)
    grid_sizes = [4, 5, 6]

    num_instances_1d = num_instances // 5
    num_instances_2d = num_instances - num_instances_1d
    selected_colors_1d = random.sample(colors_1d, num_instances_1d)


    while question_index < num_instances_1d:
        cols = random.choice(grid_sizes)
        corner1, corner2 = selected_colors_1d[question_index]
        image, full_grid, _ = create_constrained_grid_1d(corner1, corner2, cols=cols, size=size)
        shuffled_image, shuffled_grid, swaps, out_of_position, order = shuffle(full_grid, size, cols)

        while out_of_position == 0:
            shuffled_image, shuffled_grid, swaps, out_of_position, order = shuffle(full_grid, size, cols)
        
        question, answer = swap_question(1, cols), swaps

        solution = {
            "ideal_grid": full_grid.tolist(), "shuffled_grid": shuffled_grid.tolist(),
            "shuffled_order": order, "swaps": swaps, 
            "in_position": cols - out_of_position, "out_of_position": out_of_position
        }

        # Ensure no repetition of data
        if solution not in solution_set:
            solution_set.append(solution)

            instance_image = final_image(shuffled_image, image)
            fname = f"data/images/colour_hue/colour_hue_{question_index:04}.jpg"
            instance_image.save(fname, dpi=(300, 300))
            example = {
                "image": fname, "question": question, "answer": answer,
                "solution": solution
            }

            data.append(example)
            question_index += 1
            progress_bar.update(1)


    size = (1024, 1024)
    selected_colors_2d = colors_2d + random.sample(colors_2d, num_instances_2d - len(colors_2d))

    answer1, answer2, answer3 = [], [], []

    while question_index < num_instances:
        rows, cols = random.choice(grid_sizes), random.choice(grid_sizes)
        cells = rows * cols
        corner1, corner2, corner3, corner4 = selected_colors_2d[question_index - num_instances_1d]
        image, full_grid =  create_constrained_grid_2d(corner1, corner2, corner3, corner4, rows, cols, size)

        to_shuffle = random.choice(list(range(3, 12)))
        
        shuffled_image, shuffled_grid, swaps, out_of_position, order = shuffle(full_grid, size, to_shuffle)
        
        while out_of_position == 0:
            shuffled_image, shuffled_grid, swaps, out_of_position, order = shuffle(full_grid, size, to_shuffle)

        question, answer = swap_question(rows, cols), swaps

        solution = {
            "ideal_grid": full_grid.tolist(), "shuffled_grid": shuffled_grid.tolist(),
            "shuffled_order": order, "swaps": swaps, 
            "in_position": cells - out_of_position, "out_of_position": out_of_position
        }

        # Ensure no repetition of data
        if solution not in solution_set:
            solution_set.append(solution)
            
            instance_image = final_image(shuffled_image, image)
            fname = f"data/images/colour_hue/colour_hue_{question_index:04}.jpg"
            instance_image.save(fname, dpi=(300, 300))
            example = {
                "image": fname, "question": question, "answer": answer,
                "solution": solution
            }

            data.append(example)
            question_index += 1
            progress_bar.update(1)


    with open("data/colour_hue.json", "w") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")
