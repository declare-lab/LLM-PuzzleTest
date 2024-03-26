import os
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import deque
from PIL import Image, ImageDraw, ImageFont
from utils.state import State


def compute_next_states(n, root, total_moves):
    
    current = State(n, root, None, None, 0, 0)
    current_moves = 0

    queue = deque([(current, "Start")])
    while current_moves < total_moves:
        current_moves += 1
        for k in range(len(queue)):
            item = queue.popleft()
            current, direction = item[0], item[1]
            next = current.expand(n)
            for position in next:
                queue.append((position, f"{direction}-{position.direction}"))

    seen, all_current_states, paths = [], [], {}
    for item, move in queue:
        if item.state not in seen:
            seen.append(item.state)
        all_current_states.append(np.array(item.state).reshape(1, n, n))

        move_sequence = move[6:].replace("-", ", ").lower()
        paths[move_sequence] = all_current_states[-1][0]

    assert len(set([tuple(item.flatten().tolist()) for item in paths.values()])) == len(seen)

    all_current_states = np.concatenate(all_current_states, 0)
    return all_current_states, len(seen), paths


def potential_stats(next_states, count_threshold=2):
    # sum across the rows
    row_sums = np.sum(next_states, 2)

    # sum across the columns
    col_sums = np.sum(next_states, 1)
    
    stats = []
    
    for j in [0, n-1]:
        if j == 0:
            row_indicator = "top most row"
            column_indicator = "left most column"
        else:
            row_indicator = "bottom most row"
            column_indicator = "right most column"
            
        row_array = list(row_sums[:, j])
        min_row, max_row = min(row_array), max(row_array)
        if row_array.count(min_row) <= count_threshold:
            stats.append(["minimum sum", row_indicator, min_row, row_array.index(min_row)])
        if row_array.count(max_row) <= count_threshold:
            stats.append(["maximum sum", row_indicator, max_row, row_array.index(max_row)])
        
        col_array = list(col_sums[:, j])
        min_col, max_col = min(col_array), max(col_array)
    
        if col_array.count(min_col) <= count_threshold:
            stats.append(["minimum sum", column_indicator, min_col, col_array.index(min_col)])
        if col_array.count(max_col) <= count_threshold:
            stats.append(["maximum sum", column_indicator, max_col, col_array.index(max_col)])
    
    return stats


font = ImageFont.truetype("fonts/OpenSans-Regular.ttf", 100)
outer_boundary, inner_boundary, lower = 40, 10, 240

def create_image_from_state(n, state):
    image_dim = 2 * outer_boundary + (n-1) * inner_boundary + n * lower
    result = Image.new('RGB', (image_dim, image_dim), color=(255, 255, 255))
    draw = ImageDraw.Draw(result)

    offset = 30
    draw.rounded_rectangle(((offset, offset), (image_dim - offset, image_dim - offset)), 4, fill="white", outline="red", width=2)
    
    for k in range(n * n):
        row = k // n
        col = k % n
        top_x = outer_boundary + inner_boundary * col + lower * col
        bottom_x = outer_boundary + inner_boundary * col + lower * (col + 1)
    
        top_y = outer_boundary + inner_boundary * row + lower * row
        bottom_y = outer_boundary + inner_boundary * row + lower * (row + 1)
    
        if state[k] != 0:
            draw.rounded_rectangle(((top_x, top_y), (bottom_x, bottom_y)), 20, fill="white", outline="black")
            if state[k] < 10:
                draw.text(
                    ((top_x + bottom_x)//2 - 0.5 * outer_boundary, (top_y + bottom_y)//2  - 1.75 * outer_boundary), f"{state[k]}", 
                    (0, 0, 0), font=font
                )
            else:
                draw.text(
                    ((top_x + bottom_x)//2 - 1.5 * outer_boundary, (top_y + bottom_y)//2  - 1.75 * outer_boundary), f"{state[k]}", 
                    (0, 0, 0), font=font
                )
    return result


def create_question(n, question_part):
    context = f"The board shown in the image is a sliding puzzle of {n} * {n} tile dimensions. " + \
    f"It has {n*n -1} numbered tiles and one unoccupied (open) position. Tiles in the same row or column of the open position " + \
    f"can be moved by sliding them horizontally or vertically, respectively. All tiles always stay and move inside the red " + \
    f"boundary wall, as shown in the image. A move is defined as moving the open position by one tile unit in any available direction."

    return f"{context} {question_part}"


if __name__ == "__main__":

    os.makedirs("data/images/number_slide")

    data, question_index, num_instances = [], 0, 100
    progress_bar = tqdm(range(num_instances))
    solution_set = []

    while question_index < num_instances:

        x = random.uniform(0, 1)
        n = random.choices(population=[3, 4, 5], weights=[0.25, 0.45, 0.3], k=1)[0]

        # number of unique positins after n moves
        if x < 0.25:
            mode = 1
            total_moves = random.choices(population=[1, 2, 3], weights=[0.25, 0.45, 0.3], k=1)[0]

        # maximum / minimum sum after n moves
        elif x < 0.6:
            mode = 2
            total_moves = random.choices(population=[1, 2, 3, 4], weights=[0.2, 0.33, 0.33, 0.16], k=1)[0]

        # position / sum after a sequence of moves
        else:
            mode = 3
            total_moves = random.choices(population=[2, 3, 4], weights=[0.25, 0.35, 0.4], k=1)[0]
            

        root = list(range(n * n))
        random.shuffle(root)
        instance_image = create_image_from_state(n, root)
        next_states, num_configs, paths = compute_next_states(n, root, total_moves)
        stats = potential_stats(next_states)

        if total_moves == 1:
            identifier = "move"
        else:
            identifier = "moves"

        if mode == 1:
            question_part = f"You start from the board position shown in the image and perform exactly {total_moves} {identifier}. " + \
                            f"How many unique final board positions can you reach?"
            answer = num_configs
            # submode = "x"
        
        elif mode == 2:
            if len(stats) == 0:
                stats = potential_stats(next_states, count_threshold=100)
            chosen_stat = random.choice(stats)

            question_part = f"You start from the board position shown in the image and perform exactly {total_moves} {identifier}. " + \
                            f"What is the {chosen_stat[0]} that you can achieve across the {chosen_stat[1]} in the final board position?"
            answer = chosen_stat[2]
            # submode = "x"

        else:
            move_sequence = random.choice(list(paths.keys()))
            question_prefix = f"You start from the board position shown in the image and perform exactly {total_moves} {identifier} such " + \
                              f"that the open position is seen moving in the following sequence: {move_sequence}."

            current = paths[move_sequence]
            row_index, col_index = np.argwhere(current == 0)[0]
            empty_tile_row, empty_tile_col = current[row_index, :], current[:, col_index]

            submode = random.choice(list(range(6)))
            if submode == 0:
                question_part = f"{question_prefix} What is the sum of numbers of the row which now has the open position?"
                answer = sum(empty_tile_row)
            
            elif submode == 1:
                question_part = f"{question_prefix} What is the sum of numbers of the column which now has the open position?"
                answer = sum(empty_tile_col)
            
            elif submode == 2:
                question_part = f"{question_prefix} What is the maximum number in the row which now has the open position?"
                answer = max(empty_tile_row)
            
            elif submode == 3:
                question_part = f"{question_prefix} What is the maximum number in the column which now has the open position?"
                answer = max(empty_tile_col)
            
            elif submode == 4:
                question_part = f"{question_prefix} What is the minimum number in the row which now has the open position?"
                answer = sorted(empty_tile_row)[1]
            
            elif submode == 5:
                question_part = f"{question_prefix} What is the minimum number in the column which now has the open position?"
                answer = sorted(empty_tile_col)[1]
            

        question = create_question(n, question_part)

        paths = {key: val.tolist() for key, val in paths.items()}
        solution = {
            "starting_grid": np.array(root).reshape(n, n).tolist(), "total_moves": int(total_moves),
            "path_track": paths, "possible_states": next_states.tolist(), 
            "unique_positions": num_configs
        }
        
        if solution not in solution_set:
            solution_set.append(solution)
            fname = f"data/images/number_slide/number_slide_{question_index:04}.jpg"
            instance_image.save(fname, dpi=(300, 300))
            
            example = {
                "image": fname[5:], "question": question, "answer": int(answer),
                "solution": solution
            }

            data.append(example)
            question_index += 1
            progress_bar.update(1)
            

    with open("data/number_slide.json", "w") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")

