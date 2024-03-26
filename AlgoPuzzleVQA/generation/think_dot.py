import os
import copy
import json
import random
import itertools
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageDraw
from num2words import num2words


mapper = {"left": 0, "center": 1, "right": 2}

inv_mapper = {0: "left", 1: "center", 2: "right"}

color_mapper = {"b": "blue", "y": "yellow"}

image_mapper = {
    "01": [["b", "b", "b"], ["b", "b"], ["b", "b", "b"]],
    "02": [["y", "y", "y"], ["y", "y"], ["y", "y", "y"]],
    "03": [["y", "b", "y"], ["b", "b"], ["y", "b", "y"]],
    "04": [["b", "y", "b"], ["y", "y"], ["b", "y", "b"]],
    "05": [["y", "y", "y"], ["b", "b"], ["y", "y", "y"]],
    "06": [["b", "b", "b"], ["y", "y"], ["b", "b", "b"]],
}


def flip(state):
    if state == "y":
        return "b"
    else:
        return "y"


def process(start, ball_through):
    current = copy.deepcopy(start)
    
    index1 = mapper[ball_through]

    # First level
    if (index1 == 0 and current[0][index1] == "y") or (index1 == 2 and current[0][index1] == "b"):
        second_level = False
        index3 = index1
    else:
        second_level = True

    # Second level
    if second_level:
        if index1 == 0:
            index2 = 0
        elif index1 == 1:
            if current[0][index1] == "b":
                index2 = 0
            else:
                index2 = 1
        elif index1 == 2:
            index2 = 1

        # Third level
        if index2 == 0:
            if current[1][index2] == "b":
                index3 = 0
            else:
                index3 = 1
        elif index2 == 1:
            if current[1][index2] == "b":
                index3 = 1
            else:
                index3 = 2
            

    current[0][index1] = flip(current[0][index1])
    if second_level:
        current[1][index2] = flip(current[1][index2])
    current[2][index3] = flip(current[2][index3])

    return current


def verbalize(current):
    out = []
    for row in current:
        new_row = []
        for color in row:
            new_row.append(color_mapper[color])
        out.append(new_row)
    return out


def state_after_moves(start, moves):
    current = copy.deepcopy(start)
    history = [verbalize(start)]
    for move in moves:
        current = process(current, move)
        history.append(verbalize(current))
    return current, history


def optimal_moves(start, end):
    current = copy.deepcopy(start)
    moves = [0, 0, 0]

    # Row 0
    for k in range(3):
        if current[0][k] != end[0][k]:
            moves[k] += 1
            current = process(current, inv_mapper[k])
    
    # Row 1
    if current[1][0] != end[1][0] and current[1][1] == end[1][1]:
        moves[0] += 2
        for _ in range(2):
            current = process(current, inv_mapper[0])
            
    elif current[1][0] != end[1][0] and current[1][1] != end[1][1]:
        moves[1] += 2
        for _ in range(2):
            current = process(current, inv_mapper[1])

    elif current[1][0] == end[1][0] and current[1][1] != end[1][1]:
        moves[2] += 2
        for _ in range(2):
            current = process(current, inv_mapper[2])

    # Row 2
    if current[2][0] != end[2][0] and current[2][1] != end[2][1] and current[2][2] == end[2][2]:
        moves[0] += 4
        for _ in range(4):
            current = process(current, inv_mapper[0])
             
    elif current[2][0] != end[2][0] and current[2][1] == end[2][1] and current[2][2] != end[2][2]:
        moves[1] += 4
        for _ in range(4):
            current = process(current, inv_mapper[1])
            
    elif current[2][0] == end[2][0] and current[2][1] != end[2][1] and current[2][2] != end[2][2]:
        moves[2] += 4
        for _ in range(4):
            current = process(current, inv_mapper[2])
            
    elif current[2][0] == end[2][0] and current[2][1] == end[2][1] and current[2][2] == end[2][2]:
        pass
        
    else:
        return "Impossible to solve"
        
    assert current == end

    best_moves = moves[:]
    for _ in range(3):
        moves = [(move + 2) % 8 for move in moves]
        if sum(moves) < sum(best_moves):
            best_moves = moves[:]

    return best_moves


game_context = "The toy shown in the figure has eight coloured disks on its front, and three holes on its top – left, " + \
"right, and center – through which a ball bearing could be dropped. Each disk would display either a yellow or blue " + \
"face. When a ball passes through a disc it tips the disk mechanism which flips the face color. The tipping of the disc " + \
"mechanism determines whether the ball would be deflected to the left or to the right. The vertical walls between the " + \
"discs would then determine the path of motion of the ball. A dropped ball always passes through exactly one disc in " + \
"each of the top and the bottom row. Depending on the configuration of the top three discs it may or may not pass " + \
"through the middle row. Finally, when the ball falls to the bottom it would exit either to a hole on the left or the " + \
"right of the device."


if __name__ == "__main__":

    os.makedirs("data/images/think_dot")

    choices = ["left", "right", "center"]

    data, question_index, num_instances = [], 0, 100
    progress_bar = tqdm(range(num_instances))

    for setting in ["01", "02", "03", "04", "05", "06"]:
        img = Image.open(f"templates/think_dot/{setting}.png").convert("RGB") #.crop((750, 170, 2610, 2100))
        draw = ImageDraw.Draw(img)
        draw.rectangle([(25, 350), (225, 550)], outline="white", fill=(255, 255, 255))
        draw.rectangle([(1625, 350), (1825, 550)], outline="white", fill=(255, 255, 255))
        
        moves = []
        num_samples = [3, 6, 5, 3]
        for num_moves in range(1, 5):
            if num_moves == 1:
                pool = [("left", ), ("right", ), ("center", )]
            else:
                pool = list(itertools.combinations_with_replacement(choices, num_moves))
            moves += random.sample(pool, num_samples[num_moves - 1])
        
        for move in moves:
            if len(move) == 1:
                context = f"One ball in dropped through the {move[0]} hole."
                prefix = ["the ball has", "it has"]
            else:
                balls = f"{num2words(len(move)).capitalize()} balls"
                context = f"{balls} are dropped in sequence through the following holes: {', '.join(move)}."
                prefix = ["all the balls have", "they have"]
                
            start = image_mapper[setting]
            current, history = state_after_moves(start, move)
            
            color = random.choice(["b", "y"])
            x = random.uniform(0, 1)
            if x < 0.2:
                query = f"How many {color_mapper[color]} faces can be seen in the top row now?"
                answer = current[0].count(color)
            elif x < 0.4:
                query = f"How many {color_mapper[color]} faces can be seen in the middle row now?"
                answer = current[1].count(color)
            elif x < 0.6:
                query = f"How many {color_mapper[color]} faces can be seen in the bottom row now?"
                answer = current[2].count(color)
            else:
                query = f"How many {color_mapper[color]} faces can be seen in total in all the rows now?"
                answer = (current[0] + current[1] + current[2]).count(color)

            question = f"{game_context} {context} Consider the toy configuration after {prefix[0]} been dropped " + \
                       f"and {prefix[1]} exited from the bottom. {query}"
            
            if question_index < num_instances:
                fname = f"data/images/think_dot/think_dot_{question_index:04}.jpg"
                img.save(fname, dpi=(300, 300))
                example = {
                    "image": fname[5:], "question": question, "answer": answer,
                    "solution": {
                        "start_state": history[0],
                        "moves": list(move),
                        "state_after_moves": history[1:],
                        "final_state": history[-1]
                    }
                }

                data.append(example)
                question_index += 1
                progress_bar.update(1)
                

    with open("data/think_dot.json", "w") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")

