import os
import io
import json
import random
import base64
import numpy as np
from tqdm import tqdm
from PIL import Image
from copy import deepcopy
from collections import deque


class MoveBoxPattern:
    image_size: int = 512
    path_font: str = "fonts/OpenSans-Light.ttf"

    box = "templates/move_box/box.png"
    brick = "templates/move_box/brick.png"
    end = "templates/move_box/end.png"
    person = "templates/move_box/person.png"

    def calculate_answer(self, grid) -> int:
        # this loop is to get the coordinates of target, box and person. Nothing else is gained here
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == "T":
                    target = (i, j)
                if grid[i][j] == "B":
                    box = (i, j)
                if grid[i][j] == "S":
                    person = (i, j)

        # this function checks whether the given coordinates/indices are valid to go
        def valid(x, y):
            return 0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[x][y] != "#"

        # this function checks whether the person can travel from current position to the destination position.
        # used simple bfs(dfs can also be used here), should be self explainatory if you know BFS.
        def check(curr, dest, _box):
            que = deque([curr])
            v = set()
            while que:
                pos = que.popleft()
                if pos == dest:
                    return True
                new_pos = [
                    (pos[0] + 1, pos[1]),
                    (pos[0] - 1, pos[1]),
                    (pos[0], pos[1] + 1),
                    (pos[0], pos[1] - 1),
                ]
                for x, y in new_pos:
                    if valid(x, y) and (x, y) not in v and (x, y) != _box:
                        v.add((x, y))
                        que.append((x, y))
            return False

        clean_grid = deepcopy(grid)

        clean_grid[box[0]][box[1]] = ""
        clean_grid[person[0]][person[1]] = ""

        q = deque([(0, box, person, [box], [grid])])
        vis = {box + person}

        solution = {
            "ids": {
                "empty": ".",
                "brick": "#",
                "box": "B",
                "person": "S",
                "target": "T",
            },
            "timestep": {},
            "target": target,
            "box_location": {},
        }

        # this is the main bfs which gives us the answer
        while q:
            dist, box, person, box_history, grid_history = q.popleft()

            if box == target:  # return the distance if box is at the target
                solution["box_location"] = box_history

                for i, timestep in enumerate(grid_history):
                    solution["timestep"][f"grid_{i}"] = timestep

                # noinspection PyTypeChecker
                return dist, solution

            # these are the new possible coordinates/indices box can be placed in (up, down, right, left).
            b_coord = [
                (box[0] + 1, box[1]),
                (box[0] - 1, box[1]),
                (box[0], box[1] + 1),
                (box[0], box[1] - 1),
            ]
            # these are the corresponding coordinates the person has to be in to push .. the box into the new coordinates
            p_coord = [
                (box[0] - 1, box[1]),
                (box[0] + 1, box[1]),
                (box[0], box[1] - 1),
                (box[0], box[1] + 1),
            ]

            for new_box, new_person in zip(b_coord, p_coord):
                # we check if the new box coordinates are valid and our current state is not in vis
                if valid(*new_box) and new_box + box not in vis:
                    # we check corresponding person coordinates are valid and if it is possible for the person to reach the new coordinates
                    if valid(*new_person) and check(person, new_person, box):

                        new_grid = deepcopy(clean_grid)
                        new_grid[new_box[0]][new_box[1]] = "B"
                        new_grid[box[0]][box[1]] = "S"

                        vis.add(new_box + box)
                        q.append(
                            (
                                dist + 1,
                                new_box,
                                box,
                                box_history + [new_box],
                                grid_history + [new_grid],
                            )
                        )

        return -1, None

    def generate_non_overlapping_coordinates(self, num_points, x_range, y_range):
        coordinates = []

        for _ in range(num_points):
            while True:
                x = np.random.randint(*x_range)
                y = np.random.randint(*y_range)

                # Check for overlap with existing coordinates
                overlap = False
                for existing_x, existing_y in coordinates:
                    if abs(x - existing_x) < 1.0 and abs(y - existing_y) < 1.0:
                        overlap = True
                        break

                if not overlap:
                    coordinates.append((x, y))
                    break

        return coordinates

    def make_sample(self):
        size = self.image_size
        buffer = 30
        num_rows = 6
        num_cols = 6

        length = size // num_cols
        height = size // num_rows

        object_size = length
        box = Image.open(self.box).resize((length, height))
        brick = Image.open(self.brick).resize((length, height))
        end = Image.open(self.end).resize((length, height))
        person = Image.open(self.person).resize((length, height))
        answer = random.randint(1, 5)

        while True:
            image = Image.new(
                "RGB", (size + buffer * 2, size + buffer * 2), color=(255, 255, 255)
            )

            matrix = np.array([["#" for _ in range(num_cols)] for _ in range(num_rows)])

            # Include empty spaces, box, person, endpoint
            num_replaced = random.randint(10, 15)
            empty_spaces_coords = self.generate_non_overlapping_coordinates(
                num_replaced, (1, matrix.shape[0] - 1), (1, matrix.shape[1] - 1)
            )
            for i, (x, y) in enumerate(empty_spaces_coords):
                if i == 0:
                    matrix[x, y] = "B"
                elif i == 1:
                    matrix[x, y] = "S"
                elif i == 2:
                    matrix[x, y] = "T"
                else:
                    matrix[x, y] = "."

            for i, row in enumerate(matrix):
                for j, val in enumerate(row):
                    position = (
                        buffer + (j * length) + (length // 2) - object_size // 2,
                        buffer + (i * height) + (height // 2) - object_size // 2,
                    )

                    assert val in ["#", "B", "S", "T", "."]
                    # empty grid
                    if val == ".":
                        continue
                    # grid with brick
                    elif val == "#":
                        image.paste(brick, position, brick)
                    # grid with box
                    elif val == "B":
                        image.paste(box, position, box)
                    # grid with person
                    elif val == "S":
                        image.paste(person, position, person)
                    # grid with end point
                    elif val == "T":
                        image.paste(end, position, end)

            calculated_answer, calculated_solution = self.calculate_answer(
                matrix.tolist()
            )
            if calculated_answer == answer:
                solution = calculated_solution
                break

        return (
            dict(
                question="A storekeeper is a puzzle in which the player pushes boxes around in a warehouse trying to get them to target locations. The game is represented by an 6 x 6 grid of characters grid where each element is a wall, floor, or box. Your task is to move the box to the end flag under the following rules:\n\n1. The box can be moved to an adjacent free cell by standing next to the box and then moving in the direction of the box by 1 grid. This is a push.\n2. The player cannot walk through the box.\n\nWhat is the minimum number of pushes to move the box to the end flag.",
                answer=str(answer),
                solution=solution,
            ),
            image,
        )


def convert_image_to_text(image: Image) -> str:
    # This is also how OpenAI encodes images: https://platform.openai.com/docs/guides/vision
    with io.BytesIO() as output:
        image.save(output, format="PNG")
        data = output.getvalue()
    return base64.b64encode(data).decode("utf-8")


if __name__ == "__main__":

    random.seed(0)
    np.random.seed(0)

    os.makedirs(f"data/images/move_box", exist_ok=True)

    data, question_index, num_instance = [], 0, 100
    seen = []

    progress = tqdm(range(num_instance))

    pattern = MoveBoxPattern()

    while len(data) < num_instance:
        sample, image = pattern.make_sample()

        sample_check = deepcopy(sample)
        image_string = convert_image_to_text(image)
        sample_check["image_string"] = image_string

        if sample_check not in seen:
            seen.append(sample_check)

            fname = f"data/images/move_box/move_box_{question_index:04}.jpg"
            image.save(fname)

            sample = list(sample.items())
            sample.insert(0, ("image", fname[5:]))
            sample = dict(sample)

            question_index += 1
            data.append(sample)
            progress.update()

    with open(f"data/move_box.json", "w") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")
