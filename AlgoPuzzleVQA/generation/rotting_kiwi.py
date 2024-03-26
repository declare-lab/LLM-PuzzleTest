import os
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

from collections import deque
from copy import deepcopy

import io
import base64


class RottingKiwiPattern:
    image_size: int = 512
    path_font: str = "fonts/OpenSans-Light.ttf"

    kiwi = "templates/rotting_kiwi/kiwi.png"
    cross = "templates/rotting_kiwi/cross.png"

    def calculate_answer(self, grid) -> int:
        # number of rows
        rows = len(grid)
        if rows == 0:  # check if grid is empty
            return -1

        # number of columns
        cols = len(grid[0])

        # keep track of fresh oranges
        fresh_cnt = 0

        # queue with rotten oranges (for BFS)
        rotten = deque()

        # visit each cell in the grid
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 2:
                    # add the rotten orange coordinates to the queue
                    rotten.append((r, c))
                elif grid[r][c] == 1:
                    # update fresh oranges count
                    fresh_cnt += 1

        # keep track of minutes passed.
        minutes_passed = 0

        solution = {
            "ids": {"empty": 0, "fresh": 1, "rotten": 2},
            "timestep": {"minute_0": deepcopy(grid.tolist())},
        }

        # If there are rotten oranges in the queue and there are still fresh oranges in the grid keep looping
        while rotten and fresh_cnt > 0:
            # update the number of minutes passed
            # it is safe to update the minutes by 1, since we visit oranges level by level in BFS traversal.
            minutes_passed += 1

            # process rotten oranges on the current level
            for _ in range(len(rotten)):
                x, y = rotten.popleft()

                # visit all the adjacent cells
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    # calculate the coordinates of the adjacent cell
                    xx, yy = x + dx, y + dy
                    # ignore the cell if it is out of the grid boundary
                    if xx < 0 or xx == rows or yy < 0 or yy == cols:
                        continue
                    # ignore the cell if it is empty '0' or visited before '2'
                    if grid[xx][yy] == 0 or grid[xx][yy] == 2:
                        continue

                    # update the fresh oranges count
                    fresh_cnt -= 1

                    # mark the current fresh orange as rotten
                    grid[xx][yy] = 2

                    # add the current rotten to the queue
                    rotten.append((xx, yy))
                    # noinspection PyUnresolvedReferences
                    solution["timestep"][f"minute_{minutes_passed}"] = deepcopy(
                        grid.tolist()
                    )

        # return the number of minutes taken to make all the fresh oranges to be rotten
        # return -1 if there are fresh oranges left in the grid (there were no adjacent rotten oranges to make them rotten)
        answer = minutes_passed if fresh_cnt == 0 else -1

        # noinspection PyTypeChecker
        return answer, solution

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
        num_rows = 3
        num_cols = 3

        object_size = 125
        kiwi = Image.open(self.kiwi).resize((object_size, object_size))
        cross = Image.open(self.cross).resize((object_size, object_size))
        answer = random.randint(1, 6)

        while True:
            image = Image.new("RGB", (size + buffer * 2, size + buffer * 2), "white")

            # Get the drawing context
            draw = ImageDraw.Draw(image)

            length = size // num_cols
            height = size // num_rows

            # Draw the grid
            for x in range(buffer, size + buffer + 1, length):
                draw.line((x, buffer, x, size + buffer), fill="black", width=3)

            for y in range(buffer, size + buffer + 1, height):
                draw.line((buffer, y, size + buffer, y), fill="black", width=3)

            matrix = np.zeros((num_rows, num_cols)).astype(int)

            num_replaced = random.randint(2, 7)

            # Random (x, y) coordinates
            empty_spaces_coords = self.generate_non_overlapping_coordinates(
                num_replaced, (0, matrix.shape[0]), (0, matrix.shape[0])
            )
            for x, y in empty_spaces_coords[:-1]:
                matrix[x][y] = 1

            matrix[empty_spaces_coords[-1][0]][empty_spaces_coords[-1][1]] = 2

            for i, row in enumerate(matrix):
                for j, val in enumerate(row):
                    position = (
                        buffer + (j * length) + (length // 2) - object_size // 2,
                        buffer + (i * height) + (height // 2) - object_size // 2,
                    )

                    assert val in [0, 1, 2]
                    # empty grid
                    if val == 0:
                        continue
                    # grid with kiwi
                    elif val == 1:
                        image.paste(kiwi, position, kiwi)
                    # grid with rotten kiwi
                    elif val == 2:
                        image.paste(kiwi, position, kiwi)
                        image.paste(cross, position, cross)

            # noinspection PyTypeChecker
            calculated_answer, calculated_solution = self.calculate_answer(matrix)
            if calculated_answer == answer:
                solution = calculated_solution
                break

        return (
            dict(
                question="You are given a 3 x 3 grid in which each cell can contain either no kiwi, one fresh kiwi, or one rotten kiwi. Every minute, any fresh kiwi that is 4-directionally adjacent to a rotten kiwi also becomes rotten. What is the minimum number of minutes that must elapse until no cell has a fresh kiwi?",
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

    os.makedirs(f"data/images/rotting_kiwi", exist_ok=True)

    data, question_index, num_instance = [], 0, 100
    seen = []

    progress = tqdm(range(num_instance))

    pattern = RottingKiwiPattern()

    while len(data) < num_instance:
        sample, image = pattern.make_sample()

        sample_check = deepcopy(sample)
        image_string = convert_image_to_text(image)
        sample_check["image_string"] = image_string

        if sample_check not in seen:
            seen.append(sample_check)

            fname = f"data/images/rotting_kiwi/rotting_kiwi_{question_index:04}.jpg"
            image.save(fname)

            sample = list(sample.items())
            sample.insert(0, ("image", fname[5:]))
            sample = dict(sample)

            question_index += 1
            data.append(sample)
            progress.update()

    with open(f"data/rotting_kiwi.json", "w") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")
