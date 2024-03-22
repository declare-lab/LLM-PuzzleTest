import base64
import io
import os
import itertools
import copy
import json
import math
import random
from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fire import Fire
from pydantic import BaseModel
from tqdm import tqdm

Point = Tuple[float, float]


class CircleSizeNumberPattern(BaseModel):
    image_size: int = 512
    scale_factor: int = 4
    path_font: str = "fonts/OpenSans-Light.ttf"
    color: str = "#eeeeee"

    def make_sample(self):
        # Set the size of the image
        size = self.image_size * self.scale_factor
        image = Image.new("RGB", size=(size, size), color="white")
        draw = ImageDraw.Draw(image)

        numbers = list(random.sample(range(1, 7), 3)) * 2
        random.shuffle(numbers)

        answer_location = random.choice(range(len(numbers)))
        answer = numbers[answer_location]

        center = size // 2
        distance = 150 * self.scale_factor
        for i, number in enumerate(numbers):
            angle = (i / len(numbers)) * 2 * math.pi
            small_circle_x = center + int(distance * math.cos(angle))
            small_circle_y = center + int(distance * math.sin(angle))

            small_circle_radius = 50 * self.scale_factor + 15 * number
            draw.ellipse(
                [
                    (
                        small_circle_x - small_circle_radius,
                        small_circle_y - small_circle_radius,
                    ),
                    (
                        small_circle_x + small_circle_radius,
                        small_circle_y + small_circle_radius,
                    ),
                ],
                fill=self.color,
                outline="black",
                width=4,
            )

            draw.text(
                (
                    small_circle_x,
                    small_circle_y,
                ),
                str(number) if i != answer_location else "?",
                font=ImageFont.truetype(self.path_font, size=50 * self.scale_factor),
                anchor="mm",
                fill="black",
            )

        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)

        options = random.sample(list(set(range(1, 7)) - {answer}), 3)
        options.append(answer)
        random.shuffle(options)

        numbers_max = max(numbers)
        numbers_min = min(numbers)
        numbers[answer_location] = "?"

        return (
            dict(
                question="What is the missing number of the part denoted with a question mark?",
                options=options,
                answer=str(answer),
                caption=f"There are 6 numbered circles with varying sizes arranged in a ring with number {numbers} in a clockwise order.",
                explanation=f"We observe that the size of the circle is related to the number in the circle. The circle with the largest value {numbers_max} seems to be the biggest and the circle with the smallest value {numbers_min} seems to be the smallest. Thus, the pattern is that the larger the number the larger the circle.",
                deduction=f"Based on the pattern that the larger the number the larger the circle, the missing number of the circle denoted with a question mark should be {answer}.",
            ),
            image,
        )


class ColorGridPattern(BaseModel):
    image_size: int = 512
    scale_factor: int = 4
    path_font: str = "fonts/OpenSans-Medium.ttf"
    colors: Dict[str, str] = dict(
        blue="#6fa8dc",
        green="#93c47d",
        yellow="#ffd966",
        red="#e06666",
        purple="#8e7cc3",
        orange="#f6b26b",
    )

    def sample_colors(self) -> Tuple[List[str], List[str]]:
        while True:
            names = random.sample(list(self.colors), k=3)
            if "orange" in names and "yellow" in names:
                continue  # Hard to distinguish
            return names

    def draw_circle(self, draw: ImageDraw, point: Point, radius: int, color: str):
        x, y = point
        position = x - radius, y - radius, x + radius, y + radius
        line_width = self.image_size * self.scale_factor // 200
        draw.ellipse(position, fill=color, outline="black", width=line_width)

    def make_sample(self):
        # Set the size of the image
        size = self.image_size * self.scale_factor
        image = Image.new("RGB", size=(size, size), color="white")
        draw = ImageDraw.Draw(image)
        a, b, c = size // 4, size // 2, size * 3 // 4
        positions = [(x, y) for x in [a, b, c] for y in [a, b, c]]

        names = self.sample_colors()
        values = [[0, 2, 6, 8], [1, 3, 5, 7], [4]]
        mapping = {k: v for k, v in zip(names, values)}
        i_answer = random.choice([0, 2, 6, 8, 1, 3, 5, 7])
        answer = ""

        for k, lst in mapping.items():
            for i in lst:
                if i == i_answer:
                    answer = k
                    draw.text(
                        positions[i],
                        text="?",
                        font=ImageFont.truetype(self.path_font, size=size // 10),
                        anchor="mm",
                        fill="black",
                    )
                else:
                    color = self.colors[k]
                    self.draw_circle(draw, positions[i], radius=size // 10, color=color)

        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        grid = ["?"] * 9
        for k, lst in mapping.items():
            for i in lst:
                if i != i_answer:
                    grid[i] = k
        grid = grid[::-1]
        location = "at the corner" if answer == names[0] else "adjacent to the center"

        return (
            dict(
                question="What is the color of the missing part denoted with a question mark?",
                answer=answer,
                options=sample_options(answer, sorted(self.colors), k=4),
                caption=f"There are circles with different colors arranged with a grid formation in the image. The colors in the first row are {grid[:3]}, the colors in the second row are {grid[3:6]}, and the colors in the third row are {grid[6:9]}.",
                explanation=f"We observe that the circles at the corners are {names[0]}, while the circles directly adjacent to the center are {names[1]}. Only the center circle is {names[2]}. Hence, the pattern is that the circles alternate in color depending on if they are at the corner or adjacent to the center.",
                deduction=f"Based on the pattern that the circles alternate in color depending on if they are at the corner or adjacent to the center, the missing color of the part that is {location} should be {answer}.",
            ),
            image,
        )


class ColorHexagonPattern(BaseModel):
    colors: Dict[str, str] = dict(
        blue="#6fa8dc",
        green="#93c47d",
        yellow="#ffd966",
        red="#e06666",
        purple="#8e7cc3",
        orange="#f6b26b",
    )
    image_size: int = 512
    scale_factor: int = 4
    path_font: str = "fonts/OpenSans-Medium.ttf"

    @staticmethod
    def get_centroid(points: List[Tuple[float, float]]) -> Tuple[float, float]:
        x = sum(p[0] for p in points) / len(points)
        y = sum(p[1] for p in points) / len(points)
        return x, y

    def sample_colors(self) -> Tuple[List[str], List[str]]:
        while True:
            names = random.sample(list(self.colors), k=3)
            if "orange" in names and "yellow" in names:
                continue  # Hard to distinguish
            names = names + names
            colors = [self.colors[n] for n in names]
            return names, colors

    def make_sample(self):
        # Set the size of the image
        size = self.image_size * self.scale_factor
        image = Image.new("RGB", size=(size, size), color="white")
        draw = ImageDraw.Draw(image)
        center = size // 2

        # Hexagon properties
        side_length = size // 3  # Length of a side of the hexagon and triangles
        triangle_height = math.sqrt(3) / 2 * side_length

        # The vertices of the hexagon
        hexagon = [
            (center + side_length / 2, center - triangle_height),
            (center - side_length / 2, center - triangle_height),
            (center - side_length, center),
            (center - side_length / 2, center + triangle_height),
            (center + side_length / 2, center + triangle_height),
            (center + side_length, center),
        ]

        # Colors for the triangles
        names, colors = self.sample_colors()
        i_answer = random.randint(0, len(colors) - 1)
        answer = names[i_answer]
        colors[i_answer] = "#eeeeee"  # Grey

        # Draw the hexagon made of six triangles
        for i in range(6):
            # Coordinates of the triangle vertices
            triangle = [hexagon[i], hexagon[(i + 1) % 6], (center, center)]
            # Draw the triangle
            draw.polygon(triangle, fill=colors[i])
            # Draw the outline with custom width
            points = [hexagon[i], hexagon[(i + 1) % 6], (center, center), hexagon[i]]
            draw.line(points, fill="black", width=self.scale_factor * 4)
            # Draw "?" on the missing answer part
            if i == i_answer:
                draw.text(
                    self.get_centroid(triangle),
                    text="?",
                    font=ImageFont.truetype(self.path_font, size=size // 10),
                    anchor="mm",
                    fill="black",
                )

        names[i_answer] = "?"
        instances = sorted(set(n for n in names if n not in [answer, "?"]))
        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        return (
            dict(
                question="What is the missing color of the part denoted with a question mark?",
                answer=answer,
                options=sample_options(answer, options=list(self.colors), k=4),
                caption=f"There is a hexagon split into six parts with the colors {names} in an anti-clockwise order.",
                explanation=f"We observe that a {instances[0]} part is opposite another {instances[0]} part, and a {instances[1]} part is opposite another {instances[1]} part. Thus, the pattern is that the colors in opposite parts are the same.",
                deduction=f"Based on the pattern that spatially opposite parts have the same color, the missing color of the part which is opposite a {answer} part should be {answer}.",
            ),
            image,
        )


class ColorNumberHexagonPattern(BaseModel):
    colors: Dict[str, str] = dict(
        blue="#9ec5e8",
        green="#b6d7a8",
        yellow="#fee599",
        red="#ea9999",
        purple="#b4a7d6",
        orange="#f9cb9c",
    )
    numbers: List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    image_size: int = 512
    scale_factor: int = 4
    path_font: str = "fonts/OpenSans-Medium.ttf"

    @staticmethod
    def get_centroid(points: List[Tuple[float, float]]) -> Tuple[float, float]:
        x = sum(p[0] for p in points) / len(points)
        y = sum(p[1] for p in points) / len(points)
        return x, y

    def sample_colors(self) -> Tuple[List[str], List[str]]:
        while True:
            names = random.sample(list(self.colors), k=3)
            if "orange" in names and "yellow" in names:
                continue  # Hard to distinguish
            names = names + names
            colors = [self.colors[n] for n in names]
            return names, colors

    def make_sample(self):
        # Set the size of the image
        size = self.image_size * self.scale_factor
        image = Image.new("RGB", size=(size, size), color="white")
        draw = ImageDraw.Draw(image)
        center = size // 2

        # Hexagon properties
        side_length = size // 3  # Length of a side of the hexagon and triangles
        triangle_height = math.sqrt(3) / 2 * side_length

        # The vertices of the hexagon
        hexagon = [
            (center + side_length / 2, center - triangle_height),
            (center - side_length / 2, center - triangle_height),
            (center - side_length, center),
            (center - side_length / 2, center + triangle_height),
            (center + side_length / 2, center + triangle_height),
            (center + side_length, center),
        ]

        # Colors for the triangles
        numbers = random.sample(self.numbers, k=3)
        min_value = max(numbers) + 1
        max_value = min(numbers) + 9
        value = random.randint(min_value, max_value)
        numbers.extend([value - n for n in numbers])
        assert all(1 <= num <= 9 for num in numbers)
        names, colors = self.sample_colors()
        i_answer = random.randint(0, len(numbers) - 1)

        indices = random.sample(list(range(6)), k=6)
        numbers = [numbers[i] for i in indices]
        names = [names[i] for i in indices]
        colors = [colors[i] for i in indices]
        answer = numbers[i_answer]

        # Draw the hexagon made of six triangles
        for i in range(6):
            # Coordinates of the triangle vertices
            triangle = [hexagon[i], hexagon[(i + 1) % 6], (center, center)]
            # Draw the triangle
            draw.polygon(triangle, fill=colors[i])
            # Draw the outline with custom width
            points = [hexagon[i], hexagon[(i + 1) % 6], (center, center), hexagon[i]]
            draw.line(points, fill="black", width=self.scale_factor * 4)
            # Add number or "?" for missing part
            draw.text(
                self.get_centroid(triangle),
                text="?" if i == i_answer else str(numbers[i]),
                font=ImageFont.truetype(self.path_font, size=size // 10),
                anchor="mm",
                fill="black",
            )

        numbers[i_answer] = "?"
        instances = sorted(set(n for n in names if n not in [answer, "?"]))
        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)

        return (
            dict(
                question="What is the missing number of the part denoted with a question mark?",
                answer=answer,
                options=[str(o) for o in generate_number_options(answer, k=4)],
                caption=f"There is a hexagon split into six parts with the colors {names} in an anti-clockwise order. The parts are denoted with the numbers {numbers} respectively.",
                explanation=f"We observe that the numbers in the {instances[0]} parts add up to {value}. Similarly, the numbers in the {instances[1]} parts also add up to {value}. Thus, the pattern is that the numbers in the parts of the same color add up to {value}.",
                deduction=f"Based on the pattern that the numbers in the parts of the same color add up to {value}, the missing number of the {names[i_answer]} part should be {answer}.",
            ),
            image,
        )


class ColorOverlapSquaresPattern(BaseModel):
    colors: Dict[str, str] = dict(
        blue="#6fa8dc",
        green="#93c47d",
        yellow="#ffd966",
        red="#e06666",
        purple="#8e7cc3",
        orange="#f6b26b",
    )
    numbers: List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    image_size: int = 512
    scale_factor: int = 4
    path_font: str = "fonts/OpenSans-Medium.ttf"
    num_sides: int = 4

    def get_points(self, center: Point, radius: float, angle: int = 0) -> List[Point]:
        vertices = []
        for i in range(self.num_sides):
            theta = 2 * math.pi / self.num_sides * i
            theta -= math.pi / 4  # Adjust to flat rectangle by default
            theta -= math.radians(angle)
            x = center[0] + radius * math.cos(theta)
            y = center[1] + radius * math.sin(theta)
            vertices.append((x, y))
        return vertices

    def draw_squares(self, draw: ImageDraw, colors: List[str]):
        size = self.image_size * self.scale_factor
        line_width = size // 150

        # Center big square
        a = self.get_points((size / 2, size / 2), radius=size / 4)
        draw.polygon(a, outline="black", fill=colors[1], width=line_width)

        # Top right rotated square
        b = self.get_points(a[0], radius=size / 4, angle=45)
        draw.polygon(b, outline="black", fill=colors[2], width=line_width)

        # Bottom left rotated square
        c = self.get_points(a[2], radius=size / 4, angle=45)
        draw.polygon(c, outline="black", fill=colors[0], width=line_width)

        # Top right overlap triangle
        ab = [a[0], b[2], b[3]]
        draw.polygon(ab, outline="black", fill=colors[4], width=line_width)

        # Bottom left overlap triangle
        ac = [a[2], c[0], c[1]]
        draw.polygon(ac, outline="black", fill=colors[3], width=line_width)

    def sample_color_names(self) -> Tuple[str, str, str, str, str]:
        a, b, c = random.sample(["red", "yellow", "blue"], k=3)
        mapping = dict(redyellow="orange", blueyellow="green", bluered="purple")
        d = mapping["".join(sorted([a, b]))]
        e = mapping["".join(sorted([b, c]))]
        assert [x in self.colors for x in [a, b, c, d, e]]
        return a, b, c, d, e

    def make_sample(self):
        # Set the size of the image
        size = self.image_size * self.scale_factor
        image = Image.new("RGB", size=(size, size), color="white")
        draw = ImageDraw.Draw(image)

        color_names = self.sample_color_names()
        colors = [self.colors[n] for n in color_names]

        if random.random() > 0.5:
            answer = color_names[0]
            colors[0] = "#eeeeee"  # Grey
            position = (size // 4, size * 3 // 4)
        else:
            answer = color_names[2]
            colors[2] = "#eeeeee"  # Grey
            position = (size * 3 // 4, size // 4)

        self.draw_squares(draw, colors)
        draw.text(
            position,
            text="?",
            font=ImageFont.truetype(self.path_font, size=size // 10),
            anchor="mm",
            fill="black",
        )

        color_names = [("?" if n == answer else n) for n in color_names]
        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        instances = [color_names[0], color_names[1], color_names[3]]
        overlap = color_names[4]
        if "?" in instances:
            instances = [color_names[1], color_names[2], color_names[4]]
            overlap = color_names[3]

        return (
            dict(
                question="What is the missing color of the part denoted with a question mark?",
                answer=answer,
                options=sample_options(answer, sorted(self.colors), k=4),
                caption=f"There are 3 squares which overlap each other in the image. The color of the squares are {color_names[:3]}. The part where the first and second squares overlap is {color_names[3]}. The part where the second and third squares overlap is {color_names[4]}.",
                explanation=f"We observe that the {instances[0]} and {instances[1]} squares overlap to form {instances[2]}. Hence, the pattern is that the color of the part where two squares overlap is determined by mixing the two colors.",
                deduction=f"Based on the pattern that the color of the part where two squares overlap is determined by mixing the two colors, the missing color of the part which overlaps with {color_names[1]} to form {overlap} should be {answer}.",
            ),
            image,
        )


class ColorSizeCirclePattern(BaseModel):
    image_size: int = 512
    scale_factor: int = 4
    path_font: str = "fonts/OpenSans-Medium.ttf"
    colors: Dict[str, str] = dict(
        blue=["#cfe2f3", "#9fc4e8", "#6fa7dc", "#3d85c5"],
        green=["#d9ead3", "#b6d7a8", "#92c47d", "#69a84f"],
        yellow=["#fff2cc", "#fee599", "#ffd966", "#f0c232"],
        red=["#f4cccc", "#ea9999", "#e06666", "#cc0101"],
        purple=["#d9d2e9", "#b3a7d6", "#8d7cc3", "#664ea6"],
        orange=["#fbe5cd", "#f9ca9c", "#f6b16b", "#e69139"],
    )

    def draw_circle(self, draw: ImageDraw, x: int, y: int, radius: int, **kwargs):
        position = x - radius, y - radius, x + radius, y + radius
        line_width = self.image_size * self.scale_factor // 150
        draw.ellipse(position, width=line_width, **kwargs)

    def make_sample(self):
        # Set the size of the image
        size = self.image_size * self.scale_factor
        image = Image.new("RGB", size=(size, size), color="white")
        draw = ImageDraw.Draw(image)

        key = random.choice(sorted(self.colors))
        colors = self.colors[key]
        assert len(set(colors)) == len(colors)
        names = [f"light {key}", f"medium {key}", f"dark {key}"]
        if random.random() > 0.5:
            colors, names = colors[::-1], names[::-1]

        radii = [size * 0.4, size * 0.3, size * 0.2, size * 0.1]
        for i, r in enumerate(radii):
            x = y = size // 2
            fill = colors[i] if i != len(radii) - 1 else "#eeeeee"  # Grey
            self.draw_circle(draw, x, y, r, fill=fill, outline="black")

        draw.text(
            xy=(size // 2, size // 2),
            text="?",
            font=ImageFont.truetype(self.path_font, size=size // 10),
            anchor="mm",
            fill="black",
        )

        answer = names[-1]
        lst = ["light " + k for k in self.colors] + ["dark " + k for k in self.colors]
        lst.remove(names[0])
        lst.remove(names[-1])
        options = [names[0], names[-1]] + random.sample(lst, k=2)
        assert answer in options
        assert len(set(options)) == len(options)

        if "dark" in answer:
            pairs = [
                ("extra large", f"very light {key}"),
                ("large", f"light {key}"),
                ("medium", f"medium {key}"),
                ("small", "?"),
            ]
            trend = "darker"
        else:
            pairs = [
                ("extra large", f"very dark {key}"),
                ("large", f"dark {key}"),
                ("medium", f"medium {key}"),
                ("small", "?"),
            ]
            trend = "lighter"

        shuffled = random.sample(pairs, k=len(pairs))
        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        return (
            dict(
                question="What is the missing color of the part denoted with a question mark?",
                answer=answer,
                options=random.sample(options, k=len(options)),
                caption=f"There are circles of various sizes and colors in the image. The circles are {[p[0] for p in shuffled]} size, and their colors are {[p[1] for p in shuffled]}.",
                explanation=f"We observe that the largest circle is {pairs[0][1]} color, and the smaller circles change color from {pairs[1][1]} to {pairs[2][1]}. Hence, the pattern is that the circles become {trend} as they become smaller.",
                deduction=f"Based on the pattern that the circles become {trend} as they become smaller, the missing color of the smallest circle denoted with a question mark should be {answer}.",
            ),
            image,
        )


class GridNumberColorPattern(BaseModel):
    colors: Dict[str, str] = dict(
        blue="#6fa8dc",
        green="#93c47d",
        yellow="#ffd966",
        red="#e06666",
        purple="#8e7cc3",
        orange="#f6b26b",
    )
    image_size: int = 512
    scale_factor: int = 4
    path_font: str = "fonts/OpenSans-Light.ttf"

    def make_sample(self):
        size = self.image_size * self.scale_factor
        buffer = 30 * self.scale_factor
        num_rows = 3
        num_cols = 3

        image = Image.new("RGB", (size + buffer * 2, size + buffer * 2), "white")

        # Get the drawing context
        draw = ImageDraw.Draw(image)
        length = size // num_cols
        height = size // num_rows

        # Randomly choose 4 integers that will appear in the matrix
        values = random.sample(range(1, 10), 4)
        answer_number = values[-1]

        num2col = dict(zip(values, random.sample(self.colors.keys(), 4)))
        answer = num2col[answer_number]

        matrix = []

        i = None
        for i in values:
            matrix.append(i)
            matrix.append(i)
        else:
            matrix.append(i)

        random.shuffle(matrix)
        matrix = np.array(matrix).reshape(-1, 3)

        answer_location = random.choice(
            [
                (i, j)
                for i, row in enumerate(matrix)
                for j, number in enumerate(row)
                if number == answer_number
            ]
        )

        # Draw shapes and numbers
        for i, row in enumerate(matrix):
            for j, num in enumerate(row):
                color = (
                    self.colors[num2col[num]]
                    if not (i, j) == answer_location
                    else "#eeeeee"
                )
                number = str(num)

                draw.rounded_rectangle(
                    (
                        buffer + j * length,
                        buffer + i * height,
                        buffer + (j + 1) * length,
                        buffer + (i + 1) * height,
                    ),
                    fill=color,
                    outline="black",
                    width=4,
                    radius=size // 20,
                )

                draw.text(
                    (
                        buffer + (j * length) + (length // 2),
                        buffer + (i * height) + (height // 2),
                    ),
                    number if not (i, j) == answer_location else "?",
                    font=ImageFont.truetype(
                        self.path_font, size=60 * self.scale_factor
                    ),
                    anchor="mm",
                    fill="black",
                )

        options = list(num2col.values())
        random.shuffle(options)
        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)

        caption_details = []

        for i in range(len(matrix)):
            row = []
            for j in range(len(matrix[i])):
                if (i, j) == answer_location:
                    row.append((matrix[i, j], "?"))
                else:
                    row.append((matrix[i, j], num2col[matrix[i, j]]))
            caption_details.append(row)

        return (
            dict(
                question=f"What is the missing color if the part denoted with the question mark has the number {answer_number}?",
                answer=answer,
                options=options,
                caption=f"There is a 3x3 colored grid of numbers. The first row has number-color pair {caption_details[0]}, the second row is {caption_details[1]}, and the third and final row is {caption_details[2]}.",
                explanation=f"We observe that the grid cells with number {values[0]} is {num2col[values[0]]} in color, the grid cells with number {values[1]} is {num2col[values[1]]} in color, the grid cells with number {values[2]} is {num2col[values[2]]} in color, and the grid cells with number {values[3]} is {num2col[values[3]]} in color. Thus, the pattern is that the grid cell with the same number will have the same color.",
                deduction=f"Based on the pattern that the grid cell with the same number will have the same color, the missing color of the part with {answer_number} should be {answer}.",
            ),
            image,
        )


class GridNumberPattern(BaseModel):
    image_size: int = 512
    scale_factor: int = 4
    path_font: str = "fonts/OpenSans-Medium.ttf"
    color: str = "#cfe2f3"  # Light blue

    def draw_text(self, draw: ImageDraw, point: Point, text: str):
        size = self.image_size * self.scale_factor
        draw.text(
            point,
            text=text,
            font=ImageFont.truetype(self.path_font, size=size // 8),
            anchor="mm",
            fill="black",
        )

    def draw_box(self, draw: ImageDraw, point: Point):
        size = self.image_size * self.scale_factor
        width = height = size / 8
        draw.rounded_rectangle(
            [point[0] - width, point[1] - height, point[0] + width, point[1] + height],
            outline="black",
            width=size // 200,
            radius=size // 20,
            fill=self.color,
        )

    def make_sample(self):
        size = self.image_size * self.scale_factor
        image = Image.new("RGB", (size, size), "white")
        draw = ImageDraw.Draw(image)
        values = random.sample(range(1, 10), 6)
        num_rows = 3

        # Adjust the matrix to ensure that the sum of each row and column is the same
        while True:
            matrix = np.random.choice(values, size=(3, 3))
            row_sums = matrix.sum(axis=1)
            desired_sum = row_sums[0]

            if np.all(row_sums == desired_sum) and len(set(matrix.flatten())) >= 4:
                break

        answer_location = np.random.randint(0, num_rows, size=2)
        answer = matrix[answer_location[0]][answer_location[1]]
        matrix = matrix.tolist()
        matrix[answer_location[0]][answer_location[1]] = "?"

        a, b, c = size * 0.25, size * 0.50, size * 0.75
        locations = [
            [(a, a), (b, a), (c, a)],
            [(a, b), (b, b), (c, b)],
            [(a, c), (b, c), (c, c)],
        ]

        for i, row in enumerate(matrix):
            for j, val in enumerate(row):
                self.draw_box(draw, point=locations[i][j])
                self.draw_text(draw, point=locations[i][j], text=str(val))

        values.remove(answer)
        values = values[:3]
        values.append(int(answer))
        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)

        instances = [row for row in matrix if "?" not in row]
        instances.extend([row for row in matrix if "?" in row])
        assert len(instances) == len(matrix)

        return (
            dict(
                question="What is the missing number of the part denoted with a question mark?",
                answer=str(answer),
                options=list(map(str, values)),
                caption=f"There is a 3x3 grid of numbers. The first row is {matrix[0]}. The second row is {matrix[1]}. The third and last row is {matrix[2]}.",
                explanation=f"We observe that {instances[0]} sums to {sum(instances[0])}, and {instances[1]} also sums to {sum(instances[1])}. Thus, the pattern is that the numbers in each row add up to the same value.",
                deduction=f"Based on the pattern that the numbers in each row add up to the same value, the missing number of the row {instances[2]} should be {answer}.",
            ),
            image,
        )


class PolygonSidesColorPattern(BaseModel):
    colors: Dict[str, str] = dict(
        blue="#6fa8dc",
        green="#93c47d",
        yellow="#ffd966",
        red="#e06666",
        purple="#8e7cc3",
        orange="#f6b26b",
    )
    image_size: int = 512
    scale_factor: int = 4
    path_font: str = "fonts/OpenSans-Light.ttf"

    @staticmethod
    def draw_polygon(draw, sides, center, size, color):
        angle = 360 / sides
        points = []

        for i in range(sides):
            x = center[0] + size * math.cos(math.radians(i * angle))
            y = center[1] + size * math.sin(math.radians(i * angle))
            points.append((x, y))

        draw.polygon(points, outline="black", fill=color, width=4)

    def make_sample(self):
        # Set the size of the image
        size = self.image_size * self.scale_factor
        image = Image.new("RGB", size=(size, size), color="white")
        draw = ImageDraw.Draw(image)

        sides = random.sample(range(3, 10), 3)
        side2col = dict(zip(sides, random.sample(self.colors.keys(), 3)))
        sides *= 2
        random.shuffle(sides)
        answer_location = random.choice(range(len(sides)))

        answer = side2col[sides[answer_location]]

        options = set(list(self.colors.keys())) - {answer}
        options = random.sample(options, 3)
        options.append(answer)
        random.shuffle(options)

        center = size // 2
        distance = 175 * self.scale_factor

        for i, side in enumerate(sides):
            polygon_distance = distance - 0.5 * (i % 2) * distance
            angle = (i / len(sides)) * 2 * math.pi
            center_y = center - int(polygon_distance * math.cos(angle))
            center_x = center - int(polygon_distance * math.sin(angle))
            polygon_size = 60 * self.scale_factor

            color = self.colors[side2col[side]] if i != answer_location else "#eeeeee"
            self.draw_polygon(draw, side, (center_x, center_y), polygon_size, color)

            if i == answer_location:
                draw.text(
                    (
                        center_x,
                        center_y,
                    ),
                    "?",
                    font=ImageFont.truetype(
                        self.path_font, size=50 * self.scale_factor
                    ),
                    anchor="mm",
                    fill="black",
                )

        colors = [side2col[side] for side in sides]
        colors[answer_location] = "?"
        top_row = [colors[0]]
        middle_row = [colors[1], colors[5]]
        bottom_row = [colors[2], colors[3], colors[4]]

        explanation_side = list(
            set(
                [
                    side
                    for side in sides
                    if (side != "?" and side != sides[answer_location])
                ]
            )
        )

        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)

        return (
            dict(
                question="What is the missing color of the part denoted with a question mark?",
                answer=answer,
                options=options,
                caption=f"There are 6 colored polygons arranged in a triangle with color {top_row} in the top row, {middle_row} in the middle row, and {bottom_row} in the bottom row.",
                explanation=f"We observe that the polygon with {explanation_side[0]} sides is {side2col[explanation_side[0]]} in color and the polygon with {explanation_side[1]} sides is {side2col[explanation_side[1]]} in color. Thus, the pattern is that the polygons with the same number of sides have the same color.",
                deduction=f"Based on the pattern that the polygons with the same number of sides have the same color, the missing color of the part with {sides[answer_location]} sides should be {answer}.",
            ),
            image,
        )


class PolygonSidesNumberPattern(BaseModel):
    image_size: int = 512
    scale_factor: int = 4
    path_font: str = "fonts/OpenSans-Medium.ttf"
    color: str = "#d9d2e8"  # Light purple

    def draw_polygon(self, draw, sides, center, size):
        angle = 360 / sides
        points = []

        for i in range(sides):
            x = center[0] + size * math.cos(math.radians(i * angle))
            y = center[1] + size * math.sin(math.radians(i * angle))
            points.append((x, y))

        draw.polygon(points, outline="black", fill=self.color, width=12)

    def make_sample(self):
        # Set the size of the image
        size = self.image_size * self.scale_factor
        image = Image.new("RGB", size=(size, size), color="white")
        draw = ImageDraw.Draw(image)

        sides = random.sample(range(3, 10), 6)
        answer = random.choice(sides)

        options = set(list(range(3, 10))) - {answer}
        options = random.sample(options, 3)
        options.append(answer)
        random.shuffle(options)

        center = size // 2
        distance = 175 * self.scale_factor

        for i, side in enumerate(sides):
            polygon_distance = distance - 0.5 * (i % 2) * distance
            angle = (i / len(sides)) * 2 * math.pi
            center_y = center - int(polygon_distance * math.cos(angle))
            center_x = center - int(polygon_distance * math.sin(angle))
            polygon_size = 60 * self.scale_factor

            self.draw_polygon(draw, side, (center_x, center_y), polygon_size)

            # Draw text in the center of the polygon
            draw.text(
                (center_x, center_y),
                str(side) if side != answer else "?",
                font=ImageFont.truetype(self.path_font, size=50 * self.scale_factor),
                anchor="mm",
                fill="black",
            )

        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)

        sides[sides.index(answer)] = "?"
        top_row = [sides[0]]
        middle_row = [sides[1], sides[5]]
        bottom_row = [sides[2], sides[3], sides[4]]

        explanation_side = [side for side in sides if side != "?"]

        return (
            dict(
                question="What is the missing number of the part denoted with a question mark?",
                answer=str(answer),
                options=options,
                caption=f"There are 6 numbered polygons arranged in a triangle with number {top_row} in the top row, {middle_row} in the middle row, and {bottom_row} in the bottom row.",
                explanation=f"We observe that the polygon with {explanation_side[0]} sides has the number {explanation_side[0]}, the polygon with {explanation_side[1]} sides has the number {explanation_side[1]}, the polygon with {explanation_side[2]} sides has the number {explanation_side[2]}, the polygon with {explanation_side[3]} sides has the number {explanation_side[3]}, and the polygon with {explanation_side[4]} sides has the number {explanation_side[4]}. Thus, the pattern is that the number inside the polygon represents the number of sides the polygon has.",
                deduction=f"Based on the pattern that the number inside the polygon represents the number of sides of the polygon, the missing number of the polygon with {answer} sides should be {answer}.",
            ),
            image,
        )


class RectangleHeightColorPattern(BaseModel):
    colors: Dict[str, str] = dict(
        blue="#6fa8dc",
        green="#93c47d",
        yellow="#ffd966",
        red="#e06666",
        purple="#8e7cc3",
        orange="#f6b26b",
    )
    image_size: int = 512
    scale_factor: int = 4
    path_font: str = "fonts/OpenSans-Light.ttf"

    def draw_box(
        self,
        draw: ImageDraw,
        point: Point,
        width: float,
        height: float,
        color: str,
    ):
        size = self.image_size * self.scale_factor
        draw.rounded_rectangle(
            [point[0] - width, point[1] - height, point[0] + width, point[1] + height],
            outline="black",
            width=size // 200,
            radius=size // 20,
            fill=color,
        )

    def draw_text(self, draw: ImageDraw, point: Point, text: str):
        size = self.image_size * self.scale_factor
        draw.text(
            point,
            text=text,
            font=ImageFont.truetype(self.path_font, size=size // 8),
            anchor="mm",
            fill="black",
        )

    @staticmethod
    def assign_numbers(colors: List[str]) -> List[int]:
        unique = sorted(set(colors))
        numbers = [i + 1 for i in range(len(unique))]
        random.shuffle(numbers)
        mapping = {u: i for u, i in zip(unique, numbers)}
        return [mapping[c] for c in colors]

    def make_sample(self):
        size = self.image_size * self.scale_factor
        image = Image.new("RGB", (size, size), "white")
        draw = ImageDraw.Draw(image)

        colors = random.sample(sorted(self.colors), k=3) * 2
        answer = colors[0]
        random.shuffle(colors)
        colors.append(answer)
        numbers = self.assign_numbers(colors)

        for i, num in enumerate(numbers):
            factor = size / (len(numbers) + 1)
            point = (factor * (i + 1), size // 2)
            is_answer = i == len(numbers) - 1
            self.draw_box(
                draw,
                point=point,
                width=factor / 2,
                height=factor * num,
                color="#eeeeee" if is_answer else self.colors[colors[i]],
            )
            if is_answer:
                self.draw_text(draw, point, text="?")

        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        lengths = [["short", "medium", "long"][num - 1] for num in numbers]
        instances = list(set((a, b) for (a, b) in zip(lengths, colors) if b != answer))
        colors[-1] = "?"

        return (
            dict(
                question="What is the missing color of the part denoted with a question mark?",
                answer=str(answer),
                options=sample_options(answer, sorted(self.colors), k=4),
                caption=f"There are {len(numbers)} rectangles in the image with varying colors and lengths. The lengths from left to right are {lengths}. The colors from left to right are {colors}.",
                explanation=f"We observe that the {instances[0][1]} rectangles are of {instances[0][0]} length and the {instances[1][1]} rectangles are of {instances[1][0]} length. Hence, the pattern is that the color of each rectangle corresponds to its length.",
                deduction=f"Based on the pattern that the color of each rectangle corresponds to its length, the missing color of the part denoted with a question mark should be {answer}.",
            ),
            image,
        )


class RectangleHeightNumberPattern(BaseModel):
    image_size: int = 512
    scale_factor: int = 4
    path_font: str = "fonts/OpenSans-Medium.ttf"
    color: str = "#d9ead3"  # Light green

    def draw_text(self, draw: ImageDraw, point: Point, text: str):
        size = self.image_size * self.scale_factor
        draw.text(
            point,
            text=text,
            font=ImageFont.truetype(self.path_font, size=size // 10),
            anchor="mm",
            fill="black",
        )

    def draw_box(self, draw: ImageDraw, point: Point, width: float, height: float):
        size = self.image_size * self.scale_factor
        draw.rounded_rectangle(
            [point[0] - width, point[1] - height, point[0] + width, point[1] + height],
            outline="black",
            width=size // 200,
            radius=size // 20,
            fill=self.color,
        )

    def make_sample(self):
        size = self.image_size * self.scale_factor
        image = Image.new("RGB", (size, size), "white")
        draw = ImageDraw.Draw(image)

        numbers = random.sample(list(range(1, 4)), 3) * 2
        answer = numbers[0]
        random.shuffle(numbers)
        numbers.append(answer)

        for i, num in enumerate(numbers):
            factor = size / (len(numbers) + 1)
            point = (factor * (i + 1), size // 2)
            self.draw_box(
                draw,
                point=point,
                width=factor / 2,
                height=factor * num,
            )
            self.draw_text(draw, point, "?" if i == len(numbers) - 1 else str(num))

        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        lengths = [["short", "medium", "long"][num - 1] for num in numbers]
        numbers[-1] = "?"

        return (
            dict(
                question="What is the missing number of the part denoted with a question mark?",
                answer=str(answer),
                options=random.sample([1, 2, 3, 4], k=4),
                caption=f"There are {len(numbers)} rectangles in the image with varying lengths and numbers inside them. The numbers from left to right are {numbers}. The lengths from left to right are {lengths}.",
                explanation=f"We observe that the short rectangles are denoted as 1, the medium rectangles are denoted as 2, and the long rectangles are denoted as 3. Hence, the pattern is that the number in each rectangle corresponds to its length.",
                deduction=f"Based on the pattern that the number in each rectangle corresponds to its length, the missing number of the rectangle with a question mark should be {answer}.",
            ),
            image,
        )


def get_polygon_point(num_sides: int, r: int, angle: int) -> Tuple[float, float]:
    # Find any point on a regular polygon as a function of angle from center (0 to 360)
    theta = math.radians(angle) % (2 * math.pi)  # Normalize to within 0 to 2π
    if num_sides == 4:
        theta += math.pi // 4
    alpha = 2 * math.pi / num_sides  # Angle per segment in radians
    vertex_before = int(theta // alpha)  # Nearest vertex before the point
    theta_vertex = vertex_before * alpha  # Angle to the nearest vertex before the point
    if theta == theta_vertex:
        return r * math.cos(theta), r * math.sin(theta)

    # Find the coordinates of the two nearest vertices
    x1, y1 = r * math.cos(theta_vertex), r * math.sin(theta_vertex)
    x2, y2 = r * math.cos(theta_vertex + alpha), r * math.sin(theta_vertex + alpha)

    # Linear interpolation on the edge
    t = (theta - theta_vertex) / alpha
    x = (1 - t) * x1 + t * x2
    y = (1 - t) * y1 + t * y2
    return x, y


class ShapeMorphPattern(BaseModel):
    image_size: int = 512
    scale_factor: int = 4
    path_font: str = "fonts/OpenSans-Medium.ttf"
    color: str = "#f4cccb"  # Light red for all shapes
    shapes: Dict[str, int] = dict(
        triangle=3, square=4, pentagon=5, hexagon=6, circle=720
    )

    @staticmethod
    def interpolate_points(
        points_a: List[Tuple[float, float]],
        points_b: List[Tuple[float, float]],
        weight: float,
    ) -> List[Tuple[float, float]]:
        outputs = []
        assert len(points_a) == len(points_b)
        assert 0 <= weight <= 1
        for a, b in zip(points_a, points_b):
            x = a[0] * (1 - weight) + b[0] * weight
            y = a[1] * (1 - weight) + b[1] * weight
            outputs.append((x, y))
        return outputs

    @staticmethod
    def offset_points(
        points: List[Tuple[float, float]], x: float, y: float
    ) -> List[Tuple[float, float]]:
        return [(p[0] + x, p[1] + y) for p in points]

    def draw_text(self, draw: ImageDraw, x: float, y: float, text: str):
        size = self.image_size * self.scale_factor
        draw.text(
            (x, y),
            text=text,
            font=ImageFont.truetype(self.path_font, size=size // 8),
            anchor="mm",
            fill="black",
        )

    def make_sample(self):
        # Set the size of the image
        size = self.image_size * self.scale_factor
        image = Image.new("RGB", size=(size, size), color="white")
        draw = ImageDraw.Draw(image)

        names = random.sample(sorted(self.shapes), k=2)
        radius = size // 10
        angles = list(range(360))
        points_a = [get_polygon_point(self.shapes[names[0]], radius, i) for i in angles]
        points_e = [get_polygon_point(self.shapes[names[1]], radius, i) for i in angles]
        points_b = self.interpolate_points(points_a, points_e, weight=0.25)
        points_c = self.interpolate_points(points_a, points_e, weight=0.50)
        points_d = self.interpolate_points(points_a, points_e, weight=0.75)

        if random.random() > 0.5:
            points_a = []
            answer = names[0]
            names[0] = "?"
        else:
            points_e = []
            answer = names[1]
            names[1] = "?"

        for lst, (x, y) in [
            (points_a, (size * 1 // 4, size * 1 // 4)),
            (points_b, (size * 2 // 4, size * 1 // 4)),
            (points_c, (size * 3 // 4, size * 1 // 4)),
            (points_d, (size * 3 // 4, size * 2 // 4)),
            # (points_c, (size * 2 // 4, size * 2 // 4)),
            (points_b, (size * 1 // 4, size * 2 // 4)),
            (points_c, (size * 1 // 4, size * 3 // 4)),
            (points_d, (size * 2 // 4, size * 3 // 4)),
            (points_e, (size * 3 // 4, size * 3 // 4)),
        ]:
            if not lst:
                self.draw_text(draw, x, y, text="?")
                continue
            lst = self.offset_points(lst, x=x, y=y)
            draw.polygon(lst, fill=self.color, outline="black", width=size // 150)

        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        direction = (
            "top left to bottom right"
            if names[1] == "?"
            else "bottom right to top left"
        )
        start_shape = names[1] if names[0] == "?" else names[0]
        answer_location = " ".join(direction.split()[-2:])

        return (
            dict(
                question="What is the missing shape of the part denoted with a question mark?",
                answer=answer,
                options=sample_options(answer, sorted(self.shapes), k=4),
                caption=f"There are eight shapes arranged in a grid. The top left shape is a {names[0]} and the bottom right shape is a {names[1]}. The other shapes do not appear to regular shapes.",
                explanation=f"We observe that from the {direction} direction, the shapes look like a {start_shape} but gradually change shape into something like a {answer}. Hence, the pattern is the the shapes are morphing between {start_shape} and {answer} shapes.",
                deduction=f"Based on the pattern that the shapes are morphing between {start_shape} and {answer} shapes, the missing shape at the {answer_location} should be a {answer}.",
            ),
            image,
        )


class ShapeReflectPattern(BaseModel):
    image_size: int = 512
    scale_factor: int = 4
    path_font: str = "fonts/OpenSans-Medium.ttf"
    color: str = "#d9ead3"  # Light green for all shapes
    shapes: Dict[str, int] = dict(triangle=3, square=4, pentagon=5, hexagon=6)

    def draw_circle(self, draw: ImageDraw, x: int, y: int, radius: int, **kwargs):
        position = x - radius, y - radius, x + radius, y + radius
        line_width = self.image_size * self.scale_factor // 200
        draw.ellipse(position, width=line_width, **kwargs)

    def draw_dotted_circle(
        self, draw: ImageDraw, center: Tuple[float, float], radius: int, num_dots: int
    ):
        self.draw_circle(draw, *center, radius, outline="black")
        angle_between_dots = 2 * math.pi / num_dots
        for i in range(0, num_dots, 2):
            theta = angle_between_dots * i
            x = round(center[0] + radius * math.cos(theta))
            y = round(center[1] + radius * math.sin(theta))
            self.draw_circle(draw, x, y, radius=radius // 10, fill="white")

    def draw_shape(
        self,
        draw: ImageDraw,
        center: Tuple[float, float],
        num_sides: int,
        do_flip: bool,
    ):
        size = self.image_size * self.scale_factor
        if num_sides == 0:
            draw.text(
                center,
                text="?",
                font=ImageFont.truetype(self.path_font, size=size // 10),
                anchor="mm",
                fill="black",
            )
            self.draw_dotted_circle(draw, center, radius=size // 10, num_dots=32)
            return

        # Adjust start angle based on even or odd number of sides
        angle = math.pi * 2 / num_sides
        if num_sides % 2 == 0:
            start = math.pi / 2 - angle / 2
        else:
            start = 0
        if do_flip:
            start += math.pi

        radius = size // 10
        points = [
            (
                center[0] + math.sin(start + angle * i) * radius,
                center[1] - math.cos(start + angle * i) * radius,
            )
            for i in range(num_sides)
        ]

        width = size // 200
        draw.polygon(points, fill=self.color, outline="black", width=width)
        draw.line([(size // 8, size // 2), (size * 7 // 8, size // 2)], "black", width)

    def make_sample(self):
        # Set the size of the image
        size = self.image_size * self.scale_factor
        image = Image.new("RGB", size=(size, size), color="white")
        draw = ImageDraw.Draw(image)

        a, b, c = size // 4, size // 2, size * 3 // 4
        positions = [(x, y) for y in [size // 3, size * 2 // 3] for x in [a, b, c]]
        names = random.sample(sorted(self.shapes), k=3) * 2
        i_answer = random.randint(0, len(names) - 1)
        answer = names[i_answer]

        for i, n in enumerate(names):
            num_sides = 0 if i == i_answer else self.shapes[n]
            self.draw_shape(draw, positions[i], num_sides, do_flip=i >= 3)
            # self.draw_shape(draw, positions[i], num_sides, do_flip=False)

        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        names[i_answer] = "?"
        instances = sorted(set(n for n in names if n != "?"))
        return (
            dict(
                question="What is the missing shape denoted by a question mark?",
                answer=answer,
                options=sample_options(answer, sorted(self.shapes), k=4),
                caption=f"There are six shapes in the image separated by a line. In the top part there are {names[:3]}. In the bottom part there are {names[3:]}.",
                explanation=f"We observe that the {instances[0]} is reflected across the line as a {instances[0]}. Similarly, the {instances[1]} is reflected as a {instances[1]}. Hence, the pattern is that each shape in the top part is reflected in the bottom part.",
                deduction=f"Based on the pattern that each shape in the top part is reflected in the bottom part, the missing shape which is reflected from a {answer} part should be a {answer}.",
            ),
            image,
        )


class ShapeSizeGridPattern(BaseModel):
    image_size: int = 512
    scale_factor: int = 4
    path_font: str = "fonts/OpenSans-Medium.ttf"
    color: str = "#d9ead3"  # Light green for all shapes
    shapes: Dict[str, int] = dict(triangle=3, square=4, pentagon=5, hexagon=6)

    @staticmethod
    def get_points(num_sides: int, center: Point, radius: int) -> List[Point]:
        vertices = []
        for i in range(num_sides):
            theta = 2 * math.pi / num_sides * i
            if num_sides % 2 != 0:
                theta -= math.pi / 2
            elif num_sides == 4:
                theta -= math.pi / 4

            x = center[0] + radius * math.cos(theta)
            y = center[1] + radius * math.sin(theta)
            vertices.append((x, y))
        return vertices

    def draw_text(self, draw: ImageDraw, point: Point, text: str):
        size = self.image_size * self.scale_factor
        draw.text(
            point,
            text=text,
            font=ImageFont.truetype(self.path_font, size=size // 8),
            anchor="mm",
            fill="black",
        )

    @staticmethod
    def random_rotate_matrix(matrix: List[list]) -> List[list]:
        angle = random.choice([90, 180, 270, 360])
        if angle == 90:
            # Rotate by 90 degrees
            new = [list(row) for row in zip(*matrix[::-1])]
        elif angle == 180:
            # Rotate by 180 degrees
            new = [row[::-1] for row in matrix[::-1]]
        elif angle == 270:
            # Rotate by 270 degrees (or 90 degrees counter-clockwise)
            new = [list(row) for row in zip(*matrix)][::-1]
        else:
            new = matrix

        return [
            [
                (new[i][j][0], new[i][j][1], matrix[i][j][2], matrix[i][j][3])
                for j in range(len(matrix[i]))
            ]
            for i in range(len(matrix))
        ]

    def make_sample(self):
        size = self.image_size * self.scale_factor
        image = Image.new("RGB", size=(size, size), color="white")
        draw = ImageDraw.Draw(image)

        a, b, c = random.sample(sorted(self.shapes), k=3)
        mapping = dict(small=(size * 0.05), medium=(size * 0.09), large=(size * 0.13))
        d, e, f = size * 0.25, size * 0.50, size * 0.75
        data = [
            [(a, "small", d, d), (b, "small", e, d), (c, "small", f, d)],
            [(a, "medium", d, e), (b, "medium", e, e), (c, "medium", f, e)],
            [(a, "large", d, f), (b, "large", e, f), (c, "large", f, f)],
        ]
        data = self.random_rotate_matrix(data)
        answer = random.choice([item for lst in data for item in lst])

        for lst in data:
            for item in lst:
                name, radius, x, y = item
                if item == answer:
                    self.draw_text(draw, point=(x, y), text="?")
                    continue
                shape = self.get_points(
                    num_sides=self.shapes[name],
                    center=(x, y),
                    radius=mapping[radius],
                )
                draw.polygon(shape, fill=self.color, outline="black", width=size // 100)

        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        m = [[("?" if x == answer else f"{x[1]} {x[0]}") for x in row] for row in data]
        if len(set(x[1] for x in data[0])) == 1:
            trend_size = f"rows contain {data[0][0][1]} shapes, {data[1][0][1]} shapes, and {data[2][0][1]} shapes respectively"
            trend_shapes = f"columns contain {data[0][0][0]}s, {data[0][1][0]}s, and {data[0][2][0]}s respectively"
            pattern = "the shapes within each column are the same, while each row progresses the size of the shapes"
        else:
            trend_size = f"columns contain {data[0][0][1]} shapes, {data[0][1][1]} shapes, and {data[0][2][1]} shapes respectively"
            trend_shapes = f"rows contain {data[0][0][0]}s, {data[1][0][0]}s, and {data[2][0][0]}s respectively"
            pattern = "the shapes within each row are the same, while each column progresses the size of the shapes"

        return (
            dict(
                question=f"What is the size of the missing part denoted by a question mark?",
                answer=answer[1],
                options=sample_options(answer[1], sorted(mapping), k=3),
                caption=f"There are 9 shapes arranged in a grid with different sizes in the image, of which there is 1 missing shape. The first row is {m[0]}, the second row is {m[1]}, and the third row is {m[2]}.",
                explanation=f"We observe that the {trend_size}. On the other hand, the {trend_shapes}. Hence, the pattern is that {pattern}.",
                deduction=f"Based on the pattern that {pattern}, the size of the missing {answer[0]} should be {answer[1]}.",
            ),
            image,
        )


class ShapeSizeHexagonPattern(BaseModel):
    image_size: int = 512
    scale_factor: int = 4
    path_font: str = "fonts/OpenSans-Medium.ttf"
    color: str = "#fce5cd"  # Light orange for all shapes
    shapes: Dict[str, int] = dict(triangle=3, square=4, pentagon=5, hexagon=6)

    @staticmethod
    def get_points(num_sides: int, center: Point, radius: int) -> List[Point]:
        vertices = []
        for i in range(num_sides):
            theta = 2 * math.pi / num_sides * i
            if num_sides % 2 != 0:
                theta -= math.pi / 2
            elif num_sides == 4:
                theta -= math.pi / 4

            x = center[0] + radius * math.cos(theta)
            y = center[1] + radius * math.sin(theta)
            vertices.append((x, y))
        return vertices

    def draw_text(self, draw: ImageDraw, point: Point, text: str):
        size = self.image_size * self.scale_factor
        draw.text(
            point,
            text=text,
            font=ImageFont.truetype(self.path_font, size=size // 6),
            anchor="mm",
            fill="black",
        )

    def make_sample(self):
        size = self.image_size * self.scale_factor
        image = Image.new("RGB", size=(size, size), color="white")
        draw = ImageDraw.Draw(image)
        center = size // 2, size // 2

        mapping = dict(small=(size * 0.05), medium=(size * 0.10), large=(size * 0.15))
        shape_names = random.sample(sorted(self.shapes), k=3)
        size_names = random.sample(sorted(mapping), k=3)
        indices = [0, 1, 2, 0, 1, 2]
        points = self.get_points(num_sides=6, center=center, radius=size // 3)

        assert len(indices) == len(points)
        for i, p in zip(indices, points):
            shape = self.get_points(
                num_sides=self.shapes[shape_names[i]],
                center=p,
                radius=mapping[size_names[i]],
            )
            draw.polygon(shape, fill=self.color, outline="black", width=size // 100)

        answer_shape = random.choice(shape_names)
        answer_size = size_names[shape_names.index(answer_shape)]
        self.draw_text(draw, center, "?")

        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        return (
            dict(
                question=f"What is the size of the missing shape denoted with a question mark if it is a {answer_shape}?",
                answer=answer_size,
                options=sample_options(answer_size, options=size_names, k=3),
                caption=f"There are 7 shapes with different sizes in the image, of which there is a missing {answer_shape} in the center. The other shapes are arranged around the center, which are {shape_names * 2} in anti-clockwise order. Their corresponding sizes are {size_names * 2}.",
                explanation=f"We observe that the {shape_names[0]}s are {size_names[0]} size, the {shape_names[1]}s are {size_names[1]} size, and the {shape_names[2]}s are {size_names[2]} size. Hence, the pattern is that each shape appears with a distinct size.",
                deduction=f"Based on the pattern that each shape appears with a distinct size, the size of the missing {answer_shape} should be {answer_size}.",
            ),
            image,
        )


class SizeCyclePattern(BaseModel):
    image_size: int = 512
    scale_factor: int = 4
    path_font: str = "fonts/OpenSans-Medium.ttf"
    color: str = "#fff2cc"  # Light yellow for all circles

    def draw_circle(self, draw: ImageDraw, point: Point, radius: int):
        x, y = point
        position = x - radius, y - radius, x + radius, y + radius
        line_width = self.image_size * self.scale_factor // 150
        draw.ellipse(position, fill=self.color, outline="black", width=line_width)

    @staticmethod
    def get_points(n_sides: int, center: Point, radius: int, angle: int) -> List[Point]:
        def regular_polygon_vertices(num_sides):
            vertices = []
            for i in range(num_sides):
                theta = 2 * math.pi / num_sides * i
                x = center[0] + radius * math.cos(theta)
                y = center[1] + radius * math.sin(theta)
                vertices.append((x, y))
            return vertices

        def rotate_point(origin, point):
            ox, oy = origin
            px, py = point
            theta = math.radians(angle)  # Convert to radians
            qx = ox + math.cos(theta) * (px - ox) - math.sin(theta) * (py - oy)
            qy = oy + math.sin(theta) * (px - ox) + math.cos(theta) * (py - oy)
            return qx, qy

        polygon_vertices = regular_polygon_vertices(n_sides)
        # assert self.get_centroid(polygon_vertices) == center
        rotated_vertices = [rotate_point(center, v) for v in polygon_vertices]
        # assert self.get_centroid(rotated_vertices) == center
        return rotated_vertices

    def draw_text(self, draw: ImageDraw, point: Point, text: str):
        size = self.image_size * self.scale_factor
        draw.text(
            point,
            text=text,
            font=ImageFont.truetype(self.path_font, size=size // 10),
            anchor="mm",
            fill="black",
        )

    def make_sample(self):
        # Set the size of the image
        size = self.image_size * self.scale_factor
        image = Image.new("RGB", size=(size, size), color="white")
        draw = ImageDraw.Draw(image)

        center = size // 2, size // 2
        offset = random.randint(0, 360)
        mapping = dict(
            small=(size * 0.050, size // 9, 0 + offset),
            medium=(size * 0.075, size // 4, 20 + offset),
            large=(size * 0.100, size // 2.5, 45 + offset),
        )

        names = []
        num_sides = 3
        answer = ""
        i_answer = random.randint(0, num_sides * len(mapping) - 1)
        for n, (radius, distance, angle) in mapping.items():
            for p in self.get_points(num_sides, center, distance, angle):
                names.append(n)
                if len(names) - 1 == i_answer:
                    self.draw_text(draw, p, "?")
                    answer = n
                else:
                    self.draw_circle(draw, p, round(radius))

        names[i_answer] = "?"
        arms = [
            [names[0], names[3], names[6]],
            [names[1], names[4], names[7]],
            [names[2], names[5], names[8]],
        ]
        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        answer_location = dict(
            small="closest to center",
            medium="neither closest nor farthest from center",
            large="farthest from center",
        )[answer]

        return (
            dict(
                question="What is the size of the missing circle denoted with a question mark?",
                answer=answer,
                options=sample_options(answer, sorted(mapping), k=3),
                caption=f"There are circles arranged in a spiral with three arms. The first arm has circles of sizes {arms[0]}, the second arm has circles of sizes {arms[1]}, and the third arm has circles of sizes {arms[2]}.",
                explanation=f"We observe that the circles in each arm progress in size from small to medium to large. Thus, the pattern is that the circles in each arm get bigger as they progress away from the center of the spiral.",
                deduction=f"Based on the pattern that the circles in each arm get bigger as they progress away from the center of the spiral, the size of the missing part that is {answer_location} should be {answer}.",
            ),
            image,
        )


class SizeGridPattern(BaseModel):
    image_size: int = 512
    scale_factor: int = 4
    path_font: str = "fonts/OpenSans-Medium.ttf"
    color: str = "#fff2cc"  # Light yellow for all circles

    def draw_circle(self, draw: ImageDraw, x: int, y: int, radius: int):
        position = x - radius, y - radius, x + radius, y + radius
        line_width = self.image_size * self.scale_factor // 200
        draw.ellipse(position, fill=self.color, outline="black", width=line_width)

    def make_sample(self):
        # Set the size of the image
        size = self.image_size * self.scale_factor
        image = Image.new("RGB", size=(size, size), color="white")
        draw = ImageDraw.Draw(image)
        a, b, c = size // 4, size // 2, size * 3 // 4
        positions = [(x, y) for x in [a, b, c] for y in [a, b, c]]

        radii = dict(small=size // 30, medium=size // 20, large=size // 10)
        keys = random.sample(radii.keys(), k=len(radii))
        values = [[0, 2, 6, 8], [1, 3, 5, 7], [4]]
        mapping = {k: v for k, v in zip(keys, values)}
        i_answer = random.choice([0, 2, 6, 8, 1, 3, 5, 7])
        answer = ""

        for k, lst in mapping.items():
            radius = radii[k]
            for i in lst:
                if i == i_answer:
                    answer = k
                    draw.text(
                        positions[i],
                        text="?",
                        font=ImageFont.truetype(self.path_font, size=size // 10),
                        anchor="mm",
                        fill="black",
                    )
                else:
                    self.draw_circle(draw, *positions[i], radius=radius)

        options = sample_options(answer, list(radii.keys()), k=3)
        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        grid = ["?"] * 9
        for k, lst in mapping.items():
            for i in lst:
                if i != i_answer:
                    grid[i] = k
        grid = grid[::-1]
        answer_location = (
            "at the corner" if i_answer in [0, 2, 6, 8] else "adjacent to the center"
        )

        return (
            dict(
                question="What is the size of the missing part denoted with a question mark?",
                answer=answer,
                options=options,
                caption=f"There are circles arranged in a grid formation with varying sizes in the image. The sizes in the first row are {grid[:3]}, the sizes in the second row are {grid[3:6]}, and the sizes in the third row are {grid[6:9]}.",
                explanation=f"We observe that the circles at the corners are {keys[0]} size, while the circles directly adjacent to the center are {keys[1]} size. Only the center circle is {keys[2]} size. Hence, the pattern is that the circles alternate in size depending on if they are at the corner or adjacent to the center.",
                deduction=f"Based on the pattern that the circles alternate in size depending on if they are at the corner or adjacent to the center, the size of the missing part that is {answer_location} should be {answer}.",
            ),
            image,
        )


class NumbersTrianglePattern(BaseModel):
    path_font: str = "fonts/OpenSans-Medium.ttf"
    image_size: int = 512
    scale_factor: int = 4
    num_sides: int = 3
    color: str = "#cfe2f3"  # Light blue

    def get_points(self, center: Point, radius: float) -> List[Point]:
        vertices = []
        for i in range(self.num_sides):
            theta = 2 * math.pi / self.num_sides * i
            x = center[0] + radius * math.cos(theta)
            y = center[1] + radius * math.sin(theta)
            vertices.append((x, y))
        return vertices

    def draw_circle(self, draw: ImageDraw, point: Point, radius: float, **kwargs):
        x, y = point
        position = x - radius, y - radius, x + radius, y + radius
        line_width = self.image_size * self.scale_factor // 200
        draw.ellipse(position, width=line_width, fill=self.color, **kwargs)

    def draw_text(self, draw: ImageDraw, point: Point, text: str):
        size = self.image_size * self.scale_factor
        draw.text(
            point,
            text=text,
            font=ImageFont.truetype(self.path_font, size=size // 14),
            anchor="mm",
            fill="black",
        )

    def make_sample(self):
        size = self.image_size * self.scale_factor
        image = Image.new("RGB", size=(size, size), color="white")
        draw = ImageDraw.Draw(image)
        center = size / 2, size / 2

        a, b, c, d, e, f = random.sample(list(range(1, 10)), k=6)
        numbers = [a * b, a, b, c * d, c, d, e * f, e, f]
        i = random.randint(0, len(numbers) - 1)
        answer = numbers[i]
        numbers[i] = "?"

        for i, point in enumerate(self.get_points(center, radius=size / 4)):
            # noinspection PyTypeChecker
            subpoints = self.get_points(point, radius=size / 10)
            draw.polygon(subpoints, outline="black", width=size // 200)
            for j, sub in enumerate(subpoints):
                # noinspection PyTypeChecker
                self.draw_circle(draw, sub, radius=size / 16, outline="black")
                # noinspection PyTypeChecker
                self.draw_text(draw, sub, str(numbers[i * 3 + j]))

        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        groups = [numbers[:3][::-1], numbers[3:6][::-1], numbers[6:][::-1]][::-1]
        instances = [lst for lst in groups if "?" not in lst]
        instances.extend([lst for lst in groups if "?" in lst])
        assert len(instances) == 3

        return (
            dict(
                question="What is the missing number of the part denoted with a question mark?",
                options=[str(o) for o in generate_number_options(int(answer), k=4)],
                answer=answer,
                caption=f"There are three groups of numbers with a triangle arrangement in the image. The first group is {groups[0]}, the second group is {groups[1]}, and the third group is {groups[2]}.",
                explanation=f"We observe that the number {instances[0][2]} is the product of {instances[0][1]} and {instances[0][0]}. Similarly, the number {instances[1][2]} is the product of {instances[1][1]} and {instances[1][0]}. Hence, the pattern is that the rightmost number in each group is the product of the other two numbers.",
                deduction=f"Based on the pattern that the rightmost number in each group is the product of the other two numbers, the missing number of the group {instances[2]} should be {answer}.",
            ),
            image,
        )


def get_pixels(image: Image, fraction_x: float, fraction_y: float) -> Tuple[int, int]:
    x = round(image.width * fraction_x)
    y = round(image.height * fraction_y)
    return x, y


class VennPattern(BaseModel):
    path_template: str = "templates/puzzle-venn.png"
    path_font: str = "fonts/OpenSans-Medium.ttf"
    rule: str = "{} + {}"
    image_size: int = 512

    def draw_text(self, image: Image, text: str, position: Tuple[int, int]):
        draw = ImageDraw.Draw(image)
        draw.text(
            position,
            text,
            font=ImageFont.truetype(self.path_font, size=image.width // 16),
            anchor="mm",
            fill="black",
        )

    def make_sample(self):
        image = Image.open(self.path_template)
        a, b, c = random.sample(list(range(1, 10)), k=3)
        ab = eval(self.rule.format(a, b))
        bc = eval(self.rule.format(b, c))

        if random.random() > 0.5:
            answer = a
            a = "?"
        else:
            answer = c
            c = "?"

        self.draw_text(image, str(a), get_pixels(image, 0.25, 0.5))
        self.draw_text(image, str(b), get_pixels(image, 0.50, 0.5))
        self.draw_text(image, str(c), get_pixels(image, 0.75, 0.5))
        self.draw_text(image, str(ab), get_pixels(image, 0.38, 0.5))
        self.draw_text(image, str(bc), get_pixels(image, 0.62, 0.5))

        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        lst = [a, b, ab] if c == "?" else [b, c, bc]
        lst_b = [a, b, ab] if c != "?" else [b, c, bc]

        return (
            dict(
                question="What is the missing number of the part denoted with a question mark?",
                answer=answer,
                options=[str(o) for o in generate_number_options(answer, k=4)],
                caption=f"There are 3 overlapping circles containing the numbers {[a, b, c]}. The overlapping part between the first and second circle contains the number {ab}. The overlapping part between the second and third circle contains the number {bc}.",
                explanation=f"We observe that the circles with {lst[0]} and {lst[1]} overlap to form the part {lst[2]}, where {lst[0]} + {lst[1]} = {lst[2]}. Hence, the pattern is most likely that the numbers in the overlapping parts are the sum of the numbers in the corresponding circles.",
                deduction=f"Based on the pattern that the numbers in the overlapping parts are the sum of the numbers in the corresponding circles, the missing number of the circle where the overlapping part is {lst_b[-1]} should be {answer}.",
            ),
            image,
        )


def select_pattern(name: str, **kwargs):

    if name == "circle_size_number":
        return CircleSizeNumberPattern(**kwargs)
    if name == "color_grid":
        return ColorGridPattern(**kwargs)
    if name == "color_hexagon":
        return ColorHexagonPattern(**kwargs)
    if name == "color_number_hexagon":
        return ColorNumberHexagonPattern(**kwargs)
    if name == "color_overlap_squares":
        return ColorOverlapSquaresPattern(**kwargs)
    if name == "color_size_circle":
        return ColorSizeCirclePattern(**kwargs)
    if name == "grid_number_color":
        return GridNumberColorPattern(**kwargs)
    if name == "grid_number":
        return GridNumberPattern(**kwargs)
    if name == "polygon_sides_color":
        return PolygonSidesColorPattern(**kwargs)
    if name == "polygon_sides_number":
        return PolygonSidesNumberPattern(**kwargs)
    if name == "rectangle_height_color":
        return RectangleHeightColorPattern(**kwargs)
    if name == "rectangle_height_number":
        return RectangleHeightNumberPattern(**kwargs)
    if name == "shape_morph":
        return ShapeMorphPattern(**kwargs)
    if name == "shape_reflect":
        return ShapeReflectPattern(**kwargs)
    if name == "shape_size_grid":
        return ShapeSizeGridPattern(**kwargs)
    if name == "shape_size_hexagon":
        return ShapeSizeHexagonPattern(**kwargs)
    if name == "size_cycle":
        return SizeCyclePattern(**kwargs)
    if name == "size_grid":
        return SizeGridPattern(**kwargs)
    if name == "triangle":
        return NumbersTrianglePattern(**kwargs)
    if name == "venn":
        return VennPattern(**kwargs)

    raise KeyError(name)


def sample_options(answer: str, options: List[str], k: int):
    # Ensure random order and no duplicates
    options = [o for o in options if o != answer]
    assert len(options) + 1 >= k
    options = random.sample(options, k=k - 1)
    options.append(answer)
    assert len(set(options)) == k
    return random.sample(options, k=k)


def generate_number_options(num: int, k: int) -> List[int]:
    # Automatically detect the range and random.sample
    assert num >= 0, "Negative numbers not supported yet"
    options = [10, 100, 1000, 10000, 100000]
    for max_value in options:
        if num <= max_value:
            values = [i for i in range(max_value) if i != num]
            lst = random.sample(values, k=k - 1)
            lst.append(num)
            assert len(set(lst)) == len(lst)
            return random.sample(lst, k=len(lst))
    raise ValueError(f"Range exceeded: {num}, options: {options}")


def convert_image_to_text(image: Image) -> str:
    # This is also how OpenAI encodes images: https://platform.openai.com/docs/guides/vision
    with io.BytesIO() as output:
        image.save(output, format="PNG")
        data = output.getvalue()
    return base64.b64encode(data).decode("utf-8")


def create_data(
    pattern_name: str, path: str = "data", limit: int = 100, unique: bool = True
):
    random.seed(0)
    np.random.seed(0)

    os.makedirs(f"{path}/images/{pattern_name}", exist_ok=True)

    progress = tqdm(range(limit))

    pattern = select_pattern(pattern_name)
    samples = []
    seen = []
    question_idx = 0

    count = 0
    while len(samples) < limit:
        sample, image = pattern.make_sample()
        count += 1

        sample_check = copy.deepcopy(sample)
        image_string = convert_image_to_text(image)
        sample_check["image_string"] = image_string

        if sample_check not in seen or not unique:
            seen.append(sample_check)

            image_path = f"images/{pattern_name}/{pattern_name}_{question_idx:04}.png"
            image.save(f"{path}/{image_path}")

            sample = list(sample.items())
            sample.insert(0, ("image", image_path))
            sample = dict(sample)

            print(sample)

            question_idx += 1
            samples.append(sample)
            progress.update()

    with open(f"{path}/{pattern_name}.json", "w") as f:
        for line in samples:
            f.write(json.dumps(line) + "\n")


if __name__ == "__main__":
    Fire()
