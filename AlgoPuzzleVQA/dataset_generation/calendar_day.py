import os
import copy
import json
import random
import calendar
import numpy as np
import pandas as pd
from tqdm import tqdm
from IPython.display import display
from PIL import Image, ImageOps, ImageDraw, ImageFont


def month_calendar(year, month):
    view = " " * 3 + calendar.month(year, month, 4, 2).replace(f"{year}\n", "\n")
    rows = view.split("\n\n")[:-1]

    return rows


def day_of_week(year, month, day):
    
    return calendar.day_name[calendar.weekday(year, month, day)]


def choose_random_day(year, month):
    if month in [1, 3, 5, 7, 8, 10, 12]:
        day = random.randint(1, 31)
    elif month in [4, 6, 9, 10]:
        day = random.randint(1, 30)
    elif calendar.isleap(year) and month == 2:
        day = random.randint(1, 29)
    else:
        day = random.randint(1, 28)

    return day, day_of_week(year, month, day)


def calendar_image(year, month):
    image_width, image_height = 1200, 900
    text_left, text_height, height_gap = 100, 75, 100
    font = ImageFont.truetype("fonts/RobotoMono-Regular.ttf", 50)

    canvas = Image.new("RGB", (image_width, image_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    copy = canvas.copy()

    rows = month_calendar(year, month)
    for j, row in enumerate(rows):
        ImageDraw.Draw(copy).text((text_left, text_height + j * height_gap), row, (0, 0, 0), font=font)
    
    return copy


if __name__ == "__main__":

    os.makedirs("data/images/calendar")

    data, question_index, num_instances = [], 0, 100
    progress_bar = tqdm(range(num_instances))
    solution_set = []

    mapper = {True: "leap", False: "non-leap"}

    while question_index < num_instances:
        year = random.randint(1900, 2901)
        is_leap_current = calendar.isleap(year)
        months = random.sample(list(range(1, 13)), 2)
        
        mode = random.randint(1, 3)
        question = f"The image shows the calendar of a month of a particular {mapper[is_leap_current]} year. "
        
        month = months[1]
        
        if mode == 1:
            day, answer = choose_random_day(year, month)
            question += f"Which day of the week was on {calendar.month_name[month]} {day} of that year?"
            query_year = year
        
        elif mode == 2:
            day, answer = choose_random_day(year - 1, month)
            is_leap_other = calendar.isleap(year - 1)
            question += f"The previous year was a {mapper[is_leap_other]} year. "
            question += f"Which day of the week was on {calendar.month_name[month]} {day} of the previous year?"
            query_year = year - 1
            
        elif mode == 3:
            day, answer = choose_random_day(year + 1, month)
            is_leap_other = calendar.isleap(year + 1)
            question += f"The next year is a {mapper[is_leap_other]} year. "
            question += f"Which day of the week is on {calendar.month_name[month]} {day} of the next year?"
            query_year = year + 1

        core_solution = {
            "image_month": months[0],
            "query_month": month, "query_day": day,
            "question": question
        }

        if core_solution not in solution_set:
            solution_set.append(core_solution)

            instance_image = calendar_image(year, months[0])
            fname = f"data/images/calendar/calendar_{question_index:04}.jpg"
            instance_image.save(fname, dpi=(1000, 1000))
            
            example = {
                "image": fname[5:], "question": question, "answer": answer,
                "solution": {
                    "image_year": year, "image_month": (months[0], calendar.month_name[months[0]]),
                    "query_year": query_year, "query_month": (month, calendar.month_name[month]), "query_day": day
                }
            }

            data.append(example)
            question_index += 1
            progress_bar.update(1)


    with open("data/calendar.json", "w") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")

