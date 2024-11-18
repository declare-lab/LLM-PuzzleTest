import os

# 데이터셋 이름 리스트
datasets = [
    "color_size_circle.json",
    "grid_number_color.json",
    "grid_number.json",
    "polygon_sides_color.json",
    "polygon_sides_number.json",
    "rectangle_height_color.json",
    "rectangle_height_number.json",
    "shape_morph.json",
    "shape_reflect.json",
    "shape_size_grid.json",
    "shape_size_hexagon.json",
    "size_cycle.json",
    "size_grid.json",
    "triangle.json",
    "venn.json",
]

# sequential reasoning
sequential_command = (
    "python main_debate.py evaluate_multi_choice_sequential data/{dataset} "
    "--model_name gpt4o "
    "--prompt_name cot_multi_extract "
)

# single agent
single_command = (
    "python main.py evaluate_multi_choice data/{dataset} "
    "--model_name gpt4o "
    "--prompt_name cot_multi_extract "
)

# 순차적으로 명령어 실행
for dataset in datasets:
    command = single_command.format(dataset=dataset)
    print(f"Running: {command}")
    os.system(command)  # 실제 명령어 실행

# # 순차적으로 명령어 실행
# for dataset in datasets:
#     command = sequential_command.format(dataset=dataset)
#     print(f"Running: {command}")
#     os.system(command)  # 실제 명령어 실행