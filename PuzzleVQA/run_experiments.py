import os

# 데이터셋 이름 리스트
datasets = [
    "venn",
    "color_grid",
    "color_hexagon",
    "size_cycle",
    "size_grid",
    "shape_morph",
    "shape_reflect",
    "grid_number",
    "triangle"
]

# sequential reasoning
sequential_command = (
    "python main_debate.py evaluate_multi_choice_sequential data/{dataset}.json "
    "--model_name gpt4o "
    "--prompt_name cot_multi_extract "
)

# single agent
dynamic = (
    "python main_dynamic.py evaluate_multi_choice data/{dataset}.json "
    "--model_name gpt4o "
    "--prompt_name cot_multi_extract "
)



# 순차적으로 명령어 실행
for dataset in datasets:
    command = dynamic.format(dataset=dataset)
    print(f"Running: {command}")
    os.system(command)

# 순차적으로 명령어 실행
# for dataset in datasets:
#     command = sequential_command.format(dataset=dataset)
#     print(f"Running: {command}")
#     os.system(command)
