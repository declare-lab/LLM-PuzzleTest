from pathlib import Path

import pandas as pd
from fire import Fire
from pydantic import BaseModel
from tqdm import tqdm

from data_loading import Data, Sample, convert_text_to_image
from modeling import select_model
from prompting import select_prompter

import mad
import base64


class Scorer(BaseModel):
    def run(self, sample: Sample) -> float:
        raise NotImplementedError


class ExactScorer(Scorer):
    def run(self, sample: Sample) -> float:
        if sample.pred == sample.answer:
            return 1.0
        return 0.0


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_attributes(base64_image, model_name):
    prompt = f"""
    You are an attribute analysis agent. 
    Given a puzzle image, your task is to identify which attribute changes in the given image.
    Your choices are among [1. colors, 2. numbers, 3. sizes, 4. shapes]. Number of answers can be 1 or 2.
    Only return the index of choices. (Example: 3 / Example: 2, 3)
    """

    message = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    },
                },
            ],
        },
    ]
    completion_2 = mad.generate_answer(message, model_name)
    answer = mad.construct_assistant_message(completion_2)
    message.append(answer)

    attribute_mapping = {
        '1': 'colors',
        '2': 'numbers',
        '3': 'sizes',
        '4': 'shapes'
    }
    # Split the GPT output by commas and strip whitespace
    numbers = answer.split(',')
    numbers = [num.strip() for num in numbers]

    # Map each number to its corresponding attribute name
    attributes = [attribute_mapping[num] for num in numbers if num in attribute_mapping]
    print(f"Varying attributes:\n{attributes}\n")

    return attributes


def single_agent(model_name, sample, prompter, model):
    if "qwen" in model_name:
        image = sample.image
    else:
        image = convert_text_to_image(sample.image_string)

    raw_output = model.run(sample.prompt, image)
    pred = prompter.get_answer(sample.raw_output, sample.options)
    return pred, raw_output


def sequential_reasoning(base64_image, model_name, sample, prompter):
    agent_contexts = []
    example_image_2 = encode_image("data/images/size_cycle/size_cycle_0006.png")

    # Agent 1: Visual Perception
    perception = f"""
    You are an expert for visual perception agent in puzzle solving.
    Explain how the objects are arranged and what shape they form, 
    and describe all attributes [1.colors 2.numbers 3.sizes 4.shapes] of each objects.
    An in-context example is given, and don't just repeat the example answer.
    Do not provide logical inferences, describe only what you see.

    Let's think step by step.
    """

    example_prompt_2 = f"""
    Example
    There are circles arranged in a spiral with three arms. The attributes are
    1. colors: ['yellow', 'yellow', 'yellow'], ['yellow', 'yellow', 'yellow'], ['yellow', 'yellow', 'yellow']
    2. numbers: ['none', 'none', 'none'], ['none', 'none', 'none'], ['none', 'none', 'none']
    3. sizes: ['small', '?', 'large'], ['small', 'medium', 'large'], ['small', 'medium', 'large']
    4. shapes: ['circle', 'circle', 'circle'], ['circle', 'circle', 'circle'], ['circle', 'circle', 'circle']
    """

    agent_contexts.append([
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": perception,
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{example_image_2}"
                    }
                },
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": example_prompt_2,
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Now the given image is",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                },
            ],
        },
    ])
    completion_1 = mad.generate_answer(agent_contexts[0], model_name)
    visual_response_1 = mad.construct_assistant_message(completion_1)
    agent_contexts[0].append(visual_response_1)

    print(f"Agent 1 (Visual Perception) Response:\n{visual_response_1['content']}\n")

    # Agent 2: Inductive Reasoning
    inductive_reasoning_prompt = f"""
            You are an expert in pattern analysis. 
            Based on the visual perception and the image,
            your task is to observe and analyze the given patterns of attributes in the image to extract any rules or relationships present.
            Do not determine the correct answer. Simply identify and describe the pattern.

            Visual Perception: {visual_response_1['content']}

            Output format:
            - Observed Pattern: Focus on attributes that differ in each object.
            - Hypothesis: State a general rule that explains the observed pattern.

            Let's think step by step.
            """

    agent_contexts.append([
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": inductive_reasoning_prompt,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    },
                },
            ],
        }, ]
    )
    completion_2 = mad.generate_answer(agent_contexts[1], model_name)
    inductive_response = mad.construct_assistant_message(completion_2)
    agent_contexts[1].append(inductive_response)

    print(f"Agent 2 (Inductive Reasoning) Response:\n{inductive_response['content']}\n")

    # Agent 3: Deductive Reasoning
    deductive_reasoning_prompt = f"""
            You are a deductive reasoning expert. 

            Visual Perception: {visual_response_1['content']}
            Inductive Reasoning: {inductive_response['content']}
            Question: {sample.prompt}

            Based on the patterns in 1. Visual Perception and in 2. Inductive Reasoning and the provided image, the answer should be:
            Make sure to state your answer at the end of the response.
            Let's think step by step.
            """

    agent_contexts.append(
        [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": deductive_reasoning_prompt,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    },
                },
            ],
        },
        ]
    )
    completion_3 = mad.generate_answer(agent_contexts[2], model_name)
    deductive_response = mad.construct_assistant_message(completion_3)
    agent_contexts[2].append(deductive_response)

    print(f"Agent 3 (Deductive Reasoning) Response:\n{deductive_response['content']}\n")

    final_answer = prompter.get_answer(deductive_response['content'], sample.options)
    pred = final_answer
    raw_output = deductive_response['content']
    print("Final Answer:", final_answer)

    return pred, raw_output


def evaluate_multi_choice(
        data_path: str,
        image_dir: str = "data",
        prompt_name: str = "cot_multi_extract",
        output_dir: str = "outputs_sequential_2",
        **kwargs,
):
    print(locals())
    data = Data.load_with_image_dir(data_path, image_dir)
    model_name = kwargs.get("model_name")
    path_out = f"{output_dir}/{Path(data_path).stem}/{model_name}/{prompt_name}.jsonl"
    print(dict(path_out=path_out))

    is_correct = []
    progress = tqdm(data.samples, desc=path_out)
    sample: Sample
    prompter = select_prompter(prompt_name)
    scorer = ExactScorer()
    model = select_model(**kwargs)

    for sample in progress:
        sample.prompt = prompter.base_prompter.run(sample)

        image_path = f"data/{sample.image}"
        base64_image = encode_image(image_path)
        final_answer = None

        # Number of attributes
        attributes = get_attributes(base64_image, model_name)


        # Dynamic
        if len(attributes) == 1:
            sample.pred, sample.raw_output = single_agent(model_name, sample, prompter, model)
        else:
            sample.pred, sample.raw_output = sequential_reasoning(base64_image, model_name, sample, prompter)


        # fixed: scoring
        is_correct.append(scorer.run(sample))
        score = sum(is_correct) / len(is_correct)
        progress.set_postfix(score=score)
        print(sample.json(indent=2, exclude={"image_string"}))
        print(dict(is_correct=is_correct[-1]))
        data.save(path_out)



def print_results(*paths: str):
    scorer = ExactScorer()
    mapping = dict(
        triangle="numbers",
        grid_number="numbers",
        color_grid="colors",
        color_hexagon="colors",
        shape_reflect="shapes",
        shape_morph="shapes",
        size_cycle="size",
        size_grid="size",
        color_number_hexagon="colors-numbers",
        grid_number_color="colors-numbers",
        venn="numbers-shapes",
        polygon_sides_number="numbers-shapes",
        circle_size_number="numbers-size",
        rectangle_height_number="numbers-size",
        polygon_sides_color="colors-shapes",
        color_overlap_squares="colors-shapes",
        rectangle_height_color="colors-size",
        color_size_circle="colors-size",
        shape_size_hexagon="size-shapes",
        shape_size_grid="size-shapes",
    )

    records = {}
    for p in paths:
        task, model, prompt = Path(p).parts[-3:]
        data = Data.load(str(p))
        score = sum(scorer.run(s) for s in data.samples) / len(data.samples) * 100
        if any(s.pred == "" for s in data.samples):
            score = -1

        base = 25 if len(data.samples[0].options) == 4 else 100 / 3
        records.setdefault(task, {}).update(
            concepts=mapping[task], task=task, random_baseline=base
        )
        records[task][model] = score

    print("Individual task results")
    df = pd.DataFrame(list(records.values()))
    df["length"] = df["concepts"].str.len()
    df = df.sort_values(by=["length"]).drop(columns=["length"]).reset_index(drop=True)
    df.loc[len(df)] = df.mean(numeric_only=True)
    print(df.round(1))

    print("Single-concept results")
    df_single = df.dropna()[~df.dropna()["concepts"].str.contains("-")]
    average_row = df_single.mean(numeric_only=True)
    average_row["concepts"] = "~avg."
    df_single.loc[len(df_single)] = average_row
    df_single = df_single.groupby("concepts").mean(numeric_only=True)
    print(df_single.round(1))

    print("Dual-concept results")
    df_single = df.dropna()[df.dropna()["concepts"].str.contains("-")]
    average_row = df_single.mean(numeric_only=True)
    average_row["concepts"] = "~avg."
    df_single.loc[len(df_single)] = average_row
    df_single = df_single.groupby("concepts").mean(numeric_only=True)
    print(df_single.round(1))


"""
python main.py print_results outputs/*/*/*.jsonl
"""

if __name__ == "__main__":
    Fire()


