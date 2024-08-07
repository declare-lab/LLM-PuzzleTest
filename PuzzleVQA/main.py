from pathlib import Path

import pandas as pd
from fire import Fire
from pydantic import BaseModel
from tqdm import tqdm

from data_loading import Data, Sample, convert_text_to_image
from modeling import select_model
from prompting import select_prompter


class Scorer(BaseModel):
    def run(self, sample: Sample) -> float:
        raise NotImplementedError


class ExactScorer(Scorer):
    def run(self, sample: Sample) -> float:
        if sample.pred == sample.answer:
            return 1.0
        return 0.0


def evaluate_multi_choice(
    data_path: str,
    image_dir: str = "data",
    prompt_name: str = "cot_multi_extract",
    output_dir: str = "outputs",
    prevent_direct_answer: bool = False,
    use_describe_image_prompt: bool = True,
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

    # GPT-4V sometimes becomes very lazy when prompted not to directly give the final answer
    if not prevent_direct_answer:
        prompter.base_prompter.prevent_direct_answer = False
    if not use_describe_image_prompt:
        prompter.base_prompter.use_describe_image_prompt = False

    for sample in progress:
        sample.prompt = prompter.base_prompter.run(sample)

        if "qwen" in model_name:
            image = sample.image
        else:
            image = convert_text_to_image(sample.image_string)

        sample.raw_output = model.run(sample.prompt, image)
        sample.pred = prompter.get_answer(sample.raw_output, sample.options)

        # Model-based extraction if prediction not valid
        if sample.pred not in sample.options:
            sample.prompt = prompter.run(sample)
            sample.raw_output = model.run(sample.prompt, image)
            sample.pred = prompter.get_answer(sample.raw_output, sample.options)

        # Scoring
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
