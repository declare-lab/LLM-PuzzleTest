from pathlib import Path

from fire import Fire
from pydantic import BaseModel
from tqdm import tqdm

from data_loading import Data, Sample, convert_text_to_image
from modeling import select_model
from prompting import select_prompter
from python_executor import PythonExecutor, extract_program


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
    prevent_direct_answer: bool = True,
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
    if (
        "openai" in model.model_path
        or "llava" in model.model_path
        or "claude" in model.model_path
        or not prevent_direct_answer
    ):
        prompter.base_prompter.prevent_direct_answer = False
    if not use_describe_image_prompt:
        prompter.base_prompter.use_describe_image_prompt = False

    if "pot" in prompt_name:
        executor = PythonExecutor(get_answer_from_stdout=True)

    for sample in progress:
        # Initial zero-shot prompting
        sample.prompt = prompter.base_prompter.run(sample)
        print(f"sample.prompt: {sample.prompt}")
        image = convert_text_to_image(sample.image_string)
        sample.raw_output = model.run(sample.prompt, image)
        print(f"sample.raw_output: {sample.raw_output}")

        if "pot" in prompt_name:
            program = extract_program(sample.raw_output)
            prediction = executor.apply(program)
            sample.raw_output += f"\nProgram Output: {prediction[0]}\n"
            print("start" + "#" * 100)
            print(sample.raw_output)
            print("end" + "#" * 100)

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


"""
p main.py evaluate_multi_choice data/wheel_of_fortune.json --model_name gemini_vision
p main.py evaluate_multi_choice data/wheel_of_fortune.json --model_name openai_vision
p main.py evaluate_multi_choice data/wheel_of_fortune.json --model_name claude
"""


if __name__ == "__main__":
    Fire()
