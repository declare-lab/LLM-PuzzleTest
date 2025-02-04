from pathlib import Path
import string
import re

from fire import Fire
from pydantic import BaseModel
from tqdm import tqdm

from data_loading import Data, convert_text_to_image
from modeling import select_model
from prompting import FullPrompter


class Scorer(BaseModel):
    def run(self, sample) -> float:
        raise NotImplementedError


class MCQScorer(Scorer):
    def run(self, sample) -> float:
        punctuations = string.punctuation.replace(":", "")
        matches = re.findall(r"[ABCD]", sample.pred)

        mapping = {0: "A", 1: "B", 2: "C", 3: "D"}
        answer_option = mapping[sample.options.index(sample.answer)]

        if matches and len(matches) == 1:
            if matches[-1] == answer_option:
                return 1.0

        if (
            sample.answer.lower()
            in sample.pred.translate(str.maketrans("", "", punctuations)).lower()
        ):
            return 1.0

        if sample.pred.lower() == sample.answer.lower():
            return 1.0
        return 0.0


class OpenEndedScorer(Scorer):
    def run(self, sample) -> float:
        punctuations = string.punctuation.replace(":", "")

        if (
            sample.answer.lower()
            in sample.pred.translate(str.maketrans("", "", punctuations))
            .lower()
            .split()
        ):
            return 1.0
        return 0.0


class GptScorer(Scorer):
    model = select_model("gpt4o")

    def run(self, sample) -> float:
        input_prompt = f"""
Evaluate the candidate answer against the correct answer. If the candidate answer is correct, output `[correct]`; otherwise, output `[incorrect]`.

Question: {sample.question}
Candidate Answer: {sample.raw_output} 
Correct Answer: {sample.answer}
Evaluation: 
""".strip()

        output = self.model.run(input_prompt)

        print(f"{input_prompt}\n{output}")

        if "[correct]" in output:
            return 1.0
        else:
            return 0.0


def evaluate(
    dataset: str,
    puzzle: str,
    question_type: str,
    output_dir: str = "outputs",
    **kwargs,
):
    print(locals())
    image_dir = f"{dataset}/data"
    data_path = f"{dataset}/data/{puzzle}.json"

    data = Data.load_with_image_dir(data_path, image_dir)
    model_name = kwargs.get("model_name")
    path_out = f"{output_dir}/{dataset}/{question_type}/{model_name}/{puzzle}.jsonl"
    print(dict(path_out=path_out))

    is_correct = []
    progress = tqdm(data.samples, desc=path_out)

    # Get Prompter
    if model_name == "o1":
        cot = False
    else:
        cot = True

    prompter = FullPrompter(question_type=question_type, cot=cot)

    # Get Scorer
    if question_type == "mcq":
        scorer = MCQScorer()

    elif question_type == "open":
        scorer = GptScorer()

    else:
        raise f"Unknown question type: {question_type}"

    model = select_model(**kwargs)

    for sample in progress:
        # Initial zero-shot prompting
        sample.prompt = prompter.base_prompter.run(sample)
        print(f"sample.prompt: {sample.prompt}")
        image = convert_text_to_image(sample.image_string)
        sample.raw_output = model.run(sample.prompt, image)
        print(f"sample.raw_output: {sample.raw_output}")

        if question_type == "mcq":
            sample.pred = prompter.get_answer(sample.raw_output, sample.options)

            # Model-based extraction if prediction not valid
            if sample.pred not in sample.options:
                sample.prompt = prompter.run(sample)
                sample.raw_output = model.run(sample.prompt, image)
                sample.pred = prompter.get_answer(sample.raw_output, sample.options)

        elif question_type == "open":
            pass

        else:
            raise f"Unknown question type: {question_type}"

        # Scoring
        score = scorer.run(sample)

        sample.correct = score
        is_correct.append(score)
        score = sum(is_correct) / len(is_correct)
        progress.set_postfix(score=score)
        print(sample.json(indent=2, exclude={"image_string"}))
        print(dict(is_correct=is_correct[-1]))
        data.save(path_out)


"""
python main.py evaluate --dataset PuzzleVQA --puzzle color_hexagon --question_type open --model_name gpt4o --output_dir outputs_test
python main.py evaluate --dataset PuzzleVQA --puzzle color_hexagon --question_type mcq --model_name gpt4o --output_dir outputs_test
python main.py evaluate --dataset AlgoPuzzleVQA --puzzle board_tile --question_type open --model_name gpt4o --output_dir outputs_test
python main.py evaluate --dataset AlgoPuzzleVQA --puzzle board_tile --question_type mcq --model_name gpt4o --output_dir outputs_test
"""


if __name__ == "__main__":
    Fire()
