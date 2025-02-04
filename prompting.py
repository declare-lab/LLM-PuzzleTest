import re
from typing import List


class Prompter:
    def run(self, sample) -> str:
        raise NotImplementedError


class BasePrompter(Prompter):
    def __init__(self, question_type: str, cot: str):
        self.question_type = question_type
        self.cot = cot

    def run(self, sample) -> str:
        # Following a similar format as "Large Language Models are Zero-Shot Reasoners"
        size_options = {"small", "medium", "large"}
        binary_options = {"Yes", "No"}
        assert (
            len(sample.options) == 4
            or set(sample.options) == size_options
            or set(sample.options) == binary_options
        )
        assert sample.answer in sample.options

        parts = [f"Question: {sample.question.rstrip()}"]
        if self.question_type == "mcq":
            parts.append("Options:")
            for i, alphabet in enumerate(["(A)", "(B)", "(C)", "(D)"]):
                if i == len(sample.options):
                    break
                parts.append(f"{alphabet} {sample.options[i]}")

        parts.append("")
        if self.cot:
            parts.append("Answer: Let's think step by step.")
        else:
            parts.append("Answer:")

        return "\n".join(parts)


class FullPrompter(Prompter):
    def __init__(self, question_type: str, cot: str):
        self.question_type = question_type
        self.cot = cot
        self.base_prompter = BasePrompter(question_type=question_type, cot=cot)

    def run(self, sample) -> str:
        parts = [
            self.base_prompter.run(sample),
            sample.raw_output,
            "",
        ]

        if self.question_type == "mcq":
            options = ""
            for i, alphabet in enumerate(["(A)", "(B)", "(C)", "(D)"]):
                options += f" {alphabet}"
                if i == len(sample.options):
                    break
            parts.append(f"Therefore, among {options}, the answer is:")

        elif self.question_type == "open":
            raise "Open ended shouldn't have extraction."

        else:
            raise ValueError(f"Unknown question type: {self.question_type}")

        return "\n".join(parts)

    def get_answer(self, text: str, options: List[str]):
        if self.question_type == "mcq":
            mapping = {letter: o for letter, o in zip("ABCD", options)}
            matches = re.findall(r"\(([ABCD])\)", text)
            if matches and len(matches) == 1:
                return mapping.get(matches[-1], options[0])

            return f"{text}"

        elif self.question_type == "open":
            return text.split(".")[0]

        else:
            raise ValueError(f"Unknown question type: {self.question_type}")
