import re
from typing import List, Any

from pydantic import BaseModel

from data_loading import Sample


class Prompter(BaseModel):
    def run(self, sample: Sample) -> str:
        raise NotImplementedError


class StandardPrompter(Prompter):
    def run(self, sample: Sample) -> str:
        return f"Q: {sample.question.rstrip()}\nA:"


class ChainThoughtPrompter(Prompter):
    def run(self, sample: Sample) -> str:
        return f"Q: {sample.question.rstrip()}\nA: Let's think step by step."


class ChainThoughtWithCaptionPrompter(Prompter):
    def run(self, sample: Sample) -> str:
        return f"{sample.image_caption.rstrip()}\nQ: {sample.question.rstrip()}\nA: Let's think step by step."


class ChainThoughtMultiChoicePrompter(Prompter):
    prevent_direct_answer: bool = True
    use_describe_image_prompt: bool = True

    def run(self, sample: Sample) -> str:
        # Following a similar format as "Large Language Models are Zero-Shot Reasoners"
        size_options = {"small", "medium", "large"}
        binary_options = {"Yes", "No"}
        assert (
            len(sample.options) == 4
            or set(sample.options) == size_options
            or set(sample.options) == binary_options
        )
        assert sample.answer in sample.options

        parts: list[str, Any] = [
            f"Question: {sample.question.rstrip()} Do not directly give the final answer.",
            f"Options:",
            f"(A) {sample.options[0]}",
            f"(B) {sample.options[1]}",
            f"(C) {sample.options[2]}" if len(sample.options) >= 3 else "",
            f"(D) {sample.options[3]}" if len(sample.options) == 4 else "",
            "",
            f"Answer: Let's describe the image first and think step by step.",
        ]

        if set(sample.options) == size_options:
            parts.pop(-3)
        if not self.prevent_direct_answer:
            parts[0] = parts[0].replace(" Do not directly give the final answer.", "")
        if not self.use_describe_image_prompt:
            parts[-1] = parts[-1].replace(" describe the image first and", "")
        return "\n".join(parts)


class ChainThoughtCaptionMultiChoicePrompter(ChainThoughtMultiChoicePrompter):
    def run(self, sample: Sample) -> str:
        prefix = "Question:"
        text = super().run(sample).replace(prefix, "")
        assert sample.image_caption
        return f"{prefix} {sample.image_caption}{text}"


class ChainThoughtMultiExtractPrompter(Prompter):
    base_prompter: Prompter = ChainThoughtMultiChoicePrompter()

    def run(self, sample: Sample) -> str:
        parts = [
            self.base_prompter.run(sample),
            sample.raw_output,
            "",
            "Therefore, among (A) (B) (C) (D), the answer is:",
        ]

        if set(sample.options) == {"small", "medium", "large"}:
            parts[-1] = parts[-1].replace(" (D)", "")
        return "\n".join(parts)

    @staticmethod
    def get_answer(text: str, options: List[str]):
        mapping = {letter: o for letter, o in zip("ABCD", options)}
        matches = re.findall(r"\(([ABCD])\)", text)
        if matches:
            return mapping.get(matches[-1], options[0])

        matches = re.findall(r"[ABCD]", text)
        if matches:
            return mapping.get(matches[-1], options[0])

        return f"Cannot get_answer: {text}"


class ChainThoughtCaptionMultiExtractPrompter(ChainThoughtMultiExtractPrompter):
    base_prompter: Prompter = ChainThoughtCaptionMultiChoicePrompter()


class ProgramThoughtMultiChoicePrompter(Prompter):
    prevent_direct_answer: bool = True
    use_describe_image_prompt: bool = True

    def run(self, sample: Sample) -> str:
        # Following a similar format as "Large Language Models are Zero-Shot Reasoners"
        size_options = {"small", "medium", "large"}
        binary_options = {"Yes", "No"}
        assert (
            len(sample.options) == 4
            or set(sample.options) == size_options
            or set(sample.options) == binary_options
        )
        assert sample.answer in sample.options

        parts: list[str, Any] = [
            f"Question: {sample.question.rstrip()}",
            f"Options:",
            f"(A) {sample.options[0]}",
            f"(B) {sample.options[1]}",
            f"(C) {sample.options[2]}" if len(sample.options) >= 3 else "",
            f"(D) {sample.options[3]}" if len(sample.options) == 4 else "",
            "",
            f"Answer: Let's write Python code to solve this problem and print out the final value.",
        ]

        if set(sample.options) == size_options:
            parts.pop(-3)
        if not self.prevent_direct_answer:
            parts[0] = parts[0].replace(" Do not directly give the final answer.", "")
        if not self.use_describe_image_prompt:
            parts[-1] = parts[-1].replace(" describe the image first and", "")
        return "\n".join(parts)


class ProgramThoughtMultiExtractPrompter(Prompter):
    base_prompter: Prompter = ProgramThoughtMultiChoicePrompter()

    def run(self, sample: Sample) -> str:
        parts = [
            self.base_prompter.run(sample),
            sample.raw_output,
            "",
            "Therefore, among (A) (B) (C) (D), the answer is:",
        ]

        if set(sample.options) == {"small", "medium", "large"}:
            parts[-1] = parts[-1].replace(" (D)", "")
        return "\n".join(parts)

    @staticmethod
    def get_answer(text: str, options: List[str]):
        mapping = {letter: o for letter, o in zip("ABCD", options)}
        matches = re.findall(r"\(([ABCD])\)", text)
        if matches:
            return mapping.get(matches[-1], options[0])

        matches = re.findall(r"[ABCD]", text)
        if matches:
            return mapping.get(matches[-1], options[0])

        return f"Cannot get_answer: {text}"


def select_prompter(name: str):
    if name == "standard":
        return StandardPrompter()
    if name == "cot":
        return ChainThoughtPrompter()
    if name == "cot_caption":
        return ChainThoughtWithCaptionPrompter()
    if name == "cot_multi_extract":
        return ChainThoughtMultiExtractPrompter()
    if name == "cot_caption_multi_extract":
        return ChainThoughtCaptionMultiExtractPrompter()
    if name == "pot":
        return ProgramThoughtMultiExtractPrompter()
    raise KeyError(name)
