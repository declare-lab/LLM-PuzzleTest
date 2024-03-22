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
        return f"{sample.caption.rstrip()}\nQ: {sample.question.rstrip()}\nA: Let's think step by step."


class ChainThoughtMultiChoicePrompter(Prompter):
    prevent_direct_answer: bool = True
    use_describe_image_prompt: bool = True

    def run(self, sample: Sample) -> str:
        # Following a similar format as "Large Language Models are Zero-Shot Reasoners"
        size_options = {"small", "medium", "large"}
        assert len(sample.options) == 4 or set(sample.options) == size_options
        assert sample.answer in sample.options

        parts: list[str, Any] = [
            f"Question: {sample.question.rstrip()} Do not directly give the final answer.",
            f"Options:",
            f"(A) {sample.options[0]}",
            f"(B) {sample.options[1]}",
            f"(C) {sample.options[2]}",
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

    def run_train(self, sample: Sample) -> str:
        assert sample.caption
        assert sample.explanation
        assert sample.deduction
        letter = "ABCD"[sample.options.index(sample.answer)]
        return f"{self.run(sample)} {sample.caption} {sample.explanation} {sample.deduction} So the final answer is ({letter}) {sample.answer}."


class ChainThoughtCaptionMultiChoicePrompter(ChainThoughtMultiChoicePrompter):
    def run(self, sample: Sample) -> str:
        prefix = "Question:"
        text = super().run(sample).replace(prefix, "")
        assert sample.caption
        return f"{prefix} {sample.caption}{text}"


class ChainThoughtCaptionExplanationMultiChoicePrompter(
    ChainThoughtMultiChoicePrompter
):
    def run(self, sample: Sample) -> str:
        prefix = "Question:"
        text = super().run(sample).replace(prefix, "")
        assert sample.caption
        assert sample.explanation
        return f"{prefix} {sample.caption} {sample.explanation}{text}"


class ChainThoughtCaptionExplanationDeductionMultiChoicePrompter(
    ChainThoughtMultiChoicePrompter
):
    def run(self, sample: Sample) -> str:
        prefix = "Question:"
        text = super().run(sample).replace(prefix, "")
        assert sample.caption
        assert sample.explanation
        assert sample.deduction
        sep = "should be"
        assert sample.deduction.count(sep) == 1
        return f"{prefix} {sample.caption} {sample.explanation}{text} {sample.deduction.split(sep)[0] + sep}:"


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


class ChainThoughtCaptionExplanationMultiExtractPrompter(
    ChainThoughtMultiExtractPrompter
):
    base_prompter: Prompter = ChainThoughtCaptionExplanationMultiChoicePrompter()


class ChainThoughtCaptionExplanationDeductionMultiExtractPrompter(
    ChainThoughtMultiExtractPrompter
):
    base_prompter: Prompter = (
        ChainThoughtCaptionExplanationDeductionMultiChoicePrompter()
    )


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
    if name == "cot_caption_explanation_multi_extract":
        return ChainThoughtCaptionExplanationMultiExtractPrompter()
    if name == "cot_caption_explanation_deduction_multi_extract":
        return ChainThoughtCaptionExplanationDeductionMultiExtractPrompter()
    raise KeyError(name)
