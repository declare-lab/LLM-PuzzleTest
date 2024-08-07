import json
import os
import time
import torch
import anthropic
import botocore
import boto3
from PIL import Image
from dotenv import load_dotenv
from fire import Fire
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional, List
import google.generativeai as genai
from data_loading import convert_image_to_text, convert_image_to_bytes, load_image
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    LlavaProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
)


class EvalModel(BaseModel, arbitrary_types_allowed=True):
    model_path: str
    temperature: float = 0.0
    max_image_size: int = 1024

    def resize_image(self, image: Image) -> Image:
        h, w = image.size
        if h <= self.max_image_size and w <= self.max_image_size:
            return image

        factor = self.max_image_size / max(h, w)
        h = round(h * factor)
        w = round(w * factor)
        print(dict(old=image.size, resized=(h, w)))
        if image.mode == "RGBA":
            image = image.convert("RGB")
        image = image.resize((h, w), Image.LANCZOS)
        return image

    def run(self, prompt: str, image: Image = None) -> str:
        raise NotImplementedError


class GeminiModel(EvalModel):
    model_path: str = "gemini_info.json"
    timeout: int = 60
    model: Optional[genai.GenerativeModel]

    def load(self):
        if self.model is None:
            with open(self.model_path) as f:
                info = json.load(f)
                genai.configure(api_key=info["key"])
                self.model = genai.GenerativeModel(info["engine"])

    def run(self, prompt: str, image: Image = None) -> str:
        self.load()
        output = ""
        config = genai.types.GenerationConfig(
            candidate_count=1,
            temperature=self.temperature,
        )

        while not output:
            try:
                inputs = prompt if image is None else [prompt, self.resize_image(image)]
                response = self.model.generate_content(inputs, generation_config=config)
                if "block_reason" in str(vars(response)):
                    output = str(vars(response))
                elif not response.parts:
                    output = "Empty response.parts from gemini"
                else:
                    output = response.text
            except Exception as e:
                print(e)

            if not output:
                print("Model request failed, retrying.")
                time.sleep(1)

        return output


class GeminiVisionModel(GeminiModel):
    model_path = "gemini_vision_info.json"


class GeminiProVisionModel(GeminiModel):
    engine: str = "gemini-pro-vision"

    def load(self):
        if self.model is None:
            load_dotenv()
            genai.configure(api_key=os.environ["GEMINI_KEY"])
            self.model = genai.GenerativeModel(self.engine)


class GeminiProVisionNewModel(GeminiModel):
    engine: str = "gemini-1.5-pro"

    def load(self):
        if self.model is None:
            load_dotenv()
            genai.configure(api_key=os.environ["GEMINI_KEY"])
            self.model = genai.GenerativeModel(self.engine)


class OpenAIModel(EvalModel):
    model_path: str = "openai_info.json"
    timeout: int = 60
    engine: str = ""
    client: Optional[OpenAI]

    def load(self):
        with open(self.model_path) as f:
            info = json.load(f)
            self.engine = info["engine"]
            self.client = OpenAI(api_key=info["key"], timeout=self.timeout)

    def make_messages(self, prompt: str, image: Image = None) -> List[dict]:
        inputs = [{"type": "text", "text": prompt}]
        if image is not None:
            image_text = convert_image_to_text(self.resize_image(image))
            url = f"data:image/png;base64,{image_text}"
            inputs.append({"type": "image_url", "image_url": {"url": url}})

        return [{"role": "user", "content": inputs}]

    def run(self, prompt: str, image: Image = None) -> str:
        self.load()
        output = ""
        error_message = "The response was filtered"

        while not output:
            try:
                response = self.client.chat.completions.create(
                    model=self.engine,
                    messages=self.make_messages(prompt, image),
                    temperature=self.temperature,
                    max_tokens=512,
                )
                if response.choices[0].finish_reason == "content_filter":
                    raise ValueError(error_message)
                output = response.choices[0].message.content

            except Exception as e:
                print(e)
                if error_message in str(e):
                    output = error_message

            if not output:
                print("OpenAIModel request failed, retrying.")

        return output

    def run_few_shot(self, prompts: List[str], images: List[Image.Image]) -> str:
        self.load()
        output = ""
        error_message = "The response was filtered"
        content = []
        for i, p in enumerate(prompts):
            for value in self.make_messages(p, images[i])[0]["content"]:
                content.append(value)

        while not output:
            try:
                response = self.client.chat.completions.create(
                    model=self.engine,
                    messages=[{"role": "user", "content": content}],
                    temperature=self.temperature,
                    max_tokens=512,
                )
                if response.choices[0].finish_reason == "content_filter":
                    raise ValueError(error_message)
                output = response.choices[0].message.content

            except Exception as e:
                print(e)
                if error_message in str(e):
                    output = error_message

            if not output:
                print("OpenAIModel request failed, retrying.")

        return output


class OpenAIVisionModel(OpenAIModel):
    model_path = "openai_vision_info.json"


class GPT4VModel(OpenAIModel):
    engine: str = "gpt-4-vision-preview"

    def load(self):
        if self.client is None:
            load_dotenv()
            key = os.environ["OPENAI_KEY"]
            self.client = OpenAI(api_key=key, timeout=self.timeout)


class GPT4oModel(OpenAIModel):
    engine: str = "gpt-4o-2024-05-13"

    def load(self):
        if self.client is None:
            load_dotenv()
            key = os.environ["OPENAI_KEY"]
            self.client = OpenAI(api_key=key, timeout=self.timeout)


class LlavaModel(EvalModel):
    model_path = "llava-hf/llava-1.5-13b-hf"
    template = "USER: <image>\n{prompt}\nASSISTANT:"
    device: str = "cuda"
    dtype: torch.dtype = torch.float16
    model: Optional[LlavaForConditionalGeneration] = None
    processor: Optional[LlavaProcessor] = None

    def load(self):
        if self.model is None:
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=self.dtype,
            ).to(self.device)
            self.processor = AutoProcessor.from_pretrained(self.model_path)

    def run(self, prompt: str, image: Image = None) -> str:
        self.load()
        prompt = self.template.format(prompt=prompt)
        if image is not None:
            image = self.resize_image(image)

        # noinspection PyTypeChecker
        inputs = self.processor(prompt, image, return_tensors="pt").to(
            self.device, self.dtype
        )
        prompt_length = inputs["input_ids"].shape[1]

        outputs = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)[0]
        return self.processor.decode(outputs[prompt_length:], skip_special_tokens=True)


class ClaudeModel(EvalModel):
    model_path: str = "claude_info.json"
    timeout: int = 60
    engine: str = ""
    client: Optional[anthropic.Anthropic]

    def load(self):
        with open(self.model_path) as f:
            info = json.load(f)
            self.engine = info["engine"]
            self.client = anthropic.Anthropic(api_key=info["key"], timeout=self.timeout)

    def make_messages(self, prompt: str, image: Image = None) -> List[dict]:
        image_media_type = "image/png"
        image_data = convert_image_to_text(self.resize_image(image))

        inputs = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image_media_type,
                    "data": image_data,
                },
            },
            {"type": "text", "text": prompt},
        ]

        return [{"role": "user", "content": inputs}]

    def run(self, prompt: str, image: Image = None) -> str:
        self.load()
        output = ""
        error_message = "The response was filtered"

        while not output:
            try:
                response = self.client.messages.create(
                    model=self.engine,
                    messages=self.make_messages(prompt, image),
                    temperature=self.temperature,
                    max_tokens=512,
                )

                output = response.content[0].text

            except Exception as e:
                print(e)
                if error_message in str(e):
                    output = error_message

            if not output:
                print("ClaudeModel request failed, retrying.")

        return output


class ClaudeOpusModel(ClaudeModel):
    engine: str = "claude-3-opus-20240229"

    def load(self):
        if self.client is None:
            load_dotenv()
            key = os.environ["CLAUDE_KEY"]
            self.client = anthropic.Anthropic(api_key=key, timeout=self.timeout)


class ClaudeSonnetNewModel(ClaudeModel):
    engine: str = "claude-3-5-sonnet-20240620"

    def load(self):
        if self.client is None:
            load_dotenv()
            key = os.environ["CLAUDE_KEY"]
            self.client = anthropic.Anthropic(api_key=key, timeout=self.timeout)


class QwenModel(EvalModel):
    model_path = "Qwen/Qwen-VL-Chat"
    template = "USER: <image>\n{prompt}\nASSISTANT:"
    device: str = "cuda"
    dtype: torch.dtype = torch.float16
    model: Optional[AutoModelForCausalLM] = None
    tokenizer: Optional[AutoTokenizer] = None

    def load(self):
        if self.model is None:
            self.model = (
                AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map="cuda",
                    trust_remote_code=True,
                    bf16=True,
                )
                .eval()
                .to(self.device)
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )

    def run(self, prompt: str, image: str) -> str:
        self.load()
        prompt = self.template.format(prompt=prompt)

        query = self.tokenizer.from_list_format(
            [
                {"image": f"data/{image}"},
                {"text": prompt},
            ]
        )

        response, _ = self.model.chat(self.tokenizer, query=query, history=None)
        return response


class BedrockModel(EvalModel):
    model_path: str = "bedrock_info.json"
    engine: str = ""
    client: Optional[botocore.client.BaseClient]

    def load(self):
        with open(self.model_path) as f:
            info = json.load(f)
            self.engine = info["engine"]
            self.client = boto3.client('bedrock-runtime')

    def make_messages(self, prompt: str, image: Image = None) -> List[dict]:
        image_media_type = "png"
        image_data = convert_image_to_bytes(self.resize_image(image))

        inputs = [
            {
                "image": {
                    "format": image_media_type,
                    "source": {
                        "bytes": image_data
                    }
                },
            },
            {
                "text": prompt
            },
        ]

        return [{"role": "user", "content": inputs}]

    def run(self, prompt: str, image: Image = None) -> str:
        self.load()
        output = ""
        error_message = "The response was filtered"

        while not output:
            try:
                response = self.client.converse(
                    modelId=self.engine,
                    messages=self.make_messages(prompt, image),
                    inferenceConfig={
                        "temperature": self.temperature,
                        "maxTokens": 512
                    }
                )
                output = response['output']['message']['content'][0]['text']
            except Exception as e:
                print(e)
                if error_message in str(e):
                    output = error_message

            if not output:
                print("BedrockModel request failed, retrying...")

        return output


def select_model(model_name: str, **kwargs) -> EvalModel:
    model_map = dict(
        gemini_vision=GeminiVisionModel,
        openai_vision=OpenAIVisionModel,
        llava=LlavaModel,
        claude=ClaudeModel,
        qwen=QwenModel,
        gpt4v=GPT4VModel,
        gpt4o=GPT4oModel,
        claude_3_opus=ClaudeOpusModel,
        claude_35_sonnet=ClaudeSonnetNewModel,
        gemini_1_pro=GeminiProVisionModel,
        gemini_15_pro=GeminiProVisionNewModel,
        bedrock=BedrockModel,
    )
    model_class = model_map.get(model_name)
    if model_class is None:
        raise ValueError(f"{model_name}. Choose from {list(model_map.keys())}")
    return model_class(**kwargs)


def test_model(
    prompt: str = "Can you describe the image in detail?",
    image_path: str = "https://www.thesprucecrafts.com/thmb/H-VsgPFaCjTnQQ6u0fLt6X6v3Ic=/750x0/filters:no_upscale():max_bytes(150000):strip_icc()/Chesspieces-GettyImages-667749253-59c339e9685fbe001167cdce.jpg",
    model_name: str = "gemini_vision",
    **kwargs,
):
    model = select_model(model_name, **kwargs)
    print(locals())
    print(model.run(prompt, load_image(image_path)))


"""
bash evaluate.sh gpt4v
bash evaluate.sh claude_3_opus
bash evaluate.sh gemini_1_pro
bash evaluate.sh gpt4o
bash evaluate.sh claude_35_sonnet
bash evaluate.sh gemini_15_pro
"""


if __name__ == "__main__":
    Fire()
