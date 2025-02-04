import json
from PIL import Image
from fire import Fire
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional, List
from data_loading import convert_image_to_text, load_image
from openai import AzureOpenAI


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
        if image.mode == "RGBA":
            image = image.convert("RGB")
        image = image.resize((h, w), Image.LANCZOS)
        return image

    def run(self, prompt: str, image: Image = None) -> str:
        raise NotImplementedError


class GPTModel(EvalModel):
    model_path: str = ""
    timeout: int = 60
    engine: str = ""
    client: Optional[OpenAI]

    def load(self):
        with open(self.model_path) as f:
            info = json.load(f)
            self.engine = info["engine"]
            self.client = AzureOpenAI(
                azure_endpoint=info["endpoint"],
                api_key=info["key"],
                api_version=info["api_version"],
            )

    def make_messages(self, prompt: str, image: Image = None) -> List[dict]:
        inputs = [{"type": "text", "text": prompt}]

        if image:
            image_text = convert_image_to_text(self.resize_image(image))
            url = f"data:image/jpeg;base64,{image_text}"
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
                    max_tokens=1024,
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


class GPT4oModel(GPTModel):
    model_path: str = "gpt4o.json"


class GPT4tModel(GPTModel):
    model_path: str = "gpt4t.json"


class O1Model(GPTModel):
    model_path = "o1-full-high.json"
    reasoning_effort: str = ""

    def load(self):
        with open(self.model_path) as f:
            info = json.load(f)
            self.engine = info["engine"]
            self.reasoning_effort = info["reasoning_effort"]
            self.client = AzureOpenAI(
                azure_endpoint=info["endpoint"],
                api_key=info["key"],
                api_version="2024-12-01-preview",
            )

    def run(self, prompt: str, image: Image = None) -> str:
        self.load()
        output = ""
        error_message = "The response was filtered"

        while not output:
            try:
                response = self.client.chat.completions.create(
                    model=self.engine,
                    messages=self.make_messages(prompt, image),
                    reasoning_effort=self.reasoning_effort,
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


def select_model(model_name: str, **kwargs) -> EvalModel:
    model_map = dict(
        o1=O1Model,
        gpt4o=GPT4oModel,
        gpt4t=GPT4tModel,
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
python modeling.py test_model --model_name o1
python modeling.py test_model --model_name gpt4t
python modeling.py test_model --model_name gpt4v
python modeling.py test_model --model_name gpt4o
"""


if __name__ == "__main__":
    Fire()
