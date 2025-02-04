import base64
import io
import json
import random
from pathlib import Path
from typing import List, Tuple

import requests
from PIL import Image
from fire import Fire
from pydantic import BaseModel
from tqdm import tqdm

Point = Tuple[float, float]


def convert_image_to_text(image: Image) -> str:
    # This is also how OpenAI encodes images: https://platform.openai.com/docs/guides/vision
    with io.BytesIO() as output:
        image.save(output, format="PNG")
        data = output.getvalue()
    return base64.b64encode(data).decode("utf-8")


def convert_image_to_bytes(image: Image) -> bytes:
    with io.BytesIO() as output:
        image.save(output, format="PNG")
        data = output.getvalue()
    return data


def convert_text_to_image(text: str) -> Image:
    data = base64.b64decode(text.encode("utf-8"))
    return Image.open(io.BytesIO(data))


def load_image(path: str) -> Image:
    if Path(path).exists():
        return Image.open(path)

    response = requests.get(path)
    return Image.open(io.BytesIO(response.content))


def sample_options(answer: str, options: List[str], k: int):
    # Ensure random order and no duplicates
    options = [o for o in options if o != answer]
    assert len(options) + 1 >= k
    options = random.sample(options, k=k - 1)
    options.append(answer)
    assert len(set(options)) == k
    return random.sample(options, k=k)


class Sample(BaseModel):
    question: str
    answer: str
    options: List[str] = []
    image: str
    image_string: str = ""
    caption: str = ""
    explanation: str = ""
    deduction: str = ""
    prompt: str = ""
    raw_output: str = ""
    pred: str = ""
    correct: int = -1


class Data(BaseModel):
    samples: List

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for s in self.samples:
                print(s.json(), file=f)

    @classmethod
    def load(cls, path: str):
        samples = []
        with open(path) as f:
            for line in f:
                samples.append(Sample(**json.loads(line)))
        print(dict(path=path, samples=len(samples)))
        return cls(samples=samples)

    @classmethod
    def load_with_image_dir(cls, path: str, image_dir: str):
        data = cls.load(path)
        for s in tqdm(data.samples, desc=path):
            path_image = Path(image_dir, s.image)
            image = Image.open(path_image)
            s.image_string = convert_image_to_text(image)
        return data

    def analyze(self):
        for s in random.sample(self.samples, k=4):
            s = s.copy(deep=True)
            s.image_string = s.image_string[:80] + "..."
            print(s.json(indent=2))
        for s in self.samples:
            assert "..." not in s.image_string and len(s.image_string) > 100
        info = dict(
            samples=len(self.samples),
            unique_samples=len(set(s.json() for s in self.samples)),
        )
        print(json.dumps(info, indent=2))


def test_data(**kwargs):
    data = Data.load_with_image_dir(**kwargs)
    data.analyze()


if __name__ == "__main__":
    Fire()
