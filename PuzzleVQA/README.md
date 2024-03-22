## Setup

```
conda create -n abstract python=3.10 -y
conda activate abstract
pip install -r requirements.txt
```

## Dataset

The data for the puzzles is available in the [data](https://github.com/declare-lab/puzzle-reasoning/tree/master/PuzzleVQA/data) directory. Each json file contains 100 instances of each puzzle. Each line in the jsons correspond to a single puzzle instance. Each line has the following structure:

```json
{
    "image": "path/to/image/file.png", 
    "question": "information about the puzzle and the question", 
    "options": ["option1", "option2", "option3", "option4"], 
    "answer": "correct option", 
    "caption": "caption about the image",
    "explanation": "explanation of the pattern in the image",
    "deduction": "deduction statement of applying the pattern to derive final answer"
}
```

## Dataset Generation

The scripts for generating the puzzles is available in the [generation](https://github.com/declare-lab/puzzle-reasoning/tree/master/PuzzleVQA/generation) directory. The scripts can be used as follows:

```bash
cd generation

python data_generation.py create_data <puzzle_name>
```

### Current List of Puzzle Names
- `circle_size_number`
- `color_grid`
- `color_hexagon`
- `color_number_hexagon`
- `color_overlap_squares`
- `color_size_circle`
- `grid_number_color`
- `grid_number`
- `polygon_sides_color`
- `polygon_sides_number`
- `rectangle_height_color`
- `rectangle_height_number`
- `shape_morph`
- `shape_reflect`
- `shape_size_grid`
- `shape_size_hexagon`
- `size_cycle`
- `size_grid`
- `triangle`
- `venn`

## Model Evaluation

Run zero-shot evaluation with LLMs like [Gemini Pro](https://ai.google.dev/tutorials/python_quickstart?hl=en) or [GPT-4(V)](https://platform.openai.com/docs/guides/vision) or [Claude 3 Opus](https://docs.anthropic.com/claude/docs/vision) 

### Example to run evalution on "triangle" puzzle with Gemini Pro 
```bash
python main.py evaluate_multi_choice data/triangle.json \
--model_name gemini_vision \
--prompt_name cot_multi_extract \
```

### Supported Models
- `gemini_vision`
- `openai_vision`
- `claude`

### Supported Prompts
- `cot_multi_extract`
- `cot_caption_multi_extract`
- `cot_caption_explanation_multi_extract`
- `cot_caption_explanation_deduction_multi_extract`

## API Setup

Gemini Pro (multimodal): Please create a file named `gemini_vision_info.json`

```
{"engine": "gemini-pro-vision", "key": "your_api_key"}
```

GPT-4V (multimodal): Please create a file named `openai_vision_info.json`

```
{"engine": "gpt-4-vision-preview", "key": "your_api_key"}
```

Claude 3 Opus (multimodal): Please create a file named `claude_info.json`

```
{"engine": "claude-3-opus-20240229", "key": "your_api_key"}
```