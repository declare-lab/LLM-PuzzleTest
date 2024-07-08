## Setup

```
conda create -n algo python=3.10 -y
conda activate algo
pip install -r requirements.txt
```

## Dataset

The data for the puzzles is available in the [data](https://github.com/declare-lab/puzzle-reasoning/tree/master/AlgoPuzzleVQA/data) directory. Each json file contains 100 instances of each puzzle. Each line in the jsons correspond to a single puzzle instance. Each line has the following structure:

```json
{
    "image": "path/to/image/file.jpg", 
    "question": "information about the puzzle and the question", 
    "options": ["option1", "option2", "option3", "option4"], 
    "answer": "correct option", 
    "solution": {
        "puzzle_specific_information1": "..",
        "puzzle_specific_information2": "..",
        "..": ".." 
    }
}
```

## Dataset Generation

The scripts for generating the puzzles is available in the [generation](https://github.com/declare-lab/puzzle-reasoning/tree/master/AlgoPuzzleVQA/generation) directory. The scripts can be used as follows:

```bash
cd generation
mkdir data
mkdir images

python board_tile.py
python tower_of_hanoi.py
...
```


## Model Evaluation

Run zero-shot evaluation with LLMs like [Gemini Pro](https://ai.google.dev/tutorials/python_quickstart?hl=en) or [GPT-4(V)](https://platform.openai.com/docs/guides/vision) or [Claude 3 Opus](https://docs.anthropic.com/claude/docs/vision) 

The currently supported model names are `[gemini_vision, openai_vision, claude]`

```
# Run evaluation on the "wheel_of_fortune" puzzle with Gemini Pro Vision
python main.py evaluate_multi_choice data/wheel_of_fortune.json --model_name gemini_vision
[13:45<00:00,  8.25s/it, score=0.24]

# Run evaluation on the "wheel_of_fortune" puzzle with GPT-4V
python main.py evaluate_multi_choice data/wheel_of_fortune.json --model_name openai_vision
[31:15<00:00, 18.76s/it, score=0.29]

# Run evaluation on the "wheel_of_fortune" puzzle with LLaVA
python main.py evaluate_multi_choice data/wheel_of_fortune.json --model_name llava
[21:33<00:00, 12.94s/it, score=0.27]

# Run evaluation on the "wheel_of_fortune" puzzle with Claude 3 Opus
python main.py evaluate_multi_choice data/wheel_of_fortune.json --model_name claude
[33:28<00:00, 20.09s/it, score=0.20]

```

## API Setup

Gemini Pro (multimodal): Please create a file named `gemini_vision_info.json`

```json
{"engine": "gemini-pro-vision", "key": "your_api_key"}
```

GPT-4V (multimodal): Please create a file named `openai_vision_info.json`

```json
{"engine": "gpt-4-vision-preview", "key": "your_api_key"}
```

Claude 3 Opus (multimodal): Please create a file named `claude_info.json`

```json
{"engine": "claude-3-opus-20240229", "key": "your_api_key"}
```

Claude 3.5 Sonnet (multimodal) via [Amazon Bedrock](https://aws.amazon.com/bedrock/claude/): Please create a file named `bedrock_info.json`

> 💡 For more information on how to select a default AWS Region and setup AWS credentials, please refer to the [AWS Boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html) (Developer Guide > Credentials).

```json
{"engine": "anthropic.claude-3-5-sonnet-20240620-v1:0"}
```