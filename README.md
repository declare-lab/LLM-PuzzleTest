### **Setup**

```
conda create -n puzzle python=3.10 -y
conda activate puzzle
pip install -r requirements.txt
```
- As in [PuzzleVQA/README.md](https://github.com/Minseo10/LLM-PuzzleTest/tree/master/PuzzleVQA#readme)

### Openai api key
Put your openai api key in
- [PuzzleVQA/.env](https://github.com/Minseo10/LLM-PuzzleTest/blob/master/PuzzleVQA/.env): `OPENAI_KEY=your_api_key` 
- [PuzzleVQA/key.json](https://github.com/Minseo10/LLM-PuzzleTest/blob/master/PuzzleVQA/key.json): `"OPENAI_API_KEY": "your_openai_key"` 

### Command
```
cd PuzzleVQA
python main_debate.py evaluate_multi_choice data/데이터셋이름.json --model_name gpt4o --prompt_name cot_multi_extract --num_agents 2 --rounds 3
```
- --model_name: gpt-4o
    - PuzzleVQA 는 논문에서 “gpt-4-vision-preview” 사용, 그런데 openai에서 더 이상 지원 x. 일단 “gpt-4o” 사용
- --prompt_name: cot_multi_extract
- --num_agents: 2
- --rounds: 3
- 데이터셋이름: dataset name

### Output
- Saved in [PuzzleVQA/outputs](https://github.com/Minseo10/LLM-PuzzleTest/tree/master/PuzzleVQA/outputs)