import json

"""
Calculate and save the success rates of PuzzleVQA experiments
"""

def calculate_success_rate(output_dir, puzzle_name, model_name, prompt_name):
    file_path = f"{output_dir}/{puzzle_name}/{model_name}/{prompt_name}.jsonl"
    success_count = 0
    total_count = 0

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            if data['pred'] == data['answer']:
                success_count += 1
            total_count += 1

    success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
    return success_rate


def save_results(output_dir, puzzle_name, model_name, prompt_name, success_rate):
    file_path = f"{output_dir}/{model_name}_{prompt_name}_result.txt"

    results = \
    f'''
    Puzzle name: {puzzle_name}
    Success rate: Success Rate: {success_rate:.4f}%
    
    '''
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(results)


if __name__ == '__main__':

    output_dir = "outputs"
    puzzle_name = "triangle"
    model_name = "gpt4o"
    prompt_name = "cot_multi_extract"

    success_rate = calculate_success_rate(output_dir, puzzle_name, model_name, prompt_name)
    save_results(output_dir, puzzle_name, model_name, prompt_name, success_rate)

