import json

def get_accuracy(result_file_path:str):
    """
    Use result json file to visulize the accuracy.
    """
    k = 0
    line_num = 0
    with open(result_file_path, 'r') as f:
        for line in f:
            # Every line of json
            item = json.loads(line)  
            # Get function description
            passed = item["passed"]
            if passed:
                k += 1
            line_num += 1
    return k/line_num
acc = get_accuracy(result_file_path="/Users/mysteryshack/Downloads/Graduate/Courses/EECS487/prompt-code-487/human-eval/Experiments/Results/gemini-1.5-flash/MetaPrompt_results.jsonl")
print(acc)