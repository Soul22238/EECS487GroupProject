import re
import pickle
import json

def seperate_decription_example_starter_codes(dataset_path):
    """
    Seperate the original prompts in the HumanEval Dataset.
    Each problem contains two fields: description and starter_codes
    """
    problems = []
    entry_points = []
    desriptions = []
    starter_codes = []
    # examples = []
    with open(dataset_path, 'r') as f:
        for line in f:
            # Every line of json
            item = json.loads(line)  
            # Get original description
            problem = item["prompt"]
            entry_point = item["entry_point"]
            problems.append(problem)
            entry_points.append(entry_point)
    for line_num in range(len(problems)):

        problem = problems[line_num]
        entry_point = entry_points[line_num]
        # Get starter codes
        function_match_start_ind = problem.find(f"def {entry_point}(")
        function_match_end_ind = problem.find(":\n",function_match_start_ind+4+len(entry_point))
        starter_code = problem[:function_match_end_ind+2]
        starter_codes.append(starter_code)
        desription = problem[function_match_end_ind+2:len(problem)-4]
        match = re.search(r'[a-zA-Z]', desription)  # 查找第一个字母
        # Must match
        start_index = match.start() if match else -1
        desription = desription[start_index:]
        desriptions.append(desription)
        # examples_ind = desription.lower().find("example")
        # if examples_ind == -1
    outputs = [{
        "description":desriptions[i],
        "starter_codes":starter_codes[i],
    } for i in range(len(starter_codes))]
    with open("/Users/mysteryshack/Downloads/Graduate/Courses/EECS487/prompt-code-487/human-eval/Experiments/processed_prompts.pkl","wb") as f:
        pickle.dump(outputs,f)
seperate_decription_example_starter_codes("/Users/mysteryshack/Downloads/Graduate/Courses/EECS487/prompt-code-487/human-eval/data/HumanEval.jsonl")
with open("/Users/mysteryshack/Downloads/Graduate/Courses/EECS487/prompt-code-487/human-eval/Experiments/processed_prompts.pkl","rb") as f:
    a = pickle.load(f)
