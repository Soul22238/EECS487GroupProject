import pickle
import json

with open("/Users/mysteryshack/Downloads/Graduate/Courses/EECS487/Project/DS-1000/results/gemini-1.5-flash-TrimmedInstructionAndGeneratedCodes2CodesSelfEval-1000-pass.pkl","rb") as f:
    self_refine_pass_conditions = pickle.load(f)


with open("/Users/mysteryshack/Downloads/Graduate/Courses/EECS487/Project/DS-1000/results/gemini-1.5-flash-TrimmedInstruction2CodesBase-1000-pass.pkl","rb") as f:
    base_pass_conditions = pickle.load(f)

indices = []
for ind, self_pass_condition in enumerate(self_refine_pass_conditions):
    if (not base_pass_conditions[ind]) and self_pass_condition:
        indices.append(ind)


self_eval_codes = []
with open("//Users/mysteryshack/Downloads/Graduate/Courses/EECS487/Project/DS-1000/data/gemini-1.5-flash-TrimmedInstructionAndGeneratedCodes2CodesSelfEval-1000-answers.jsonl", 'r') as f:
    for line in f:
        item = json.loads(line)  
        codes = item["code"]
        self_eval_codes.append(codes)


 
base_codes = []   
with open("/Users/mysteryshack/Downloads/Graduate/Courses/EECS487/Project/DS-1000/data/gemini-1.5-flash-TrimmedInstruction2CodesBase-1000-answers.jsonl", 'r') as f:
    for line in f:
        item = json.loads(line)  
        codes = item["code"]
        base_codes.append(codes)


problems = []
with open("/Users/mysteryshack/Downloads/Graduate/Courses/EECS487/Project/DS-1000/data/ds1000.jsonl", 'r') as f:
    for line in f:
        item = json.loads(line)  
        problem = item["prompt"]
        problems.append(problem)


for ind in indices:
    print(ind)
    print(problems[ind])
    print(repr(base_codes[ind]))
    print("--------------------------------")
    print(repr(self_eval_codes[ind]))
    print("===============================================")