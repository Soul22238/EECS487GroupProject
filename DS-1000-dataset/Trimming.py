"""
Shorten the prompts.
Each prompt consists three fields:
1. Trimmed instructions/Descriptions in words
2. Examples (plotting problems do not have examples, leaving this field blank)
3. Starter codes
"""
import google.generativeai as genai
import pickle 
import re
import json
import os
import time

with open("/Users/mysteryshack/Downloads/Graduate/Courses/EECS487/Project/DS-1000/data/prompts_cache.pkl", 'rb') as f:
    problems = pickle.load(f)

def useAPI(model, prompt, candidate_count = 1, temperature = 0.6):
    """
    Use model and APIs to generate response.
    Then extract the plain text from the response.
    """
    responses = model.generate_content(
        contents = prompt,
        generation_config={
            'candidate_count': candidate_count,  
            'temperature': temperature,    # Make it more creative
        }
    )
    # Extract plain text from response
    text = None
    for response in responses.candidates:
        if response.content.parts:
            text = response.content.parts[0].text
    if not text:
        print("Not getting any meaningfule text!!")
        print("==================================================")
        print(response)
        print("==================================================")
    return text


def seperate(text: str, keywords: list) -> dict:
    result = {}
    for keyword in keywords:
        pattern = fr"<{keyword}>(.*?)</{keyword}>"
        match = re.search(pattern, text, re.DOTALL)
        result[keyword] = match.group(1).strip() if match else ""
    
    return result

with open('/Users/mysteryshack/Downloads/Graduate/Courses/EECS487/Project/DS-1000/trim_config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

# Set API key
genai.configure(api_key=config["key"])

# Set model
model = genai.GenerativeModel(config["model_name"])

# pattern = r"A:\s*\n<code>"
pattern = r"A:\s*\n(?:[^\n<]*\n)?<code>"
num = 0
#starter_codes_1000 = []
trimmed_prompts = []
problematic_responses = []
# Make directions
new_path = os.path.join( config["out_path"], "trimmed_prompts2(summarize_instructions).pkl")
os.makedirs(os.path.dirname(new_path), exist_ok=True)

text = """
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

x = np.linspace(0, 2 * np.pi, 400)
y1 = np.sin(x)
y2 = np.cos(x)

```

```
<comment>
plot x vs y1 and x vs y2 in two subplots
remove the frames from the subplots
</comment>
```

"""
# def extract_first_code_block(text):
#     pattern = r"```(?:\w*\n)?(.*?)```"
#     match = re.search(pattern, text, re.DOTALL)
#     codes = match.group(1).strip()
#     comment = 
# extract_first_code_block(text)


problems = problems[780:]
for problem in problems:
    match = re.search(pattern, problem)
    if match:
        # Not a plot problem
        start_index = match.start()  
        end_index = match.end()  
        problem_description = problem[:start_index] 
        starter_codes = problem[end_index:]  
        
        # Ask the LLM to seperate examples and description.
        """
        1. Reorganize the problem by separating it into two fields: a concise description wrapped in <description>...</description> and the original example wrapped in <example>...</example>. \n
        2. The description should summarize the task clearly in a few sentences. \n
        3. Do not change the content, only reorganize it. \n
        The <example> field is optional and should only be included if there is example code or data in the original input.\n
        Your response should look like <description>...</description> or <description>...</description>\n<example>...</example>

        The problem is: \n
        """
        prompt = config["prompt1"] + problem_description
        response = useAPI(model, prompt, candidate_count = 1, temperature = 0.6)

        # Seperate description and example
        result = seperate(text=response, keywords=["description", "example"])
        instructions, example = result["description"], result["example"]
        trimmed_prompt = {
            "description": instructions,
            "examples": example,
            "starter_codes": starter_codes
        }
        if not trimmed_prompt['description'] or not trimmed_prompt["starter_codes"]:
            problematic_responses.append((num, response))
            print(num)
        

    else:
        
        instructions = []
        codes = []
        for line in problem.splitlines():
            stripped = line.strip()
            if "SOLUTION START" in stripped.upper() and "#" in stripped:
                codes.append(line)
            elif stripped.startswith("#"):
                cleaned = line.lstrip()[1:].lstrip()
                instructions.append(cleaned)
            elif stripped != "":
                codes.append(line)

        
        trimmed_prompt = {
            "description": "\n".join(instructions),
            "examples": "",
            "starter_codes": "\n".join(codes)
        }
    trimmed_prompts.append(trimmed_prompt)


    num += 1
    if num %10 == 0 and num > 0:
        with open(new_path, 'wb') as f:
            print(f"first {num} saved")
            pickle.dump(trimmed_prompts, f)
        time.sleep(32)
    time.sleep(1)

with open(new_path, 'wb') as f:
    pickle.dump(trimmed_prompts, f)

with open(new_path, 'rb') as f:
    a = pickle.load(f)
print(len(a))
problematic_prompts_path = os.path.join( config["out_path"], "problematic_prompts.pkl")
with open(problematic_prompts_path, 'wb') as f:
    pickle.dump(problematic_responses, f)
with open(problematic_prompts_path, 'rb') as f:
    b = pickle.load(f)
print(len(b))
