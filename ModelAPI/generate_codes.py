import json
import google.generativeai as genai
import time
import pickle
import random
import re
import os

with open('/Users/mysteryshack/Downloads/Graduate/Courses/EECS487/Project/DS-1000/ModelAPI/config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

# Set API key
genai.configure(api_key=config["key"])

# Set model
model = genai.GenerativeModel(config["model_name"])

def useAPI(model, prompt, candidate_count = 3, temperature = 0.5):
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


def getPrompt(instructions:dict, dataset_path, method:str, ids:list, trimmed_prompts_path=None):
    """
    Given the original DS1000 dataset and method of prompting/instruction technique,
    return the prompts that feed into the model.
    ids controls the indices of the problems that 
    """
    prompts = None
    if method == "Instruction2CodesBase" or method == "Instruction2CodesSelfEval":
        # Define a cache file path for pickle
        cache_file = os.path.join(os.path.dirname(dataset_path), 'prompts_cache.pkl')

        # Try to load from cache first
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                problems = pickle.load(f)
            print(f"Loaded problems from cache: {cache_file}")
        else:
            # Get a list of the original prompts from DS1000 dataset
            problems = []
            with open(dataset_path, 'r') as f:
                for line in f:
                    # Every line of json
                    item = json.loads(line)  
                    # Get problem description
                    problem = item["prompt"]
                    problems.append(problem)

            # Save to cache
            with open(cache_file, 'wb') as f:
                pickle.dump(problems, f)
            print(f"Saved problems to cache: {cache_file}")

        # Filter by ids if specified and add instruction
        instruction = instructions[method]
        prompts = [[instruction+problems[i]] for i in ids]
    elif method == "TrimmedInstruction2CodesBase":
        # The trimmed prompts have three fields: description, examples, starter_codes
        if os.path.exists(trimmed_prompts_path):
            with open(trimmed_prompts_path, 'rb') as f:
                trimmed_prompts = pickle.load(f)
            print(f"Loaded trimmed_prompts from cache: {trimmed_prompts_path}")
        instruction = instructions[method]
        prompts = [[instruction + "\n" + trimmed_prompts[i]["description"]+"\nExamples:\n"+trimmed_prompts[i]["examples"]+"\nStater Codes:\n"+trimmed_prompts[i]["starter_codes"]] for i in ids]
    elif method == "TrimmedInstruction2CodesWithoutExamples":
        if os.path.exists(trimmed_prompts_path):
            with open(trimmed_prompts_path, 'rb') as f:
                trimmed_prompts = pickle.load(f)
            print(f"Loaded trimmed_prompts from cache: {trimmed_prompts_path}")
        instruction = instructions[method]
        prompts = [[instruction + "\n" +trimmed_prompts[i]["description"]+"\nStater Codes:\n"+trimmed_prompts[i]["starter_codes"]] for i in ids]
    elif method == "TrimmedInstruction2PromptsImprovedPrompts(ExcludingStarterCodes)":
        # Use models to generate prompts
        if os.path.exists(trimmed_prompts_path):
            with open(trimmed_prompts_path, 'rb') as f:
                trimmed_prompts = pickle.load(f)
            print(f"Loaded trimmed_prompts from cache: {trimmed_prompts_path}")
        instruction = instructions[method]
        prompts = [[instruction + "\nThe prompt is as follows:\n" + trimmed_prompts[i]["description"]+"\nExamples:\n"+trimmed_prompts[i]["examples"]] for i in ids]

    return prompts
    
 
def prompt2reaponse(model,
                 prompts:list[list],
                 method:str,
                 candidate_count,
                 temperature, 
                 output_folder,
                 maximum:int
                ):
    """
    Given the prompts, iteratively call the API to get each response for each prompt.
    param: each list in the prompts correspond to one data point (one line) in DS1000 dataset
    return: responses, each list in the responses correspond to a line in the dataset
    """
    # Make directions to save the intermidiate results.
    new_path = os.path.join(output_folder, config["model_name"], method, f"{len(prompts)}raw_responses.pkl")
    os.makedirs(os.path.dirname(new_path), exist_ok=True)

    k = 0
    raw_responses = []
    for candidate_prompts in prompts:
        responses = []
        for prompt in candidate_prompts:
            response = useAPI(model, 
                              prompt, 
                              candidate_count, 
                              temperature)
            responses.append(response)
            # Slowly process to not reach maximum
            time.sleep(1)
        raw_responses.append(responses)
        if k % 10 == 0 and k != 0:
            with open(new_path, 'wb') as f:
                print(f"first{k} saved")
                pickle.dump(raw_responses, f)
            time.sleep(5)
        k += 1

        # If we want to first try on a small number of points
        if k >= maximum:
            break
    # Save the entire raw responses
    with open(new_path, 'wb') as f:
        pickle.dump(raw_responses, f)
    return raw_responses

def response2improvedprompts():
    pass

def response2answers(raw_responses, dataset_path, output_folder, ids):
    """
    Preprocess the response(contianing codes) and write these response in json format to 
    allow the DS1000 examine the accuracy
    """
    
    completetions = []
    for responses_for_each_data in raw_responses:
        codes_for_each_task = []
        for response in responses_for_each_data:
            clean_code = response.replace("```python\n", "").replace("\n```", "").strip()
            codes_for_each_task.append(clean_code)
        completetions.append(codes_for_each_task)
    ids = set(ids)

    # Output json samples
    line_num = 0
    out_samples = []
    with open(dataset_path, 'r') as f:
        for line in f:
            
            item = json.loads(line)  
            problem_id = item["metadata"]["problem_id"]
            if problem_id not in ids:
                continue
            del item["prompt"]
            del item["reference_code"]
            item["id"] = problem_id
            sample = completetions[line_num]
            for code in sample:
                item["code"] = code
                out_samples.append(item)
            line_num += 1

    # Output json file
    filename = config["model_name"]+"-"+config["method"]+"-"+str(len(ids))+"-answers.jsonl"
    file_path = os.path.join(output_folder, filename)  
    with open(file_path, 'w') as outfile:
        for entry in out_samples:
            json_line = json.dumps(entry)  
            outfile.write(json_line + '\n')  

#Only select some of the samples for test


# Make a selected dataset for ds1000
def sample_DS1000(sample_size = 10000):

    # Generate sample line num
    random.seed(487)
    if sample_size < 1000:
        ids = random.sample(range(1000), sample_size)
        ids.sort()
    else:
        ids = list(range(1000))

    line_num = 0
    samples = []
    with open(config["dataset_path"], "r") as f:
        for line in f:
            if line_num in ids:
                samples.append(json.loads(line))
            line_num += 1
    # Write jsonl file
    with open(config["output_path"]+f"sample{sample_size}ds1000.jsonl", 'w') as outfile:
        for sample in samples:
            json_line = json.dumps(sample)  
            outfile.write(json_line + '\n')
    return ids

ids = sample_DS1000(sample_size = config["num"])  



prompts = getPrompt(instructions=config["instructions"], 
                    dataset_path=config["dataset_path"], 
                    method=config["method"], 
                    ids=ids,
                    trimmed_prompts_path=config["trimmed_prompts_path"])

raw_responses = prompt2reaponse(model=model,
                 prompts=prompts,
                 method=config["method"],
                 candidate_count=config["num_responses"],
                 temperature=config["temperature"], 
                 output_folder=config["intermediate_raw_codes_path"],
                 maximum=10000
                )
raw_response_path = os.path.join(config["intermediate_raw_codes_path"], config["model_name"], config["method"], f"{config['num']}raw_responses.pkl")
print(raw_response_path)
with open(raw_response_path, 'rb') as f:
    #raw_responses = pickle.load(f)
    raw_all_responses = pickle.load(f)
print(len(raw_all_responses))
raw_responses = []
for id, responses in enumerate(raw_all_responses):
    if id not in ids:
        continue
    raw_responses.append(responses)


# Check if there are unsuccessful responses
# If there are, then ask the model to response again
num = 0
for responses in raw_responses:
    for index, response in enumerate(responses):
        if not response:
            prompt = prompts[num][index]
            another_response = useAPI(model, prompt, config["num_responses"], config["temperature"])
            print(another_response)
            print(num)
            responses[index] = another_response
    num += 1
# 
# ## Save the entire raw responses
with open(raw_response_path, 'wb') as f:
    pickle.dump(raw_responses, f)
response2answers(raw_responses, config["dataset_path"], output_folder=config["output_path"],ids=ids)

