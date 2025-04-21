# Gemini key: AIzaSyDzzdnYzhjR6k-A0JiE-4L9S8RK8q3XKTI

import json
import google.generativeai as genai
import time
import pickle
import re
import os

# Config
# Set API key
genai.configure(api_key="your_api_key")

# # Model choose
model = genai.GenerativeModel("gemini-1.5-flash")

def useAPI(model, prefix_prompt, func_description, candidate_count = 3, temperature = 0.5):
    """
    Use model and APIs to generate response
    """
    content = prefix_prompt + func_description,
    response = model.generate_content(
        contents = content,
        generation_config={
            'candidate_count': candidate_count,  
            'temperature': temperature,    # Make it more creative
        }
    )
    return response

def process_response(responses):
    """
    response: the object that API returns (usually of json format)
    From the model's response, get the raw string (not processed).
    """
    prompts = []
    for candidate in responses.candidates:
        if candidate.content.parts:
            text = candidate.content.parts[0].text  # 用属性访问方式
            # clean_text = text.replace("```json\n", "").replace("\n```", "").strip()
            # json_data = json.loads(clean_text)
            # prompts.append(json_data["description"])
            prompts.append(text)
    return prompts

def description2prompt(model, humaneval_path, 
                       prefix_prompt = """Generate a structured prompt to guide an AI model in completing a Python function.  
The prompt should:  
1. Clearly define the function’s purpose.  
2. Ensure the function achieves its intended goal.  

Output format (JSON):  
{  
    "description": "A concise description of the function's purpose"  
}  

Function description: """, 
                       candidate_count = 2, 
                       temperature = 0.6, 
                       output_folder = "./API/Google/meta_prompts",
                       num = 10):
    """
    From initial human_eval data set function dexcription to generate prompt.
    The output is a python list object stored in the output folder called meta_prompt.pkl
    The meta_prompt.pkl contains a list of list - list(list(str)), each string is a raw response
    """
    meta_prompts = []
    k = 0
    with open(humaneval_path, 'r') as f:
        for line in f:
            # Every line of json
            item = json.loads(line)  
            # Get function description
            func_description = item["prompt"]

            # Put the prefix and function dexcription as input
            responses = useAPI(model, prefix_prompt, func_description, candidate_count,temperature )
            responses = process_response(responses)
            meta_prompts.append(responses)

            # Make direction id the path does not exist
            os.makedirs(output_folder, exist_ok=True)
            file_path = os.path.join(output_folder, 'meta_prompts.pkl')  # 固定文件名
            if k % 10 == 0 and k != 0:
                with open(file_path, 'wb') as f:
                    print(f"The first {k} data saved.")
                    pickle.dump(meta_prompts, f)
                    time.sleep(35)
            # Control the number of data processed
            if k >= num:
                break
            time.sleep(2)
            k += 1
        with open(file_path, 'wb') as f:
            print(f"The first {k} data saved.")
            pickle.dump(meta_prompts, f)

    processed_meta_prompts = []

    # Preprocess the prompt to get the pure prompt
    # The raw prompt is like {"description": ..., "Entry Point":..., }
    processed_meta_prompts = []
    for each_line in meta_prompts:
        tasks = []
        for meata_prompt in each_line:
            clean_text = meata_prompt.replace("```json\n", "").replace("\n```", "").strip()
            # 找到第一个 `{` 和最后一个 `}` 的位置
            start = clean_text.find('{')
            end = clean_text.rfind('}')

            if start != -1 and end != -1 and end > start:
                # 提取 `{` 和 `}` 之间的内容（不包含 `{` 和 `}`）
                trimmed_content = clean_text[start + 1 : end].strip()
                tasks.append(trimmed_content)
            else:
                print("Did not find valid json format.")
        processed_meta_prompts.append(tasks)
    
    return processed_meta_prompts
            
def prompt2codes(model, 
                 humaneval_path,
                 processed_meta_prompts:list[list], 
                 prefix_prompt = """Please help me to complete this python function based on the function description.
    Finish the codes after def and pay attention to the indentation (do not include def ...)
    The prompt should follow this JSON format:
    {
        "codes": <code> ... <code>,
    }
    indentation looks like:"    import time\n    time.sleep(10)\n    return 1"
    or "\treturn 1"
    """, 
                 candidate_count = 2, 
                 temperature = 0.6, 
                 output_folder = "./API/Google/samples"):
    """
    From processed prompt, utilize API to generate codes
    """
    raw_codes = []
    k = 0
    os.makedirs(output_folder, exist_ok=True)

    file_path = os.path.join(output_folder, 'raw_codes.pkl')  # 固定文件名
    for meta_prompts_line in processed_meta_prompts:
        code_line = []
        for processed_meta_prompt in meta_prompts_line:
            responses = useAPI(model, prefix_prompt, processed_meta_prompt,candidate_count, temperature)
            raw_response = process_response(responses)
            code_line.extend(raw_response)
            time.sleep(2)
            if k % 10 == 0 and k != 0:
                with open(file_path, 'wb') as f:
                    pickle.dump(raw_codes, f)
                time.sleep(35)
            k += 1
            print("generate codes",k)
        raw_codes.append(code_line)
    # Save the list
    with open(file_path, 'wb') as f:
        pickle.dump(raw_codes, f)

    # Preprocess the raw codes
    

    def extract_function_body(code):
        # 去掉代码块标记（如果有）
        code = code.replace("```python\n", "").replace("\n```", "")
        # 去掉函数定义（从开头到第一个冒号）：假设函数定义在一行内或多行以冒号结尾
        code = re.sub(r'^\s*def\s+\w+\s*\(.*?\)\s*:\s*', '', code, flags=re.DOTALL)
        # 去掉开头的 docstring（支持 ''' 或 """）
        code = re.sub(r'^(\'\'\'|""")(?:.|\n)*?\1\s*', '', code, flags=re.DOTALL)
        return code  # 保持正确的缩进
    
    samples = []
    for each_prompt_codes in raw_codes:
        each_prompt = []
        for code in each_prompt_codes:
            clean_code = code.replace("```json\n", "").replace("\n```", "").strip()
            try:
                extracted_code = json.loads(clean_code)["codes"]
            except:
                match = re.match(r'^\s*\{\s*"codes":\s*"(.*?)"\s*(,|\})', clean_code, re.DOTALL)
                extracted_code = match.group(1) if match else clean_code
            # 提取函数体，保持缩进
            each_prompt.append(extract_function_body(extracted_code))
        samples.append(each_prompt)

    # Output json samples
    line_num = 0
    total_lines = len(samples)
    out_samples = []
    with open(humaneval_path, 'r') as f:
        for line in f:
            if line_num >= total_lines:
                break
            sample = samples[line_num]
            item = json.loads(line)  
            task_id = item["task_id"]
            for code in sample:
                out_samples.append({
                    "task_id": task_id,
                    "completion": code
                })
            line_num += 1

    file_path = os.path.join(output_folder, 'samples.jsonl')  # 固定文件名
    with open(file_path, 'w') as outfile:
        for entry in out_samples:
            json_line = json.dumps(entry)  # 将字典转换为 JSON 字符串
            outfile.write(json_line + '\n')  # 写入文件并换行

# Trial 1
humaneval_path = '/Users/mysteryshack/Downloads/Graduate/Courses/EECS487/Project/human-eval/data/HumanEval.jsonl'
processed = description2prompt(model,
                               humaneval_path=humaneval_path,
                               prefix_prompt="""Generate a structured prompt to guide an AI model in completing a Python function.  
The prompt should:  
1. Clearly defines the function’s purpose in "purpose" based on function definition.  
2. Clearly states the entry point of the python function in "Entry Point".
3. List the core steps needed to implement this function in "Steps".
4. Extract the test cases from the function description in "Tests".

Output format (JSON):  
{  
    "Purpose": "A concise description of the function's purpose.",
    "Entry Point": "def ....(....):"
    "Steps": "Concise stpes to complete the function body."
    "Tests": " a few test cases"
}  

Function description: """,
                               num=170, 
                               candidate_count = 3,
                               output_folder = "./API/Google/meta_prompts1_trail2")



print(len(processed))
prompt2codes(model, humaneval_path, processed, candidate_count = 1, 
             output_folder = "./API/Google/meta_prompts1_trail2")







