import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import json
import os
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# === Metaprompt builder ===
def build_structured_prompt(function_description):
    return f"""Generate a structured prompt to guide an AI model in completing a Python function.  
The prompt should:  
1. Clearly defines the function’s purpose in "purpose" based on function definition.  
2. Clearly states the entry point of the python function in "Entry Point".
3. List the core steps needed to implement this function in "Steps".
4. Extract the test cases from the function description in "Tests".

Output format (JSON):  
{{  
    "Purpose": "A concise description of the function's purpose.",
    "Entry Point": "def ....(....):"
    "Steps": "Concise steps to complete the function body."
    "Tests": "a few test cases"
}}  

Function description: {function_description.strip()}"""

# === Extract markdown-style code block content ===
def extract_generation_code(raw_output):
    import re
    match = re.search(r"```(?:json|python)?\s*([\s\S]*?)```", raw_output)
    if match:
        return match.group(1).strip()
    else:
        return raw_output.strip()

# === Try to parse structured JSON from model output ===
def extract_structured_json(raw_output):
    try:
        content = extract_generation_code(raw_output)
        return json.loads(content)
    except Exception as e:
        return {"error": str(e), "raw_output": raw_output}

# === Load HumanEval dataset ===
def load_humaneval_jsonl(path, limit=None):
    data = []
    with open(path, "r") as f:
        for idx, line in enumerate(f):
            if limit is not None and idx >= limit:
                break
            data.append(json.loads(line))
    return data

# === Call the model to generate response ===
def generate_response(prompt, tokenizer, model, stop_id, max_new_tokens=512):
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        return_tensors="pt",
        add_generation_prompt=True
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=stop_id,
            pad_token_id=stop_id
        )

    output = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    return output

# === Main ===
if __name__ == "__main__":
    model_name = "deepseek-ai/deepseek-coder-6.7b-instruct"
    humaneval_path = "instruct_humanEval/HumanEval.jsonl"
    output_dir = "instruct_humanEval"
    output_path = os.path.join(output_dir, "metaprompting_output.jsonl")
    os.makedirs(output_dir, exist_ok=True)

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, use_auth_token=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    stop_id = tokenizer.convert_tokens_to_ids("<|EOT|>")
    assert isinstance(stop_id, int), "Could not find <|EOT|> token ID in tokenizer."

    print("Loading HumanEval data...")
    humaneval_data = load_humaneval_jsonl(humaneval_path)

    structured_outputs = []
    print("Running two-stage prompting...")
    for item in tqdm(humaneval_data):
        # === Stage 1: Metaprompt to get structured JSON ===
        metaprompt = build_structured_prompt(item["prompt"])
        raw_structured_response = generate_response(metaprompt, tokenizer, model, stop_id)
        structured_json = extract_structured_json(raw_structured_response)

        # === Stage 2: Feed entire structured JSON as instruction to generate code ===
        code_prompt = f"""Based on the following JSON instruction, please continue to complete the function in Python. You are not allowed to modify the given code and do the completion only. Please return all completed function in a codeblock.
```json
{json.dumps(structured_json, ensure_ascii=False, indent=2)}
```"""

        raw_code_response = generate_response(code_prompt, tokenizer, model, stop_id)
        completed_code = extract_generation_code(raw_code_response)

        structured_outputs.append({
            "task_id": item["task_id"],
            "structured_info": structured_json,
            "completion": completed_code
        })

    with open(output_path, "w") as f:
        for record in structured_outputs:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {len(structured_outputs)} results to: {output_path}")
