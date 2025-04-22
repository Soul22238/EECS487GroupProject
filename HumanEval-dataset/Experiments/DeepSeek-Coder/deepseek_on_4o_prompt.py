import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import json
import os
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# === Utility: DeepSeek-style prompt builder ===
def build_instruction_prompt(language, question):
    return f"""Please continue to complete the function. You are not allowed to modify the given code and do the completion only. Please return all completed function in a codeblock. Here is the given code to do completion:
```{language.lower()}
{question.strip()}
```"""

# === Utility: clean generated output to extract function code ===
def extract_generation_code(raw_output):
    """
    Extract code from markdown-style codeblock: ```python\n<code>\n```
    If multiple codeblocks or stray text, returns best-effort cleanup.
    """
    import re
    match = re.search(r"```(?:python)?\s*([\s\S]*?)```", raw_output)
    if match:
        return match.group(1).strip()
    else:
        return raw_output.strip()

# === Load HumanEval dataset ===
def load_humaneval_jsonl(path, limit=None):
    data = []
    with open(path, "r") as f:
        for idx, line in enumerate(f):
            if limit is not None and idx >= limit:
                break
            data.append(json.loads(line))
    return data

# === Generation logic ===
def generate_code(prompt, tokenizer, model, stop_id, max_new_tokens=512):
    # Chat-style input
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
    return extract_generation_code(output)

# === Main script ===
if __name__ == "__main__":
    model_name = "deepseek-ai/deepseek-coder-6.7b-instruct"
    humaneval_path = "instruct_humanEval/4o_RefinedPrompts.jsonl"
    output_dir = "instruct_humanEval"
    output_path = os.path.join(output_dir, "instruct_output_with_refined_prompt.jsonl")
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

    completions = []
    print("Generating completions...")
    for item in tqdm(humaneval_data):
        prompt = build_instruction_prompt("python", item["prompt"])
        completion_code = generate_code(prompt, tokenizer, model, stop_id)
        completions.append({
            "task_id": item["task_id"],
            "completion": completion_code
        })

    with open(output_path, "w") as f:
        for sample in completions:
            f.write(json.dumps(sample) + "\n")

    print(f"Saved {len(completions)} cleaned completions to: {output_path}")