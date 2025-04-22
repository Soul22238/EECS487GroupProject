import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import gzip
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
    import re
    match = re.search(r"```(?:python)?\s*([\s\S]*?)```", raw_output)
    if match:
        return match.group(1).strip()
    else:
        return raw_output.strip()

# === Load DS-1000 dataset ===
def load_ds1000(path, limit=None):
    data = []
    with gzip.open(path, "rt") as f:
        for idx, line in enumerate(f):
            if limit is not None and idx >= limit:
                break
            data.append(json.loads(line))
    return data

def load_jsonl_gz(path, limit=None):
    data = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit is not None and idx >= limit:
                break
            obj = json.loads(line)
            data.append(obj)
    return data

# === Generation logic with memory-safe tricks ===
def generate_code(prompt, tokenizer, model, stop_id, max_new_tokens=512):
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        return_tensors="pt",
        add_generation_prompt=True
    ).to(model.device)

    torch.cuda.empty_cache()  # 🧹 Free up unused memory

    with torch.no_grad():
        try:
            outputs = model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=stop_id,
                pad_token_id=stop_id
            )
            output = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
            return extract_generation_code(output)
        except torch.cuda.OutOfMemoryError:
            print("⚠️ CUDA OOM encountered! Skipping prompt...")
            torch.cuda.empty_cache()
            return "# CUDA OOM: skipped"

# === Main script ===
if __name__ == "__main__":
    model_name = "deepseek-ai/deepseek-coder-6.7b-instruct"
    ds1000_path = "instruct_ds-1000/ds1000.jsonl.gz"
    prompt_path = "instruct_ds-1000/trimmed_prompts.jsonl.gz"
    output_dir = "instruct_ds-1000_result"
    output_path = os.path.join(output_dir, "deepseek-trimmed-no-examples-answers.jsonl")
    os.makedirs(output_dir, exist_ok=True)

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, use_auth_token=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"  # ✅ Automatically places model across GPU/CPU to reduce memory load
    )
    model.eval()

    stop_id = tokenizer.convert_tokens_to_ids("<|EOT|>")
    assert isinstance(stop_id, int), "Could not find <|EOT|> token ID in tokenizer."

    print("Loading DS-1000 data...")
    ds1000_data = load_ds1000(ds1000_path)
    prompt_data = load_jsonl_gz(prompt_path)

    completions = []
    print("Generating completions...")
    for item in tqdm(ds1000_data):
    # Build the full instruction prompt using the corresponding prompt metadata
        pid = item["metadata"]["problem_id"]
        composed_prompt = (
            "You are an expert programmer. Complete the following task:\n\n"
            + prompt_data[pid]["description"]
            #+ "\nExamples:\n" + prompt_data[pid]["examples"]
            + "\nStarter Codes:\n" + prompt_data[pid]["starter_codes"]
        )
        
        final_prompt = build_instruction_prompt("python", composed_prompt)
    
        # Generate model output
        completion_code = generate_code(
            final_prompt,
            tokenizer,
            model,
            stop_id=stop_id,
            max_new_tokens=384  # can be adjusted
        )
    
        # Append formatted output
        completions.append({
            "id": pid,
            "code": [f"<code>\n{completion_code}\n</code>\nEND SOLUTION\n</code>"],
            "metadata": item["metadata"]
        })

    with open(output_path, "w") as f:
        for sample in completions:
            f.write(json.dumps(sample) + "\n")

    print(f"Saved {len(completions)} completions to: {output_path}")