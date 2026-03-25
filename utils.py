import numpy as np
import torch
import json
import hmac
import hashlib
from tqdm import tqdm
import re
import ast
import os
import ftfy
from datasets import load_dataset
from transformers import AutoTokenizer

def process_text(text, prompt_len):
    truncated_text = ' '.join(text.split()[:prompt_len])
    
    if not any(punctuation in truncated_text for punctuation in '.!?'):
        return None
    
    return max((truncated_text.rsplit(punct, 1)[0] + punct
                for punct in '.!?'
                if punct in truncated_text),
               key=len)

def load_data(dataset_name, prompt_len=100, num_test=10000, ds_start_point=0, sliding_prompt=0, model_name=None):
    if dataset_name.lower() == 'c4':
        dataset = load_dataset("allenai/c4", "realnewslike", split="validation", streaming=True, trust_remote_code=True).shuffle(seed=42)
        ds_iterator = iter(dataset)
        prompt_len = prompt_len
        
        t = 0
        prompts = []
        prompt_idx = []
        prompt_cnt = -1
        human_written = []
        true_num_test = 0
        while t < num_test:
            if prompt_cnt < ds_start_point:
                example = next(ds_iterator)
                prompt_cnt += 1
            else:
                example = next(ds_iterator)
                prompt_cnt += 1
                if sliding_prompt > 0:
                    for i in range(sliding_prompt):
                        if i*prompt_len >= len(example['text'].split()):
                            break
                        text = ' '.join(example['text'].split()[i*prompt_len:(i+1)*prompt_len])
                        prompts.append(text)
                        prompt_idx.append(prompt_cnt)
                        true_num_test += 1
                else:
                    text = process_text(example['text'], prompt_len)
                    if text is None:
                        continue
                    prompts.append(text)
                    prompt_idx.append(prompt_cnt)
                    true_num_test += 1
                human_written.append(ftfy.fix_text(example['text'][len(text):]))
                t += 1
    elif dataset_name.lower() == 'opengen':
        file_path = 'OpenGen.jsonl'
        prompts, human_written = extract_prefixes(file_path=file_path, start_idx=ds_start_point, count=num_test)
        prompt_idx = list(range(ds_start_point, ds_start_point+num_test))
        true_num_test = num_test
    elif dataset_name.lower() == 'vocab':
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        vocab = tokenizer.get_vocab()
        prompts = list(vocab.keys()) 
        prompt_idx = [vocab[key] for key in prompts]
        human_written = []
        true_num_test = len(prompts)
    return prompt_idx, prompts, human_written, true_num_test


def process_valid_text(gen_text):
    cleaned_texts = []
    for text in gen_text:
        cleaned_text = re.sub(r'<pad>', '', text)
        cleaned_text = re.sub(r'<\|end_of_text\|>', '', cleaned_text)
        cleaned_text = re.sub(r'<\|endoftext\|>', '', cleaned_text)
        cleaned_texts.append(ftfy.fix_text(cleaned_text))
    return cleaned_texts        

def record_data(prompts, tokenizer, gen_token, idx_list, save_dir, params, bits=None, output_text=False, num_return_sequences=1):
    if output_text:
        valid_text = gen_token
    else:
        gen_text = tokenizer.batch_decode(gen_token, skip_special_tokens=True)
        if num_return_sequences > 1:
            valid_text = [
                gen_text[i:i + num_return_sequences]
                for i in range(0, len(gen_text), num_return_sequences)
            ]
        else:
            valid_text = process_valid_text(gen_text)

    jsonl_data = []
    for prompt, text in zip(prompts, valid_text):
        item = {
            "prompt_idx": idx_list.pop(0),
            "prompt": prompt,
            "generation_text": text
        }
        if bits is not None:
            item["bits"] = bits
        jsonl_data.append(item)
    
    # Append to JSON data file
    content_path = os.path.join(save_dir, "generation_text.jsonl")
    with open(content_path, 'a', encoding='utf-8') as f:
        for item in jsonl_data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')  # Add a newline for readability between items
    
    # Append to parameters file (only if it's empty or for the first batch)
    params_path = os.path.join(save_dir, "generation_params.json")
    if not os.path.exists(params_path) or os.path.getsize(params_path) == 0:
        with open(params_path, 'w', encoding='utf-8') as f:
            json.dump(params, f, ensure_ascii=False, indent=4)

def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except UnicodeDecodeError:
        # If UTF-8 fails, try with UTF-8-SIG (UTF-8 with BOM)
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {str(e)}")
        # If JSON decoding fails, read the file content and print it for inspection
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        print(f"File content:\n{content}")
        raise

def read_jsonl_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"Error parsing line in {file_path}: {line}")
                print(f"Error message: {str(e)}")
    return data

def extract_prefixes(file_path, start_idx=0, count=None):
    prefixes = []
    targets = []
    if count is not None:
        with open(file_path, 'r') as file:
            for i, line in enumerate(file):
                if (i < start_idx):
                    continue 
                if (i >= start_idx + count):
                    break
                try:
                    data = json.loads(line)
                    prefixes.append(data['prefix'])
                    targets.append(data['targets'])
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line}")
    else:
        with open(file_path, 'r') as file:
            for line in file:
                try:
                    data = json.loads(line)
                    prefixes.append(data['prefix'])
                    targets.append(data['targets'])
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line}")    
    return prefixes, targets


def prf(seed: torch.LongTensor, secret_key: int):
    if seed.dim() == 1:
        seed_str = ''.join(map(str, seed.tolist())) + str(secret_key)
        hash_digest = hashlib.sha256(seed_str.encode()).hexdigest()
        hash_int = int(hash_digest, 16)
        return hash_int % 2**32
    else:
        result = []
        for row in seed:
            seed_str = ''.join(map(str, row.tolist())) + str(secret_key)
            hash_digest = hashlib.sha256(seed_str.encode()).hexdigest()
            hash_int = int(hash_digest, 16)
            result.append(hash_int % 2**32) 
    return result


def init_eh_state(prefix: torch.LongTensor, state_key: int, bits_len: int) -> int:
    """Initialize E+H state-machine state from prefix."""
    if bits_len <= 0:
        return 0
    return prf(prefix, state_key) % bits_len


def eh_select_bit(prefix: torch.LongTensor, state: int, bit_counts: list, bits_len: int,
                  state_key: int, sched_key: int, candidate_width: int = 4):
    """Deterministic bit selection for E+H (budget + state-machine schedule)."""
    if bits_len <= 0:
        return 0, 0, 0.0

    transition_seed = prf(prefix, state_key)
    # odd multiplier for full cycle tendency
    a = 2 * (transition_seed % max(bits_len, 1)) + 1
    b = (transition_seed // 7) % bits_len
    new_state = (a * state + b) % bits_len

    width = max(1, min(candidate_width, bits_len))
    candidates = [int((new_state + i) % bits_len) for i in range(width)]
    selected = min(candidates, key=lambda idx: bit_counts[idx])

    credit_seed = prf(prefix, sched_key)
    credit = (credit_seed % 10000) / 10000.0
    return selected, new_state, credit


def eh_should_embed(credit: float, bit_counts: list, bit_idx: int, step_idx: int,
                    total_steps: int, min_credit: float = 0.35, boost_gap: int = 2) -> bool:
    """Deterministic budget gate used by both encoder and detector."""
    if total_steps <= 0:
        total_steps = 1
    # keep short-text capacity in early generation steps
    progress = step_idx / total_steps
    if progress <= 0.6:
        return True

    avg_count = (sum(bit_counts) / max(len(bit_counts), 1)) if bit_counts else 0
    need_boost = bit_counts[bit_idx] + boost_gap < avg_count
    late_stage = progress > 0.85
    threshold = min_credit - (0.10 if need_boost else 0.0) - (0.06 if late_stage else 0.0)
    threshold = max(0.15, threshold)
    return credit >= threshold

