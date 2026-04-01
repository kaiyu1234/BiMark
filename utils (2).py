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
from datasets import load_dataset
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
        dataset = load_dataset("allenai/c4", "realnewslike", split="validation", streaming=True).shuffle(seed=42)
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
    elif dataset_name.lower() == 'cnn_dailymail':
        # 自动下载并加载 CNN/DailyMail 摘要数据集 (使用 3.0.0 版本)

        dataset = load_dataset("cnn_dailymail", "3.0.0", split="test", streaming=True)
        ds_iterator = iter(dataset)
        
        t = 0
        prompts = []
        prompt_idx = []
        human_written = []
        
        # 可选：定义文章的最大截断长度（单词数），防止过长导致模型 OOM
        max_article_words = 600 
        
        while t < num_test:
            try:
                example = next(ds_iterator)
            except StopIteration:
                break
                
            # 获取并稍微清理原文（可根据需要截断，因为 CNN/DM 文章较长）
            article_text = example['article']
            truncated_article = " ".join(article_text.split()[:max_article_words])
            
            # 构建摘要任务的 Prompt
            # text = f"Summarize the following article in 60 to 65 words and retain key information while expressing it concisely:\n{truncated_article}\n\nSummary:"
            text = f"Summarize the following article concisely (2-3 sentences):\n{truncated_article}\n\nSummary:"
            # text = truncated_article
            prompts.append(text)
            prompt_idx.append(t)
            
            # 真实的摘要作为 Reference (CNN/DM 中对应的键是 'highlights')
            human_written.append(example['highlights'])
            t += 1
            
        true_num_test = t
    elif dataset_name.lower() == 'wmt16':
        # 自动下载并加载 WMT16英罗翻译数据集
        dataset = load_dataset("wmt16", "ro-en", split="test", streaming=True)
        ds_iterator = iter(dataset)
        
        t = 0
        prompts = []
        prompt_idx = []
        human_written = []
        while t < num_test:
            example = next(ds_iterator)
            # 构建英译罗的 Prompt
            text = f"Translate the following English text to Romanian:\n{example['translation']['en']}\n\nTranslation:"
            prompts.append(text)
            prompt_idx.append(t)
            # 真实的罗语翻译作为 Reference
            human_written.append(example['translation']['ro'])
            t += 1
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

