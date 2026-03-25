import argparse
from tqdm import tqdm
import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
import pandas as pd
import os
import numpy as np
from random import randint
from utils import record_data, load_data, read_jsonl_file
from WatermarkBimark import WatermarkBimark

import time
from datetime import datetime
import os
import json


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print('cuda available:', torch.cuda.is_available())
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def save_human_written_as_json(prompt_idx, prompts, human_written, save_dir):
    print('save_dir', save_dir)
    json_data = []
    for idx, prompt, completion in zip(prompt_idx, prompts, human_written):
        json_data.append({
            "prompt_idx": idx,
            "prompt": prompt,
            "human_completion": completion
        })
    
    # Save JSON data
    with open(os.path.join(save_dir, "human_written.jsonl"), 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)

def append_runtime(save_dir, runtime):
    params_path = os.path.join(save_dir, "generation_params.json")
    with open(params_path, 'r', encoding='utf-8') as f:
        params = json.load(f)
    params['runtime'] = runtime
    with open(params_path, 'w', encoding='utf-8') as f:
        json.dump(params, f, ensure_ascii=False, indent=4)
        
def create_save_dir(args, method, time_str):
    print('creating save directory ...')
    save_dir = f"./output_dump/{method}_{args.dataset.replace('/','_')}_{args.model_name.replace('/','-')}_"
    save_dir = save_dir + time_str
    os.mkdir(save_dir)
    print('save_dir:', save_dir)
    return save_dir

def main(args):
    begin = time.time()
    # print('args.load_partition_mask: ', args.load_partition_mask)
    now = datetime.now()
    time_str = now.strftime("%Y-%m-%d %H:%M:%S").replace(' ','_').replace(":","-")
    save_dir = create_save_dir(args, args.method,  time_str)
    
    auth_token = "xxx"
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map='auto', torch_dtype=torch.bfloat16, use_auth_token=auth_token)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, use_auth_token=auth_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    
    prob_delta = args.prob_delta
    
    model.eval()
    print('model.device', model.device)

    
    prompt_idx, prompts, human_written, num_test = load_data(args.dataset, args.prompt_len, args.num_test)

    with open(os.path.join(save_dir, "prompt_idx.txt"), "w") as f:
        for idx in prompt_idx:
            f.write(str(idx)+'\n')

    save_human_written_as_json(prompt_idx, prompts, human_written, save_dir)
            
    vocab_size = model.config.vocab_size

    batch_size = args.batch_size
    print("num_test: ", num_test)
    n_batches = int(np.ceil(num_test / batch_size))
    pbar = tqdm(total=n_batches)
    
    for batch in range(n_batches):
            
        batch_prompts = prompts[batch * batch_size: min(num_test, (batch +1) * batch_size)]
        
        inputs = tokenizer(
                batch_prompts,
                padding=True,
                return_tensors="pt",
                truncation=True
            )
        
        input_ids = inputs["input_ids"].to(model.device)
        print('input_ids.size()', input_ids.size())
        attention_mask = inputs["attention_mask"].to(model.device)
        
        prompt_tokens_len = (input_ids).size(-1)
        print('prompt_tokens_len', prompt_tokens_len)
        
        print('bitlen', len(args.message))
        
        if args.method.lower() == 'bimark':
            if args.random_message:
                tmp = []
                for _ in range(len(args.message)):
                    tmp.append(str(randint(0, 1)))
                bits = "".join(tmp)
            else:
                bits = args.message
            watermark_processor_bimark = WatermarkBimark(tokenizer=tokenizer, vocab_size=vocab_size, device=device, top_k=args.top_k, partition_seeds=args.partition_seeds, 
                                                         c_key=args.c_key, bit_idx_key=args.bit_idx_key, delta=args.prob_delta, window_size=args.window_size, bits=bits,
                                                         eh_enable=args.eh_enable, eh_state_key=args.eh_state_key, eh_sched_key=args.eh_sched_key,
                                                         eh_candidate_width=args.eh_candidate_width, eh_min_credit=args.eh_min_credit,
                                                         eh_allow_skip=args.eh_allow_skip,
                                                         max_new_tokens=args.max_new_tokens)
            generate_args_bimark = {'logits_processor': [watermark_processor_bimark],  'max_new_tokens': args.max_new_tokens,'temperature': args.temperature,  'attention_mask': attention_mask, 
                                  'do_sample': args.do_sample, 'top_k': args.top_k}
        
        elif args.method.lower() == 'no_watermark':
            generate_args_null = {'max_new_tokens': args.max_new_tokens, 'attention_mask': attention_mask,  'temperature': args.temperature,
                                  'do_sample': args.do_sample, 'top_k': args.top_k}
              
        
        idx_list = [prompt_idx[i] for i in range(batch * batch_size, min(num_test, (batch + 1) * batch_size))]
        
        if args.method.lower() == 'no_watermark':
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            with torch.inference_mode():
                null_token = model.generate(input_ids, **generate_args_null)
            params = {
                "method": "no_watermark",
                "model_name": args.model_name,
                "vocab_size": vocab_size,
                'temperature': args.temperature, 
                "top_k": args.top_k,
                "do_sample": args.do_sample,
                "max_length": args.max_new_tokens,
                "time_stamp": time_str,
            }
            record_data(batch_prompts, tokenizer, null_token[:,prompt_tokens_len:].tolist(), idx_list, save_dir, params)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if args.method.lower() == 'bimark':
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            with torch.inference_mode():
                new_token = model.generate(input_ids, **generate_args_bimark)
                print('new_token.shape', new_token.shape)
                print('new_token.size()', new_token.size())
            params = {
                "method": "bimark",
                "model_name": args.model_name,
                "vocab_size": vocab_size,
                "bits_len": len(args.message),
                "top_k": args.top_k,
                "do_sample": args.do_sample,
                'temperature': args.temperature, 
                "max_length": args.max_new_tokens,
                "prob_delta": prob_delta,
                "window_size": args.window_size,
                "c_key": args.c_key,
                "bit_idx_key": args.bit_idx_key,
                "partition_seeds": args.partition_seeds,
                "eh_enable": args.eh_enable,
                "eh_state_key": args.eh_state_key,
                "eh_sched_key": args.eh_sched_key,
                "eh_candidate_width": args.eh_candidate_width,
                "eh_min_credit": args.eh_min_credit,
                "eh_allow_skip": args.eh_allow_skip,
                "time_stamp": time_str,
            }
            record_data(batch_prompts, tokenizer, new_token[:,prompt_tokens_len:].tolist(), idx_list, save_dir, params, bits=bits)

        pbar.update(1)
    
    end = time.time()
    runtime = end - begin
    print("time cost", runtime)
    
    append_runtime(save_dir, runtime)
        
    print("Finished!")


if __name__ == "__main__":

    def list_of_ints(arg):
        if type(arg) is list:
            return arg
        else:
            return list(map(int, arg.split(',')))
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B") # meta-llama/Meta-Llama-3-8B, Qwen/Qwen2.5-3B
    parser.add_argument("--method", type=str, default='bimark')  # no_watermark
    parser.add_argument("--prob_delta", type=float, default=0.2, help="base scaling factor of BiMark")  
    parser.add_argument("--partition_seeds", type=list_of_ints, default=[int(x) for x in np.random.choice(10000, size=20, replace=False)], help="seeds for random vocabulary partitions") 
    parser.add_argument("--c_key", type=int, default=8214793, help="key for bit flipping")
    parser.add_argument("--bit_idx_key", type=int, default=283519, help="key for randomly selecting one position of a message.")
    parser.add_argument("--dataset", type=str, default="c4", help="the dataset for text generation.")  
    parser.add_argument("--max_new_tokens", type=int, default=350, help="the maximum number of generated tokens.")
    parser.add_argument("--prompt_len", type=int, default=100, help="the length of prompts")
    parser.add_argument("--num_test", type=int, default=1000)
    parser.add_argument("--window_size", type=int, default=2, help="e.g. using 2 previous tokens as seed for pseudorandom number generator for bit flipping, selecting message indices, vocabulary permutation, or vocabulary partition.") 
    parser.add_argument("--message", type=str, default="0"*1, help="hidden message carried by watermarks, capable for BiMark, MPAC")
    parser.add_argument("--random_message", action='store_true', help="create random message for each text generation.")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--do_sample", action='store_true')
    parser.add_argument("--eh_enable", action='store_true', help="enable E+H (budget optimization + state-machine bit scheduling)")
    parser.add_argument("--eh_state_key", type=int, default=99431, help="state-machine key for E+H")
    parser.add_argument("--eh_sched_key", type=int, default=137631, help="budget scheduling key for E+H")
    parser.add_argument("--eh_candidate_width", type=int, default=8, help="candidate width for state-machine bit selection")
    parser.add_argument("--eh_min_credit", type=float, default=0.25, help="minimum budget credit for applying watermark at a step")
    parser.add_argument("--eh_allow_skip", action='store_true', help="allow budget gating to skip embedding; default is fallback to baseline embedding")
    
    args = parser.parse_args()
    main(args)
 
