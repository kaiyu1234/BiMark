import argparse
import torch
from transformers import AutoTokenizer
import pandas as pd
import os
import numpy as np
import time
import json
from perplexity import LocalModel
from utils import read_json_file, read_jsonl_file
from detect_watermark_dump import WatermarkDetector
from tqdm import tqdm
from dipper import DipperParaphraser

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print('cuda available:', torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else ('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu')
print('device', device)
      
   
def main(args):
    perplexity = args.perplexity
    detect = args.detect
    length_all = args.length_all
    ppl_length = args.ppl_length
    local_model = args.local_model
    ppl_model_name =  "google/gemma-2-9b" 
    
    path = os.path.join(
        "output_dump",
        args.data_dir
    )
    
    prefix_list = [path]
    

    flg_load = True
    for prefix in prefix_list:
        if args.paraphrase_detect:
            content_path = os.path.join(os.getcwd(), f"{prefix}", f"paraphrase_text_{args.lex_diversity}_{args.order_diversity}.jsonl")
        else:
            content_path = os.path.join(os.getcwd(), f"{prefix}", f"generation_text.jsonl")


        param_path = os.path.join(os.getcwd(), f"{prefix}", f"generation_params.json")

        if args.paraphrase_detect:
            detect_save_path = os.path.join(os.getcwd(), f"{prefix}", f"detect_result_wm_dp_{args.lex_diversity}_{args.order_diversity}.csv")
        else:
            detect_save_path = os.path.join(os.getcwd(), f"{prefix}", f"detect_result_wm.csv")

        if not length_all:
            ppl_save_path = os.path.join(os.getcwd(), f"{prefix}", f"ppl_result_{ppl_length}_gemma.csv".replace('/','-'))
        else:
            ppl_save_path = os.path.join(os.getcwd(), f"{prefix}",  f"ppl_result_all.csv".replace('/','-'))
        
        if args.paraphrase_attack:
            paraphrase_save_path = os.path.join(os.getcwd(), f"{prefix}", f"paraphrase_text_{args.lex_diversity}_{args.order_diversity}.jsonl")

        params = read_json_file(param_path)
        print('params:', params)
        method_name = params['method']

        # Set up parameters based on method
        
        if method_name == 'bimark':
            partition_seeds = params["partition_seeds"]
            c_key = params["c_key"]
            bit_idx_key = params["bit_idx_key"]
            vocab_size = params["vocab_size"]
            gamma = 0.5
            window_size = params['window_size']

        elif 'mpac' in method_name.lower():
            bit_idx_key = params["bit_idx_key"]
            vocab_size = params["vocab_size"]
            gamma = 0.5
            c_key = params['c_key']
            window_size = params['window_size']
        elif 'dipmark' in method_name.lower():
            c_key = params['c_key']
            vocab_size = params['vocab_size']
            gamma = args.gamma
            window_size = params['window_size']
        elif 'kgw' in method_name.lower():
            c_key = params['global_key']
            vocab_size = params['vocab_size']
            gamma = params["gamma"]
            window_size = params['window_size']

        content = read_jsonl_file(content_path)
        
        model_name = params['model_name']
        print('model_name', model_name)

        auth_token = "xxx"

        if flg_load:
            if detect:
                tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype=torch.bfloat16, use_auth_token=auth_token)
                tokenizer.pad_token = tokenizer.eos_token

            if perplexity:
                tokenizer_ppl = AutoTokenizer.from_pretrained(ppl_model_name, torch_dtype=torch.bfloat16, use_auth_token=auth_token)
                scorer = LocalModel(ppl_model_name, device)
            
            if args.paraphrase_attack:
                dp = DipperParaphraser(model="kalpeshk2011/dipper-paraphraser-xxl")

            flg_load = False

        if perplexity:
            count = -1
            flg_ppl = True
            for idx, item in enumerate(tqdm(content)):
                count += 1
                if count > 10000:
                    break
                prompt = item['prompt']
                generation_text = item['generation_text']

                if not length_all:
                    valid_tokens = tokenizer_ppl.encode(generation_text, truncation=False, return_tensors='pt', add_special_tokens=False)[0].to(device)
                    print(f'len(valid_tokens): {len(valid_tokens)}')
                    if len(valid_tokens) < ppl_length:
                        continue
                    else:
                        text_for_score = tokenizer_ppl.decode(valid_tokens[:ppl_length], skip_special_tokens=True)
                else:
                    if len(generation_text) < 0:
                        continue
                    text_for_score = generation_text

                ppl = scorer.get_perplexity(prompt, [text_for_score])                
                print(f'index_{idx}, model_{ppl_model_name}, method_{method_name}, ppl_{ppl}')
                ppl = pd.DataFrame(ppl, columns=[f'{method_name}_ppl'])
                if flg_ppl:
                    ppl.to_csv(ppl_save_path,header=True, index=False)
                    flg_ppl = False
                else:
                    ppl.to_csv(ppl_save_path, mode='a', header=False, index=False)

        flg_record = True

        if args.paraphrase_attack:
            print("Start paraphrasing...")
            batch_size = args.paraphrase_batch

            for batch_start in tqdm(range(0, len(content), batch_size)):
                batch = content[batch_start:batch_start + batch_size]
                # generation_texts = [item['generation_text'] for item in batch if item['generation_text']]
                generation_texts = [item.get('generation_text') if item.get('generation_text') else "111111" for item in batch]
                prompts = [item.get('prompt', '') for item in batch]

                paraphrased_outputs = dp.paraphrase_batch(
                    generation_texts,
                    lex_diversity=args.lex_diversity,
                    order_diversity=args.order_diversity,
                    prefixes=prompts,
                    do_sample=True,
                    top_p=0.75,
                    top_k=None,
                    max_length=512,
                )

                with open(paraphrase_save_path, 'a', encoding='utf-8') as f:
                    for item, para_text in zip(batch, paraphrased_outputs):
                        paraphrase_item = {
                            "prompt_idx": item['prompt_idx'],
                            "prompt": item['prompt'],
                            "generation_text": item['generation_text'],
                            "paraphrase_text": para_text,
                            "lex_diversity": args.lex_diversity,
                            "order_diversity": args.order_diversity,
                        }
                        json.dump(paraphrase_item, f, ensure_ascii=False)
                        f.write('\n')

            print(f"Paraphrasing complete. Saved to: {paraphrase_save_path}")
            

        if detect:
            # Detect watermark
            detector = WatermarkDetector(tokenizer, vocab_size, window_size=window_size, gamma=gamma)

            start, stride = 25, 25
            idx = 0
            count = -1
            for idx, item in enumerate(tqdm(content)):
                count += 1
                if count > 10000:
                    break
                idx += 1
                prompt = item['prompt']

                if args.paraphrase_detect:
                    generation_text = item['paraphrase_text']
                else:
                    generation_text = item['generation_text']
                
                generate_tokens = tokenizer.encode(generation_text, truncation=False, return_tensors='pt', add_special_tokens=False)[0].to(device)

                print('len(generate_tokens):', len(generate_tokens))
                if len(generate_tokens) < start:
                    continue
                
                if 'bimark' in method_name.lower():
                    try:
                        bits = item['bits']
                    except:
                        bits = "0"
                    verify_z_score, verify_z_p_value, verify_green_count,  verify_generate_counts, verify_valid_counts, stride_list, bits_green_count, bits_valid_count, z_score_bits, z_p_value_bits  = detector.verify_bimark_multibit(
                                        detect_gen_tokens=generate_tokens, partition_seeds=partition_seeds,  
                                        c_key=c_key, bit_idx_key=bit_idx_key, bits=bits, weight=args.weight, start=start, stride=stride)
                    
                    COUNTS, detect_generate_counts, detect_green_counts, detect_valid_counts, detect_z_scores, detect_p_values, decode_bits, hit, hit_rate = detector.decode_bimark_multibit_watermark(
                                        inputs=generate_tokens,  partition_seeds=partition_seeds,  c_key=c_key, 
                                        bit_idx_key=bit_idx_key, bits=bits, bits_len=len(bits), weight=args.weight, start=start, stride=stride)
                    
                    result = {'stride_list': stride_list, 'verify_z_score': verify_z_score, 'verify_z_p_value': verify_z_p_value, 'verify_green_count': verify_green_count, 'decode_bits': decode_bits, 'hit': hit, 'hit_rate': hit_rate,
                              'verify_generate_counts': verify_generate_counts, 'verify_valid_counts': verify_valid_counts, 'detect_generate_counts': detect_generate_counts, 'detect_green_counts': detect_green_counts, 
                              'detect_valid_counts': detect_valid_counts, 'detect_z_scores': detect_z_scores, 'detect_p_values': detect_p_values, 'COUNTS': COUNTS}
                    
                    bits_green_count = list(map(list, zip(*bits_green_count)))
                    bits_valid_count = list(map(list, zip(*bits_valid_count)))
                    z_score_bits = list(map(list, zip(*z_score_bits)))
                    z_p_value_bits = list(map(list, zip(*z_p_value_bits)))
                    
                    print('bits_green_count:', bits_green_count)
                    print('bits_valid_count:', bits_valid_count)
                    print('z_score_bits:', z_score_bits)
                    print('z_p_value_bits:', z_p_value_bits)
                    for i in range(len(bits)):
                        result[f'bits_green_count_{i}'] = bits_green_count[i]
                        result[f'bits_valid_count_{i}'] = bits_valid_count[i]
                        result[f'z_score_bits_{i}'] = z_score_bits[i]
                        result[f'z_p_value_bits_{i}'] = z_p_value_bits[i]


                    record_params = dict()
                    record_params.update(result)
                    try:
                        record_params.pop('stride_list')
                    except:
                        pass

                    num = len(result['stride_list'])

                    record_params['method_name'] = [method_name] * num
                    record_params['length'] = list(np.array(result['stride_list']))
                    record_params['detected_gen_text']= [tokenizer.decode(generate_tokens[:length]) for length in result['stride_list']]

                    columns_result = ['method_name', 'length', 'verify_z_score', 'verify_z_p_value', 'decode_bits', 'hit', 'hit_rate', 'verify_green_count', 
                                    'verify_generate_counts', 'verify_valid_counts', 'detected_gen_text', 
                                    'COUNTS', 'detect_green_counts', 'detect_valid_counts', 'detect_z_scores', 'detect_p_values']

                    columns_result += [f'bits_green_count_{i}' for i in range(len(bits))]
                    columns_result += [f'bits_valid_count_{i}' for i in range(len(bits))]
                    columns_result += [f'z_score_bits_{i}' for i in range(len(bits))]
                    columns_result += [f'z_p_value_bits_{i}' for i in range(len(bits))]

                    print('columns_result:', columns_result)

                    df_result = pd.DataFrame(record_params, columns=columns_result)
                    if flg_record:
                        df_result.to_csv(detect_save_path, mode='w', index=False, header=True, columns=columns_result)
                        flg_record = False
                    else:
                        df_result.to_csv(detect_save_path, mode='a', index=False, header=False, columns=columns_result)
                

def detect_watermark(key, tokenizer,  detector, detected_gen_text, start=0, stride=0):
    detected_gen_token = tokenizer.encode(detected_gen_text, truncation=False, return_tensors='pt', add_special_tokens=False)[0].to(device)      
    z_score, z_p_value, green_count, generate_tokens, valid_tokens, stride_list = detector.zerobit_watermark_detector(key, detected_gen_token,  start=start, stride=stride)
    result = {'stride_list': stride_list, 'z_score': z_score, 'z_p_value': z_p_value, 'green_count': green_count, 'generate_tokens': generate_tokens, 'valid_tokens': valid_tokens}
    return result

if __name__ == "__main__":
    def list_of_ints(arg):
        if type(arg) is list:
            return arg
        else:
            return list(map(int, arg.split(',')))
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, help="data directory", default="bimark_c4_meta-llama-Meta-Llama-3-8B_2025-03-31_21-31-34")
    parser.add_argument("--perplexity", action="store_true", help="calculate perplexity")
    parser.add_argument("--weight", type=list_of_ints, default=[1 for _ in range(30)], help="weight for each layer, make sure it matches the number of layers")
    parser.add_argument("--ppl_length", type=int, default=200, help="length for perplexity")
    parser.add_argument("--detect", action="store_true", help="detect watermark")
    parser.add_argument("--length_all", action="store_true", help="calculate perplexity for all length")
    parser.add_argument("--paraphrase_attack", action="store_true", help="pharase watermarked text via DIPPER")
    parser.add_argument("--paraphrase_detect", action="store_true", help="detect watermark based on pharased watermarked text")
    parser.add_argument("--lex_diversity", type=int, default=0, help="The lexical diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity")
    parser.add_argument("--order_diversity", type=int, default=20, help="The order diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity")
    parser.add_argument("--paraphrase_batch", type=int, default=1)
    args = parser.parse_args()

    begin = time.time()
    main(args)
    end = time.time()
    print("time cost", end - begin)
