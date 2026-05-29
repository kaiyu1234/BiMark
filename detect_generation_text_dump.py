import argparse
import torch
from transformers import AutoTokenizer
import pandas as pd
import os
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "30"
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
    # local_model = args.local_model
    # ppl_model_name =  "google/gemma-2-9b" 
    ppl_model_name =  "meta-llama/Meta-Llama-3.1-8B" 
    #ppl_model_name =  "gpt2" 
    path = os.path.join(
        #"output_dump",
        args.data_dir
    )
    
    prefix_list = [path]
    

    flg_load = True
    for prefix in prefix_list:
        if args.paraphrase_detect:
            content_path = os.path.join(os.getcwd(), f"{prefix}", f"paraphrase_text_{args.lex_diversity}_{args.order_diversity}.jsonl")
            detect_save_path = os.path.join(os.getcwd(), f"{prefix}", f"detect_result_wm_dp_{args.lex_diversity}_{args.order_diversity}.csv")
        # --- 新增替换攻击的路径 ---
        elif args.substitution_detect:
            content_path = os.path.join(os.getcwd(), f"{prefix}", f"substitution_text_{args.substitution_ratio}.jsonl")
            detect_save_path = os.path.join(os.getcwd(), f"{prefix}", f"detect_result_wm_sub_{args.substitution_ratio}.csv")
        # ------------------------
        else:
            content_path = os.path.join(os.getcwd(), f"{prefix}", f"generation_text.jsonl")
            detect_save_path = os.path.join(os.getcwd(), f"{prefix}", f"detect_result_wm.csv")


        param_path = os.path.join(os.getcwd(), f"{prefix}", f"generation_params.json")

        
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
        dummy_bits = None
        if method_name == 'no_watermark' and args.baseline_method is not None:
            print(f"⚠️ 正在检测无水印基线，强制使用 {args.baseline_method} 探测器寻找目标比特: {args.baseline_bits}")
            method_name = args.baseline_method  # 覆盖方法名
            dummy_bits = args.baseline_bits     # 记录假想比特
            c_key = args.c_key
            bit_idx_key = args.bit_idx_key
            vocab_size = params.get('vocab_size', 128256)
            window_size = params.get('window_size', 2)
            gamma = 0.5
            partition_seeds = args.partition_seeds

        
        elif method_name == 'bimark':
            partition_seeds = params["partition_seeds"]
            c_key = params["c_key"]
            bit_idx_key = params["bit_idx_key"]
            vocab_size = params["vocab_size"]
            gamma = 0.5
            window_size = params['window_size']
        # 👇 添加这段针对 watermod 的参数读取代码 👇
        elif method_name == 'watermod':
            c_key = params["c_key"]
            vocab_size = params["vocab_size"]
            window_size = params['window_size']
            gamma = 0.5 # 占位符，WaterMod 不使用传统的 gamma
        # 👆 添加结束 👆
            
        elif method_name == 'xmark':
            vocab_size = params["vocab_size"]
            window_size = params.get('window_size', 2)
            # XMark 使用固定的 hash_key，如果在生成端你把它加进 params 了可以像下面这样读：
            # hash_key = params.get("hash_key", 15485863) 
            gamma = 0.5 # 占位，XMark也不需要传统的gamma
        elif method_name == 'omnimark':
            c_key = params["c_key"]
            vocab_size = params["vocab_size"]
            window_size = params['window_size']
            gamma = 0.5 # 占位，OmniMark其实不需要用到传统的gamma
        # elif 'mpac' in method_name.lower():
        #     bit_idx_key = params["bit_idx_key"]
        #     vocab_size = params["vocab_size"]
        #     gamma = 0.5
        #     c_key = params['c_key']
        #     window_size = params['window_size']
        elif method_name == 'mpac':
            c_key = params["c_key"]
            vocab_size = params["vocab_size"]
            window_size = params['window_size']
            # 读取我们在生成时保存的 gamma，如果没有保存则默认使用 MPAC 论文里的 0.25
            gamma = params.get("gamma", 0.25)
        
        elif method_name == 'stealthink':
            c_key = params["c_key"]
            vocab_size = params["vocab_size"]
            window_size = params.get('window_size', 2)
            R = params.get("R", 0.25)  # 读取刚刚保存的 R，兼容以前生成的 0.5
            gamma = R
            
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

        auth_token = "***"

        if flg_load:
            if detect:
                tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype=torch.bfloat16, token=auth_token)
                #tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype=torch.bfloat16)
                tokenizer.pad_token = tokenizer.eos_token
                model_for_detection = None
                if method_name == 'watermod':
                    print("Loading LLM model for WaterMod detection...")
                    from transformers import AutoModelForCausalLM
                    model_for_detection = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map='auto', token=auth_token)
                    model_for_detection.eval()

            if perplexity:
                tokenizer_ppl = AutoTokenizer.from_pretrained(ppl_model_name, torch_dtype=torch.bfloat16, token=auth_token)
                # tokenizer_ppl = AutoTokenizer.from_pretrained(ppl_model_name, torch_dtype=torch.bfloat16)
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
                        if 'bits' in item:
                            paraphrase_item['bits'] = item['bits']
                        json.dump(paraphrase_item, f, ensure_ascii=False)
                        f.write('\n')

            print(f"Paraphrasing complete. Saved to: {paraphrase_save_path}")
            

        if detect:
            # Detect watermark
            detector = WatermarkDetector(tokenizer, vocab_size, window_size=window_size, gamma=gamma)

            start, stride = 25, 25
            total_decode_time = 0.0
            valid_sample_count = 0
            idx = 0
            count = -1
            for idx, item in enumerate(tqdm(content)):
                count += 1
                if count > 10000:
                    break
                idx += 1
                prompt = item['prompt']

                # 👇 添加这行代码，将 prompt 转化为 tokens 👇
                prompt_tokens = tokenizer.encode(prompt, truncation=False, return_tensors='pt', add_special_tokens=False)[0].to(device)
                # 👆 添加结束
                
                if args.paraphrase_detect:
                    generation_text = item['paraphrase_text']
                # --- 新增：读取被替换后的文本 ---
                elif args.substitution_detect:
                    generation_text = item['substitution_text']
                else:
                    generation_text = item['generation_text']
                
                generate_tokens = tokenizer.encode(generation_text, truncation=False, return_tensors='pt', add_special_tokens=False)[0].to(device)

                print('len(generate_tokens):', len(generate_tokens))
                if len(generate_tokens) < 45: #### 最小检测长度
                    continue
                
                if 'bimark' in method_name.lower():
                    try:
                        bits = item['bits']
                    except:
                        bits = "0"
                    decode_start_time = time.time()
                    verify_z_score, verify_z_p_value, verify_green_count,  verify_generate_counts, verify_valid_counts, stride_list, bits_green_count, bits_valid_count, z_score_bits, z_p_value_bits  = detector.verify_bimark_multibit(
                                        detect_gen_tokens=generate_tokens, partition_seeds=partition_seeds,  
                                        c_key=c_key, bit_idx_key=bit_idx_key, bits=bits, weight=args.weight, start=start, stride=stride)
                    
                    COUNTS, detect_generate_counts, detect_green_counts, detect_valid_counts, detect_z_scores, detect_p_values, decode_bits, hit, hit_rate = detector.decode_bimark_multibit_watermark(
                                        inputs=generate_tokens,  partition_seeds=partition_seeds,  c_key=c_key, 
                                        bit_idx_key=bit_idx_key, bits=bits, bits_len=len(bits), weight=args.weight, start=start, stride=stride)
                    
                    # 取最后一个stride（最终结果）
                    final_index = -1
                    final_length = stride_list[final_index]
                    # ================= 新增：记录解码结束时间并累加 =================
                    decode_end_time = time.time()
                    total_decode_time += (decode_end_time - decode_start_time)
                    valid_sample_count += 1
                
                    # 转置后的数据，同样取最后一个元素
                    bits_green_count = list(map(list, zip(*bits_green_count)))
                    bits_valid_count = list(map(list, zip(*bits_valid_count)))
                    z_score_bits = list(map(list, zip(*z_score_bits)))
                    z_p_value_bits = list(map(list, zip(*z_p_value_bits)))
                
                    # 打印最终结果（可选保留）
                    print('最终步长:', final_length)
                    print('bits_green_count (最终):', [x[final_index] for x in bits_green_count])
                    print('bits_valid_count (最终):', [x[final_index] for x in bits_valid_count])
                    print('z_score_bits (最终):', [x[final_index] for x in z_score_bits])
                    print('z_p_value_bits (最终):', [x[final_index] for x in z_p_value_bits])
                
                    # 构建单条记录（仅最终结果）
                    record_params = {
                        # 基础信息
                        'method_name': method_name,
                        'length': final_length,
                        'detected_gen_text': tokenizer.decode(generate_tokens[:final_length]),
                        # 验证结果
                        'verify_z_score': verify_z_score[final_index],
                        'verify_z_p_value': verify_z_p_value[final_index],
                        'verify_green_count': verify_green_count[final_index],
                        'verify_generate_counts': verify_generate_counts[final_index],
                        'verify_valid_counts': verify_valid_counts[final_index],
                        # 解码结果
                        'decode_bits': decode_bits,
                        'hit': hit[final_index],
                        'hit_rate': hit_rate[final_index],
                        'COUNTS': COUNTS[final_index],
                        'detect_green_counts': detect_green_counts[final_index],
                        'detect_valid_counts': detect_valid_counts[final_index],
                        'detect_z_scores': detect_z_scores[final_index],
                        'detect_p_values': detect_p_values[final_index],
                    }
                
                    # 逐bit添加最终结果（仅最后一个值）
                    for i in range(len(bits)):
                        record_params[f'bits_green_count_{i}'] = bits_green_count[i][final_index]
                        record_params[f'bits_valid_count_{i}'] = bits_valid_count[i][final_index]
                        record_params[f'z_score_bits_{i}'] = z_score_bits[i][final_index]
                        record_params[f'z_p_value_bits_{i}'] = z_p_value_bits[i][final_index]
                
                    # 列名保持不变
                    columns_result = ['method_name', 'length', 'verify_z_score', 'verify_z_p_value', 'decode_bits', 'hit', 'hit_rate', 'verify_green_count', 
                                    'verify_generate_counts', 'verify_valid_counts', 'detected_gen_text', 
                                    'COUNTS', 'detect_green_counts', 'detect_valid_counts', 'detect_z_scores', 'detect_p_values']
                
                    columns_result += [f'bits_green_count_{i}' for i in range(len(bits))]
                    columns_result += [f'bits_valid_count_{i}' for i in range(len(bits))]
                    columns_result += [f'z_score_bits_{i}' for i in range(len(bits))]
                    columns_result += [f'z_p_value_bits_{i}' for i in range(len(bits))]
                
                    print('columns_result:', columns_result)
                
                    # 生成【单行】DataFrame（每个样本仅1行）
                    df_result = pd.DataFrame([record_params], columns=columns_result)
                
                    # 写入CSV（逻辑不变：首次写表头，后续追加）
                    if flg_record:
                        df_result.to_csv(detect_save_path, mode='w', index=False, header=True, columns=columns_result)
                        flg_record = False
                    else:
                        df_result.to_csv(detect_save_path, mode='a', index=False, header=False, columns=columns_result)

                elif 'omnimark' in method_name.lower():
                    if dummy_bits is not None:
                        bits = dummy_bits
                    else:
                        try:
                            bits = item.get('bits', "0101") # 如果没保存bits，默认给一个测试用的
                        except:
                            bits = "0"
                    
                    
                    decode_start_time = time.time()
                    # 调用我们刚刚在 detect_watermark_dump.py 中写好的 OmniMark 解码器
                    V_decode, decode_bits, hit, hit_rate, valid_tokens, z_scores_bits, p_values_bits, global_z_score, global_p_value = detector.decode_omnimark_multibit(
                        inputs=generate_tokens, c_key=c_key, bits=bits, window_size=window_size, start=start, stride=stride
                    )
                    decode_end_time = time.time()
                    total_decode_time += (decode_end_time - decode_start_time)
                    valid_sample_count += 1
                    
                    # 构造要保存的记录
                    result = {
                        'method_name': [method_name],
                        'length': [valid_tokens],
                        'decode_bits': [decode_bits],
                        'hit': [hit],
                        'hit_rate': [hit_rate],
                        'valid_count': [valid_tokens],
                        'global_z_score': [global_z_score],
                        'global_p_value': [global_p_value]
                    }
                    
                    # 把每个 bit 的 z-score 也存下来方便画图
                    for i in range(len(bits)):
                        result[f'z_score_bit_{i}'] = [z_scores_bits[i]]
                        
                    df_result = pd.DataFrame(result)
                    if flg_record:
                        df_result.to_csv(detect_save_path, mode='w', index=False, header=True)
                        flg_record = False
                    else:
                        df_result.to_csv(detect_save_path, mode='a', index=False, header=False)


                elif 'watermod' in method_name.lower():
                    try:
                        bits = item.get('bits', "0"*16)
                    except:
                        bits = "0"*16
                    decode_start_time = time.time()
                    C, decode_bits, hit, hit_rate, valid_tokens, z_score, p_value = detector.decode_watermod_multibit(
                        prompt_tokens=prompt_tokens, gen_tokens=generate_tokens, c_key=c_key, bits=bits, 
                        model=model_for_detection, k=4, window_size=window_size, start=start, stride=stride
                    )

                    # ================= 新增：记录解码结束时间并累加 =================
                    decode_end_time = time.time()
                    total_decode_time += (decode_end_time - decode_start_time)
                    valid_sample_count += 1
                    result = {
                        'method_name': [method_name],
                        'length': [valid_tokens],
                        'decode_bits': [decode_bits],
                        'hit': [hit],
                        'hit_rate': [hit_rate],
                        'valid_count': [valid_tokens],
                        'z_score': [z_score],
                        'p_value': [p_value]
                    }
                    
                    df_result = pd.DataFrame(result)
                    if flg_record:
                        df_result.to_csv(detect_save_path, mode='w', index=False, header=True)
                        flg_record = False
                    else:
                        df_result.to_csv(detect_save_path, mode='a', index=False, header=False)
                
                elif 'xmark' in method_name.lower():
                    try:
                        bits = item.get('bits', "0"*32) 
                    except:
                        bits = "0"*32
                    decode_start_time = time.time()
                    counts, decode_bits, hit, hit_rate, valid_tokens = detector.decode_xmark_multibit(
                        inputs=generate_tokens, bits=bits, window_size=window_size
                    )
                    decode_end_time = time.time()
                    total_decode_time += (decode_end_time - decode_start_time)
                    valid_sample_count += 1
                    result = {
                        'method_name': [method_name],
                        'length': [valid_tokens],
                        'decode_bits': [decode_bits],
                        'hit': [hit],
                        'hit_rate': [hit_rate],
                        'valid_count': [valid_tokens]
                    }
                        
                    df_result = pd.DataFrame(result)
                    if flg_record:
                        df_result.to_csv(detect_save_path, mode='w', index=False, header=True)
                        flg_record = False
                    else:
                        df_result.to_csv(detect_save_path, mode='a', index=False, header=False)
                        

                elif method_name == 'mpac':
                    try:
                        bits = item['bits']
                    except:
                        bits = "0"
                    decode_start_time = time.time()
                    # 这里的 true_bits (或者 bits) 是你从数据中读取出的该条文本嵌入的水印 bit 字符串
                    cl_total, decode_bits, hit, hit_rate, valid_tokens, z_scores_symbols = detector.decode_mpac_multibit(
                        # inputs=detect_gen_tokens, # 或者是 input_ids, 视你原本代码的变量名而定
                        inputs=generate_tokens,
                        c_key=c_key,
                        bits=bits, 
                        gamma=gamma
                    )
                    decode_end_time = time.time()
                    total_decode_time += (decode_end_time - decode_start_time)
                    valid_sample_count += 1
                    # 接着像 OmniMark 一样，把 Hit Rate, Z-scores 等结果记录下来
                    p_values = [detector._compute_p_value(z) for z in z_scores_symbols]
                    # 构造要保存的记录
                    result = {
                        'method_name': [method_name],
                        'length': [valid_tokens],
                        'decode_bits': [decode_bits],
                        'hit': [hit],
                        'hit_rate': [hit_rate],
                        'valid_count': [valid_tokens],
                        'global_z_score': [z_scores_symbols],
                        'global_p_value': [p_values]
                    }
                    
                    # 把每个 bit 的 z-score 也存下来方便画图
                    # for i in range(len(bits)):
                    #     result[f'z_score_bit_{i}'] = [z_scores_bits[i]]
                        
                    df_result = pd.DataFrame(result)
                    if flg_record:
                        df_result.to_csv(detect_save_path, mode='w', index=False, header=True)
                        flg_record = False
                    else:
                        df_result.to_csv(detect_save_path, mode='a', index=False, header=False)

                elif 'stealthink' in method_name.lower():
                    try:
                        bits = item.get('bits', "0101") # 获取该条目埋入的 message
                    except:
                        bits = "0"
                    decode_start_time = time.time()
                    # 调用我们在 detect_watermark_dump.py 中写好的 StealthInk 解码器
                    cl_total, decode_bits, hit, hit_rate, valid_tokens = detector.decode_stealthink_multibit(
                        inputs=generate_tokens, c_key=c_key, bits=bits, R=R, window_size=window_size, start=start, stride=stride
                    )
                    decode_end_time = time.time()
                    total_decode_time += (decode_end_time - decode_start_time)
                    valid_sample_count += 1
                    # 构造要保存为 DataFrame 的结果字典
                    result = {
                        'method_name': [method_name],
                        'length': [valid_tokens],
                        'decode_bits': [decode_bits],
                        'hit': [hit],
                        'hit_rate': [hit_rate],
                        'valid_count': [valid_tokens]
                    }
                    
                    # 写入 CSV (复用您的 flg_record 逻辑，如果是第一条则带表头，否则追加)
                    df_result = pd.DataFrame(result)
                    if flg_record:
                        df_result.to_csv(detect_save_path, mode='w', index=False, header=True)
                        flg_record = False
                    else:
                        df_result.to_csv(detect_save_path, mode='a', index=False, header=False)
                #
            if valid_sample_count > 0:
                avg_decode_time = total_decode_time / valid_sample_count
                print(f"\n" + "="*50)
                print(f"Total Decode Time: {total_decode_time:.4f} seconds")
                print(f"Valid Samples Processed: {valid_sample_count}")
                print(f"Average Decoding Time: {avg_decode_time:.4f} seconds/sample")
                print("="*50 + "\n")
            params['avg_decode_time_seconds'] = avg_decode_time
            with open(param_path, 'w', encoding='utf-8') as f:
                    json.dump(params, f, ensure_ascii=False, indent=4)
        
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
    parser.add_argument("--substitution_detect", action="store_true", help="detect watermark based on substituted text")
    parser.add_argument("--substitution_ratio", type=float, default=0.1, help="The ratio used during substitution attack")
    


    
    # 在 if __main__ == "__main__": 下面的 parser 定义中加上这几行：
    parser.add_argument("--baseline_method", type=str, default="omnimark", help="测试无水印数据时，强行指定探测器 (如 bimark, xmark, omnimark)")
    parser.add_argument("--baseline_bits", type=str, default="1010101010101010", help="测试无水印数据时，探测器寻找的假想目标比特")
    parser.add_argument("--c_key", type=int, default=8214793)
    parser.add_argument("--bit_idx_key", type=int, default=283519)
    parser.add_argument("--partition_seeds", type=list_of_ints, default=[int(x) for x in np.random.choice(10000, size=20, replace=False)])
    args = parser.parse_args()
    begin = time.time()
    main(args)
    end = time.time()
    print("time cost", end - begin)
