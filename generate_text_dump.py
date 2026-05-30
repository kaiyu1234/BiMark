import argparse
from tqdm import tqdm
import torch
import random
import os
import seaborn as sns
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "30"
from transformers import AutoModelForCausalLM,AutoModelForSeq2SeqLM, AutoTokenizer, LogitsProcessorList
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, LogitsProcessorList
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np
from random import randint
from utils import record_data, load_data, read_jsonl_file
from WatermarkBimark import WatermarkBimark
from WatermarkOmniMark import WatermarkOmniMark  # 添加这行
from WatermarkWaterMod import WatermarkWaterMod  # <--- 新增这行
import time
from datetime import datetime
import os
import json
from WatermarkStealthInk import WatermarkStealthInk  # <--- 添加这行
from WatermarkMPAC import WatermarkMPAC  # <--- 添加这行
from WatermarkXMark import WatermarkXMark
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print('cuda available:', torch.cuda.is_available())
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# def save_human_written_as_json(prompt_idx, prompts, human_written, save_dir):
#     print('save_dir', save_dir)
#     json_data = []
#     for idx, prompt, completion in zip(prompt_idx, prompts, human_written):
#         json_data.append({
#             "prompt_idx": idx,
#             "prompt": prompt,
#             "human_completion": completion
#         })
    
#     # Save JSON data
#     with open(os.path.join(save_dir, "human_written.jsonl"), 'w', encoding='utf-8') as f:
#         json.dump(json_data, f, ensure_ascii=False, indent=4)

def plot_omnimark_paper_figures(processor: WatermarkOmniMark, save_dir="./"):
    """
    processor: 必须是跑完生成的 WatermarkOmniMark 实例
    save_dir: 图表保存路径
    """
    S_hist = np.array(processor.S_history)  # Shape: (steps, L)
    H_hist = np.array(processor.H_history)  # Shape: (steps, L)
    M = processor.M.cpu().numpy()           # Shape: (L,)
    L = processor.L
    steps = S_hist.shape[0]
    font_path = "/root/autodl-tmp/fonts/TIMES.TTF"
    print("font_path =", font_path)
    print("repr(font_path) =", repr(font_path))
    print("isabs =", os.path.isabs(font_path))
    print("exists =", os.path.exists(font_path))
    print("cwd =", os.getcwd())
    if steps == 0:
        print("未记录到任何生成步数，无法作图。")
        return
    
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"字体文件不存在: {font_path}")

    # 注册字体
    fm.fontManager.addfont(font_path)
    
    # 获取字体属性
    prop = fm.FontProperties(fname=font_path)
    font_name = prop.get_name()
    plt.rcParams.update({
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    })
    
    print("识别到的字体名:", font_name)
    
    # 全局设置
    plt.rcParams["font.family"] = font_name
    plt.rcParams["font.size"] = 20
    plt.rcParams["axes.unicode_minus"] = False

    # ==========================================
    # 1. Bit-level Trajectory Plot
    # ==========================================
    plt.figure(figsize=(8, 4))
    colors = sns.color_palette("husl", L)
    
    for i in range(L):
        # 画出每个 bit 的实际累加曲线
        plt.plot(range(1, steps+1), S_hist[:, i], color=colors[i], 
                 label=f'Bit {i} ($m_i={int(M[i])}$)', linewidth=1.5)

    # 画出期望对齐的 Target Trajectory 辅助线
    plt.plot(range(1, steps+1), 0.4 * np.arange(1, steps+1), 'k--', 
             linewidth=2, alpha=0.8, label='Target $+0.4t$')
    plt.plot(range(1, steps+1), -0.4 * np.arange(1, steps+1), 'k--', 
             linewidth=2, alpha=0.8, label='Target $-0.4t$')

    # [修改点 1] 放大坐标轴标题 (可以去掉 fontproperties=prop，因为你已经全局设置了字体，直接用 fontsize 即可避免冲突)
    plt.xlabel('Valid Steps $t$', fontsize=20)
    plt.ylabel('Cumulative Sum $S_t[i]$', fontsize=20)
    # plt.title('Bit-level Trajectory for OmniMark', fontsize=15)
    # [修改点 2] 放大坐标轴的刻度数字
    plt.tick_params(axis='both', which='major', labelsize=20)
    # 将图例放在图外侧，避免遮挡曲线
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., fontsize=15)
    plt.tight_layout()
    traj_path = os.path.join(save_dir, 'bit_trajectory.pdf') # 保存为 PDF 适合论文排版
    plt.savefig(traj_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Trajectory 图已保存至: {traj_path}")

    # ==========================================
    # 2. Score Heatmap Plot
    # ==========================================
    plt.figure(figsize=(10, 4))
    # 将数据转置，使得纵轴为 bit dimension，横轴为生成步数 t
    # 使用 RdYlBu_r (红黄蓝渐变)，正值为暖色，负值为冷色，直观展示正向/反向得分修正
    ax = sns.heatmap(H_hist.T, cmap='RdYlBu_r', center=0, 
                     cbar_kws={'label': '$F(c_t, v_t)[i] \\times E_t[i]$'})
    
    plt.xlabel('Valid Steps $t$', fontproperties=prop)
    plt.ylabel('Bit Dimension $i$',fontproperties=prop)
    # ax.invert_yaxis() # 如果你想让 Bit 0 在最下方可以取消注释
    
    plt.tight_layout()
    heatmap_path = os.path.join(save_dir, 'score_heatmap.pdf')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Heatmap 图已保存至: {heatmap_path}")



def save_human_written_as_json(prompt_idx, prompts, human_written, save_dir):
    print('save_dir', save_dir)
    
    # Save as JSONL (逐行写入)
    with open(os.path.join(save_dir, "human_written.jsonl"), 'w', encoding='utf-8') as f:
        for idx, prompt, completion in zip(prompt_idx, prompts, human_written):
            # 构建单行的 JSON 对象
            line_obj = {
                "prompt_idx": idx,
                "prompt": prompt,
                "human_completion": completion
            }
            # 使用 json.dumps 序列化单行，并写入文件，最后加换行符
            f.write(json.dumps(line_obj, ensure_ascii=False) + '\n')

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
    
    auth_token = "***"
    
    # # model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map='auto', torch_dtype=torch.bfloat16, use_auth_token=auth_token)
    # model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, device_map='auto', torch_dtype=torch.bfloat16)
    # # model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map='auto', torch_dtype=torch.bfloat16)
    # # tokenizer = AutoTokenizer.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, use_auth_token=auth_token)
    # tokenizer = AutoTokenizer.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    # tokenizer.pad_token = tokenizer.eos_token
    # # tokenizer.padding_side = 'left'
    if "mbart" in args.model_name.lower() or "bart" in args.model_name.lower():
        from transformers import AutoModelForSeq2SeqLM
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, device_map='auto')
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map='auto', torch_dtype=torch.bfloat16, token=auth_token)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name,token=auth_token)
    is_seq2seq = model.config.is_encoder_decoder
    if not is_seq2seq:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
    else:
        # BART 等模型通常不需要干预，默认即可 (right padding, 且自带 pad_token)
        tokenizer.padding_side = 'right'
    
    prob_delta = args.prob_delta
    forced_bos_token_id = None
    if "mbart" in args.model_name.lower():
        tokenizer.src_lang = "ro_RO"  # 源语言：罗马尼亚语
        forced_bos_token_id = tokenizer.lang_code_to_id["en_XX"]  # 强制生成的起始符（目标语言）：英语








    
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
                truncation=True,
                max_length=1024   # 新增参数：明确限制为 BART 的最大输入长度
            )
        
        input_ids = inputs["input_ids"].to(model.device)
        print('input_ids.size()', input_ids.size())
        attention_mask = inputs["attention_mask"].to(model.device)
        
        prompt_tokens_len = (input_ids).size(-1)
        print('prompt_tokens_len', prompt_tokens_len)
        
        print('bitlen', len(args.message))
        
        if args.method.lower() == 'bimark':
            if hasattr(args, 'message_len') and args.message_len > 0:
                tmp = []
                for _ in range(args.message_len):
                    tmp.append(str(randint(0, 1)))
                bits = "".join(tmp)
                print(f"随机生成的 {args.message_len} 位水印信息为: {bits}")
            
            # 兼容原有的逻辑
            elif args.random_message:
                tmp = []
                for _ in range(len(args.message)):
                    tmp.append(str(randint(0, 1)))
                bits = "".join(tmp)
                print(f"随机生成的水印信息为: {bits}")
            else:
                bits = args.message
            watermark_processor_bimark = WatermarkBimark(tokenizer=tokenizer, vocab_size=vocab_size, device=device, top_k=args.top_k, partition_seeds=args.partition_seeds, 
                                                         c_key=args.c_key, bit_idx_key=args.bit_idx_key, delta=args.prob_delta, window_size=args.window_size, bits=bits)
            generate_args_bimark = {'logits_processor': [watermark_processor_bimark],  'max_new_tokens': args.max_new_tokens,'temperature': args.temperature,  'attention_mask': attention_mask, 
                                  'do_sample': args.do_sample, 
                                    'top_k': args.top_k,
                                    'max_length': 60,          
                                    'min_length': 10,            # 强制要求摘要不能太短}
                                   }
            
            # 把语言 ID 传给模型
            if forced_bos_token_id is not None:
                generate_args_bimark['forced_bos_token_id'] = forced_bos_token_id

        elif args.method.lower() == 'mpac':
            if hasattr(args, 'message_len') and args.message_len > 0:
                tmp = []
                for _ in range(args.message_len):
                    tmp.append(str(randint(0, 1)))
                bits = "".join(tmp)
                print(f"随机生成的 {args.message_len} 位水印信息为: {bits}")
            
            # 兼容原有的逻辑
            elif args.random_message:
                tmp = []
                for _ in range(len(args.message)):
                    tmp.append(str(randint(0, 1)))
                bits = "".join(tmp)
                print(f"随机生成的水印信息为: {bits}")
            else:
                bits = args.message
                
            watermark_processor_mpac = WatermarkMPAC(
                tokenizer=tokenizer, vocab_size=vocab_size, 
                window_size=args.window_size, bits=bits, c_key=args.c_key, gamma=0.25, delta=args.prob_delta
            )
            generate_args_mpac = {
                'logits_processor': LogitsProcessorList([watermark_processor_mpac]), 
                'max_new_tokens': args.max_new_tokens,
                'temperature': args.temperature, 
                'attention_mask': attention_mask, 
                'do_sample': args.do_sample, 
                'top_k': args.top_k,
                'max_length': 60,          
                'min_length': 10,
            }

        elif args.method.lower() == 'xmark':
            if hasattr(args, 'message_len') and args.message_len > 0:
                tmp = []
                for _ in range(args.message_len):
                    tmp.append(str(randint(0, 1)))
                bits = "".join(tmp)
                print(f"随机生成的 {args.message_len} 位水印信息为: {bits}")
            
            # 兼容原有的逻辑
            elif args.random_message:
                tmp = []
                for _ in range(len(args.message)):
                    tmp.append(str(randint(0, 1)))
                bits = "".join(tmp)
                print(f"随机生成的水印信息为: {bits}")
            else:
                bits = args.message
                
            watermark_processor_xmark = WatermarkXMark(
                tokenizer=tokenizer, device=device, vocab_size=vocab_size,
                delta=args.prob_delta, window_size=args.window_size, bits=bits
            )
            generate_args_xmark = {
                'logits_processor': LogitsProcessorList([watermark_processor_xmark]), 
                'max_new_tokens': args.max_new_tokens,
                'temperature': args.temperature, 
                'attention_mask': attention_mask, 
                'do_sample': args.do_sample, 
                'top_k': args.top_k
            }
        

        elif args.method.lower() == 'watermod':
            if hasattr(args, 'message_len') and args.message_len > 0:
                tmp = []
                for _ in range(args.message_len):
                    tmp.append(str(randint(0, 1)))
                bits = "".join(tmp)
                print(f"随机生成的 {args.message_len} 位水印信息为: {bits}")
            
            # 兼容原有的逻辑
            elif args.random_message:
                tmp = []
                for _ in range(len(args.message)):
                    tmp.append(str(randint(0, 1)))
                bits = "".join(tmp)
                print(f"随机生成的水印信息为: {bits}")
            else:
                bits = args.message
                
            watermark_processor_watermod = WatermarkWaterMod(
                tokenizer=tokenizer, vocab_size=vocab_size, device=device,
                c_key=args.c_key, delta=args.prob_delta, window_size=args.window_size, bits=bits, k=4
            )
            generate_args_watermod = {
                'logits_processor': LogitsProcessorList([watermark_processor_watermod]), 
                'max_new_tokens': args.max_new_tokens,
                'temperature': args.temperature, 
                'attention_mask': attention_mask, 
                'do_sample': args.do_sample, 
                'top_k': args.top_k
            }
            
        elif args.method.lower() == 'omnimark':
            if hasattr(args, 'message_len') and args.message_len > 0:
                tmp = []
                for _ in range(args.message_len):
                    tmp.append(str(randint(0, 1)))
                bits = "".join(tmp)
                print(f"随机生成的 {args.message_len} 位水印信息为: {bits}")
            
            # 兼容原有的逻辑
            elif args.random_message:
                tmp = []
                for _ in range(len(args.message)):
                    tmp.append(str(randint(0, 1)))
                bits = "".join(tmp)
                print(f"随机生成的水印信息为: {bits}")
            else:
                bits = args.message
                
            # 这里 prob_delta 被复用为 logits 的偏移常数 delta（建议在入参时设置为 2.0-5.0，而不再是 0.2）
            watermark_processor_omnimark = WatermarkOmniMark(
                tokenizer=tokenizer, vocab_size=vocab_size, device=device,
                c_key=args.c_key, delta=args.prob_delta, window_size=args.window_size, bits=bits,top_k=args.top_k
            )
            generate_args_omnimark = {
                'logits_processor': LogitsProcessorList([watermark_processor_omnimark]), 
                'max_new_tokens': args.max_new_tokens,
                'temperature': args.temperature, 
                
                'max_length': 60,           # 摘要最大长度
                'min_length': 10,            # 强制要求摘要不能太短
                
                'attention_mask': attention_mask, 
                'do_sample': args.do_sample, 
                'top_k': args.top_k
                 # 👇 下面这一块是我帮你改好、专门提分的固定写法
                # 'do_sample': False,        # 关掉采样，改用 beam search，分数暴涨
                # #'num_beams': 4,            # BART 摘要最常用、最稳
                # #'length_penalty': 2.0,
                # 'top_k': args.top_k,
                # #'no_repeat_ngram_size': 3,
                # #'early_stopping': True,
            }

            
            # 把语言 ID 传给模型
            if forced_bos_token_id is not None:
                generate_args_omnimark['forced_bos_token_id'] = forced_bos_token_id




        elif args.method.lower() == 'stealthink':
            if hasattr(args, 'message_len') and args.message_len > 0:
                tmp = []
                for _ in range(args.message_len):
                    tmp.append(str(randint(0, 1)))
                bits = "".join(tmp)
                print(f"随机生成的 {args.message_len} 位水印信息为: {bits}")
            
            # 兼容原有的逻辑
            elif args.random_message:
                tmp = []
                for _ in range(len(args.message)):
                    tmp.append(str(randint(0, 1)))
                bits = "".join(tmp)
                print(f"随机生成的水印信息为: {bits}")
            else:
                bits = args.message

            R = 0.25  # 您可以在这里修改 R，例如 0.5 或 0.25
            import math
            chunk_capacity = int(math.log2(1/R)) # 如果 R=0.25, capacity=2
            
            # 如果 message 长度不能被 capacity 整除，在末尾补 0
            if len(bits) % chunk_capacity != 0:
                bits += '0' * (chunk_capacity - len(bits) % chunk_capacity)
                
            # 将二进制字符串（如 "01011001"）按 chunk_capacity 分组并转为整数（如 [1, 1, 2, 1]）
            embedded_message = [int(bits[i:i+chunk_capacity], 2) for i in range(0, len(bits), chunk_capacity)]
            converted_msg_length = len(embedded_message)
            # ==========================================
            
            watermark_processor_stealthink = WatermarkStealthInk(
                vocab_size=vocab_size, 
                embedded_message=embedded_message,
                n_gram_len=args.window_size,
                R=R, # 传入 0.25
                converted_msg_length=converted_msg_length,
                c_key=args.c_key
            )
            generate_args_stealthink = {
                'logits_processor': LogitsProcessorList([watermark_processor_stealthink]), 
                'max_new_tokens': args.max_new_tokens,
                'temperature': args.temperature, 
                'attention_mask': attention_mask, 
                'do_sample': args.do_sample, 
                'top_k': args.top_k,
                'max_length': 60,          
                'min_length': 10,
            }

        
        elif args.method.lower() == 'no_watermark':
            generate_args_null = {'max_new_tokens': args.max_new_tokens, 'attention_mask': attention_mask,  'temperature': args.temperature,
                                  'do_sample': args.do_sample, 'top_k': args.top_k}
              
            # 把语言 ID 传给模型
            if forced_bos_token_id is not None:
                generate_args_null['forced_bos_token_id'] = forced_bos_token_id
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
                # "max_length": args.max_new_tokens,
                'max_new_tokens': args.max_new_tokens,
                "time_stamp": time_str,
            }
            record_data(batch_prompts, tokenizer, null_token[:,prompt_tokens_len:].tolist(), idx_list, save_dir, params)
            # record_data(batch_prompts, tokenizer, null_token.tolist(), idx_list, save_dir, params) # 下游任务时使用
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


        if args.method.lower() == 'mpac':
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            with torch.inference_mode():
                new_token = model.generate(input_ids, **generate_args_mpac)
            params = {
                "method": "mpac",
                "model_name": args.model_name,
                "vocab_size": vocab_size,
                # "bits_len": len(args.message),
                "bits_len": len(bits),
                "gamma": 0.25,
                "prob_delta": prob_delta,
                "window_size": args.window_size,
                "c_key": args.c_key,
                "time_stamp": time_str,
            }
            record_data(batch_prompts, tokenizer, new_token[:,prompt_tokens_len:].tolist(), idx_list, save_dir, params, bits=bits)
            # record_data(batch_prompts, tokenizer, new_token.tolist(), idx_list, save_dir, params) # 下游任务时使用


        if args.method.lower() == 'xmark':
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            with torch.inference_mode():
                new_token = model.generate(input_ids, **generate_args_xmark)
            params = {
                "method": "xmark",
                "model_name": args.model_name,
                "vocab_size": vocab_size,
                # "bits_len": len(args.message),
                "bits_len": len(bits),
                "prob_delta": prob_delta,  
                "window_size": args.window_size,
                "time_stamp": time_str,
            }
            record_data(batch_prompts, tokenizer, new_token[:,prompt_tokens_len:].tolist(), idx_list, save_dir, params, bits=bits)
            # record_data(batch_prompts, tokenizer, new_token.tolist(), idx_list, save_dir, params) # 下游任务时使用
            
        if args.method.lower() == 'omnimark':
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            with torch.inference_mode():
                new_token = model.generate(input_ids, **generate_args_omnimark)
            params = {
                "method": "omnimark",
                "model_name": args.model_name,
                "vocab_size": vocab_size,
                # "bits_len": len(args.message),
                "bits_len": len(bits),
                "prob_delta": prob_delta,  # 这里的 delta 表示 logits offset
                "window_size": args.window_size,
                "c_key": args.c_key,
                "time_stamp": time_str,
            }
            record_data(batch_prompts, tokenizer, new_token[:,prompt_tokens_len:].tolist(), idx_list, save_dir, params, bits=bits)
            # record_data(batch_prompts, tokenizer, new_token.tolist(), idx_list, save_dir, params, bits=bits) # 下游任务时使用 
            if batch == 0:
                plot_omnimark_paper_figures(processor=watermark_processor_omnimark, save_dir=save_dir)


        if args.method.lower() == 'watermod':
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            with torch.inference_mode():
                new_token = model.generate(input_ids, **generate_args_watermod)
            params = {
                "method": "watermod",
                "model_name": args.model_name,
                "vocab_size": vocab_size,
                "bits_len": len(args.message),
                "prob_delta": prob_delta,  # delta bias
                "window_size": args.window_size,
                "c_key": args.c_key,
                "k_base": 4, # 记录 k
                "time_stamp": time_str,
            }
            record_data(batch_prompts, tokenizer, new_token[:,prompt_tokens_len:].tolist(), idx_list, save_dir, params, bits=bits)
            # record_data(batch_prompts, tokenizer, new_token.tolist(), idx_list, save_dir, params, bits=bits) # 下游任务时使用 
        if args.method.lower() == 'stealthink':
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            with torch.inference_mode():
                new_token = model.generate(input_ids, **generate_args_stealthink)
            params = {
                "method": "stealthink",
                "model_name": args.model_name,
                "vocab_size": vocab_size,
                # "bits_len": len(args.message),
                "bits_len": len(bits),
                "window_size": args.window_size,
                "c_key": args.c_key,
                "R": R,  # 记录下生成的 R 值
                "time_stamp": time_str,
            }
            record_data(batch_prompts, tokenizer, new_token[:,prompt_tokens_len:].tolist(), idx_list, save_dir, params, bits=bits)
            # record_data(batch_prompts, tokenizer, new_token.tolist(), idx_list, save_dir, params, bits=bits) # 下游任务时使用 

            
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
                # "bits_len": len(args.message),
                "bits_len": len(bits),
                "top_k": args.top_k,
                "do_sample": args.do_sample,
                'temperature': args.temperature, 
                "max_length": args.max_new_tokens,
                "prob_delta": prob_delta,
                "window_size": args.window_size,
                "c_key": args.c_key,
                "bit_idx_key": args.bit_idx_key,
                "partition_seeds": args.partition_seeds,
                "time_stamp": time_str,
            }
            record_data(batch_prompts, tokenizer, new_token[:,prompt_tokens_len:].tolist(), idx_list, save_dir, params, bits=bits)
            # record_data(batch_prompts, tokenizer, new_token.tolist(), idx_list, save_dir, params, bits=bits)

        pbar.update(1)
    
    end = time.time()
    runtime = end - begin
    print("time cost", runtime)
    watermark_processor_omnimark.export_and_plot(save_name="mean_error_trajectory")
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
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--do_sample", action='store_true')
    parser.add_argument("--message_len", type=int, default=0, help="The length of the randomly generated binary message.")
    
    args = parser.parse_args()
    main(args)
 