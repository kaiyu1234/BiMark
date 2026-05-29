import torch
import math
import json
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

# 导入你最新优化的 OmniMark 逻辑
from WatermarkOmniMark import WatermarkOmniMark
from utils import prf

# ==========================================
# 1. 奖励计算器 (Reward Calculator)
# ==========================================
def calculate_reward(model, tokenizer, prompt_ids, generated_ids, c_key, bits, device):
    """
    计算一条完整轨迹的全局 Reward = alpha * Watermark_Score - beta * PPL
    注意：此处的解码逻辑必须与你最新的 WatermarkOmniMark 中的 token_seed 生成逻辑完全一致！
    """
    window_size = 2
    L = len(bits)
    
    # 将 bits 转为目标向量 M (+1 / -1)
    m_list = [1 if b == '1' else -1 for b in bits]
    M = torch.tensor(m_list, device=device, dtype=torch.float32)
    
    # ---------------------------------
    # (A) 计算 PPL (流畅度惩罚)
    # ---------------------------------
    full_ids = torch.cat([prompt_ids, generated_ids], dim=-1).unsqueeze(0)
    with torch.no_grad():
        # 用 LLM 算一遍 Cross Entropy Loss，其指数就是 PPL
        outputs = model(input_ids=full_ids, labels=full_ids)
        loss = outputs.loss.item()
        ppl = math.exp(loss) if loss < 20 else 10000.0 # 防止极端崩坏的值

    # ---------------------------------
    # (B) 计算 Watermark Score (提取匹配度)
    # ---------------------------------
    S = torch.zeros(L, device=device)
    valid_steps = 0
    full_ids_1d = full_ids.squeeze(0)
    prompt_len = prompt_ids.shape[0]
    seq_len = full_ids_1d.shape[0]
    
    # 模拟检测端的解码累加过程
    for t in range(prompt_len, seq_len):
        if t >= prompt_len + window_size - 1:
            prev_prefix = full_ids_1d[t - window_size : t]
            token_sampled = full_ids_1d[t].item()
            
            # 【核心对齐】: 使用你最新发明的单 Token Seed 高效映射算法！
            seed = prf(prev_prefix.unsqueeze(0), c_key)[0]
            token_seed = (int(seed) + int(token_sampled) * 2654435761) % (2**32)
            token_rng = np.random.default_rng(token_seed)
            
            # 直接生成这 1 个词的 L 维向量
            F_sampled_np = token_rng.integers(0, 2, size=L) * 2 - 1
            F_sampled = torch.tensor(F_sampled_np, device=device, dtype=torch.float32)
            
            S += F_sampled
            valid_steps += 1
            
    # 计算累加向量 S 与目标向量 M 的余弦相似度/内积匹配度
    # 因为你的代码目标是 S 逼近 0.4 * valid_steps * M
    # 所以 (S * M) / valid_steps 的理想值大概是 0.4
    if valid_steps > 0:
        watermark_score = torch.dot(S, M).item() / valid_steps 
    else:
        watermark_score = 0.0

    # ---------------------------------
    # (C) 综合 Reward
    # ---------------------------------
    # 为了保证价值网络学到好东西，必须重奖 PPL 优秀的句子
    # Alpha 和 Beta 的比例可以调：希望水印更强增大alpha；希望语句更自然增大beta
    alpha = 15.0  # 因为 watermark_score 均值大概在 0.2~0.4 之间
    beta = 1.0
    reward = alpha * watermark_score - beta * ppl
    
    return reward, watermark_score, ppl

# ==========================================
# 2. 数据采样主流程
# ==========================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading Model and Tokenizer...")
    model_name = "meta-llama/Meta-Llama-3.1-8B" # 替换为你的本地路径
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 强制用 bfloat16 加快生成并节省显存
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    
    # 全局参数与你的 OmniMark 参数对应
    c_key = 530773
    bits = '1010101010101010' # 16-bit
    vocab_size = len(tokenizer)
    gen_length = 200 # 每句话往后生成 200 个 token
    
    # TODO: 替换为你 utils.py 里的 load_data 逻辑
    # 比如: prompt_idx, prompts, human_written, num_test = load_data('c4', prompt_len=50, num_test=10000)
    # dataset_prompts = [
    #     "The most important thing in life is",
    #     "Artificial intelligence will fundamentally change",
    #     "In a galaxy far far away, there was",
    # ] * 1000 # 这里仅为示例，正式运行请换成真实的 10000 个 C4 prompt

    from utils import load_data
    print("Loading real C4 dataset for offline training...")
    _, prompts, _, _ = load_data(
        dataset_name='c4', 
        prompt_len=50,       # 给 LLM 50 个词作为前缀提示
        num_test=5000,      # 需要 10,000 条训练数据
        ds_start_point=0     # 从头开始取
    )


    
    output_file = "training_data.jsonl"
    print(f"Start generating trajectories. Saving to {output_file}...")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for i, prompt_text in enumerate(tqdm(prompts)):
            
            input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
            prompt_len = input_ids.shape[1]
            
            # 使用你最新的 OmniMark 代码初始化
            omni_processor = WatermarkOmniMark(
                tokenizer=tokenizer,
                device=device,
                vocab_size=vocab_size,
                c_key=c_key,
                delta=0.8,         # 基础强度，你的代码内部也有动态规划
                window_size=2,
                bits=bits,
                top_k=50           # 与你的最新逻辑对齐
            )
            logits_processor = LogitsProcessorList([omni_processor])
            
            # 💡 【核心探索设定】：开启 do_sample!
            # 依靠 temperature=1.0 和 top_k=50，让模型在好几个合理词中随机选
            # 这样才会产生出“放弃眼前最高得分，意外获得更好语句”的价值轨迹
            with torch.no_grad():
                generated_outputs = model.generate(
                    input_ids,
                    max_new_tokens=gen_length,
                    min_new_tokens=gen_length,
                    logits_processor=logits_processor,
                    do_sample=True,      
                    temperature=1.0,     
                    top_k=50,            
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # 拆分获取生成的这一段 ids
            prompt_ids_1d = generated_outputs[0][:prompt_len]
            generated_ids_1d = generated_outputs[0][prompt_len:]
            
            # 计算轨迹的 Reward
            reward, w_score, ppl = calculate_reward(
                model=model, 
                tokenizer=tokenizer, 
                prompt_ids=prompt_ids_1d, 
                generated_ids=generated_ids_1d, 
                c_key=c_key, 
                bits=bits, 
                device=device
            )
            
            gen_text = tokenizer.decode(generated_ids_1d, skip_special_tokens=True)
            
            # 组装为字典落盘
            record = {
                "prompt_text": prompt_text,
                "generated_text": gen_text,
                "prompt_len": prompt_len,
                "reward": float(reward),
                "watermark_score": float(w_score),
                "ppl": float(ppl)
            }
            f.write(json.dumps(record) + "\n")
            
            if (i + 1) % 50 == 0:
                print(f" Sample {i+1} | Reward: {reward:.2f} | W_Score: {w_score:.4f} | PPL: {ppl:.2f}")

if __name__ == "__main__":
    main()