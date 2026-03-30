import torch
import numpy as np
from transformers import LogitsProcessor
from utils import prf
import time

class WatermarkOmniMark(LogitsProcessor):
    def __init__(self, tokenizer, device, vocab_size, c_key=530773, delta=2.5, window_size=2, bits='0'*16):
        self.tokenizer = tokenizer
        self.device = device
        self.vocab_size = vocab_size
        self.c_key = c_key
        self.delta = delta
        self.window_size = window_size
        self.bits = bits
        self.L = len(bits)
        
        # 将 "0101" 映射为 [-1, 1, -1, 1] 的目标方向向量 M
        m_list = [1 if b == '1' else -1 for b in self.bits]
        self.M = torch.tensor(m_list, device=device, dtype=torch.float32)
        
        self.prompt_len = None
        self.history_cache = {}  # 记录每个 batch 的 (生成的长度, 有效步数, 累加向量S)
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = input_ids.shape[0]
        current_len = input_ids.shape[1]
        
        # 记录初始 prompt 的长度，作为生成的起点
        if self.prompt_len is None:
            self.prompt_len = current_len
            
        new_scores = scores.clone()
        prefix = input_ids[:, -self.window_size:]
        c_seeds = prf(prefix, self.c_key)
        
        for i in range(batch_size):
            input_seq = input_ids[i]
            
            # 1. 状态恢复与更新：计算到目前为止，实际生成的累加指纹 S
            if i not in self.history_cache:
                S = torch.zeros(self.L, device=self.device)
                valid_steps = 0
                last_len = self.prompt_len
            else:
                last_len, valid_steps, S = self.history_cache[i]
            
            # 追溯从上一次调用到现在的“实际采样”的 token，累加它们的指纹
            for t in range(last_len, current_len):
                if t >= self.prompt_len + self.window_size: 
                    prev_prefix = input_seq[t - self.window_size : t]
                    token_sampled = input_seq[t].item()
                    
                    seed = prf(prev_prefix.unsqueeze(0), self.c_key)[0]
                    rng = np.random.default_rng(seed)
                    F_np = rng.integers(0, 2, size=(self.vocab_size, self.L)) * 2 - 1
                    F_sampled = torch.tensor(F_np[token_sampled], device=self.device, dtype=torch.float32)
                    
                    S += F_sampled
                    valid_steps += 1
            
            # 将当前状态写回缓存
            self.history_cache[i] = (current_len, valid_steps, S)
            
            if current_len < self.prompt_len + self.window_size:
                continue # context 不够，跳过提权
            
            # ====== 2. 闭环负反馈核心 (Delta-Sigma Modulation) ======
            # 目标轨迹：我们希望经过 valid_steps+1 步后，累加和达到多少？
            # 乘以 0.5 是一个工程调优技巧，表示不需要完美追求绝对值，只要方向一致即可
            target_S = (valid_steps + 1) * 0.5 * self.M 
            
            # 误差向量：当前最急需弥补的方向
            E = target_S - S
            
            # 3. 对词表进行打分：谁能最大程度修正当前的误差 E，谁就获得最高的提权
            rng = np.random.default_rng(c_seeds[i])
            F_np = rng.integers(0, 2, size=(self.vocab_size, self.L)) * 2 - 1
            F_candidates = torch.tensor(F_np, device=self.device, dtype=torch.float32)
            
            # 分数 = 候选特征与误差向量的内积 (归一化防止数值爆炸)
            score = torch.matmul(F_candidates, E) / self.L
            
            new_scores[i] += self.delta * score

        return new_scores