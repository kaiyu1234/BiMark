import torch
import numpy as np
from transformers import LogitsProcessor
from utils import prf
import collections
import pandas as pd
import matplotlib.pyplot as plt

class WatermarkOmniMark(LogitsProcessor):
    def __init__(self, tokenizer, device, vocab_size, extractor=None, value_head=None, c_key=530773, delta=2.5, window_size=2, bits='0'*16, top_k=50):
        self.tokenizer = tokenizer
        self.device = device
        self.vocab_size = vocab_size
        self.c_key = c_key
        self.delta = delta
        self.window_size = window_size
        self.bits = bits
        self.L = len(bits)
        self.top_k = top_k  # 记录 top_k 阈值
        
        # ================= 新增：ForesightMark 接口 =================
        self.extractor = extractor    # 隐藏层状态抓取器 (用于获取 h_t)
        self.value_head = value_head  # 训练好的价值网络 (Value Head)
        # ==========================================================

        # 将 "0101" 映射为 [-1, 1, -1, 1] 的目标方向向量 M
        m_list = [1 if b == '1' else -1 for b in self.bits]
        self.M = torch.tensor(m_list, device=device, dtype=torch.float32)
        
        self.error_dict = collections.defaultdict(list) # 键为 valid_steps，值为一个包含该步数下所有样本误差模长的列表
        self.prompt_len = None
        self.history_cache = {}  # 记录每个 batch 的 (生成的长度, 有效步数, 累加向量S)
        
        # ================= 用于记录画图数据的变量 =================
        self.record_history = True # 如果批量跑 10000 条数据收集，建议在外部将其设为 False 防止爆内存
        self.S_history = []        # 记录轨迹
        self.H_history = []        # 记录 Heatmap 得分
        # ==========================================================
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = input_ids.shape[0]
        current_len = input_ids.shape[1]
        
        # 记录初始 prompt 的长度，作为生成的起点
        if self.prompt_len is None:
            self.prompt_len = current_len
            
        new_scores = scores.clone()
        prefix = input_ids[:, -self.window_size:]
        c_seeds = prf(prefix, self.c_key)
        
        # 准备记录整个 batch 的 E_t 向量，以备传入 Value Head
        E_t_batch = torch.zeros((batch_size, self.L), device=self.device)

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
                    # O(1) 优化：直接用哈希得出对应词的伪随机数
                    token_seed = (int(seed) + int(token_sampled) * 2654435761) % (2**32)
                    token_rng = np.random.default_rng(token_seed)
                    # 直接只生成这 1 个词的 L 维向量 (长度16)
                    F_sampled_np = token_rng.integers(0, 2, size=self.L) * 2 - 1
                    F_sampled = torch.tensor(F_sampled_np, device=self.device, dtype=torch.float32)
                    
                    # 记录画图数据 (只记录 batch 第一条)
                    if i == 0 and self.record_history:
                        target_S_used = (valid_steps + 1) * 0.4 * self.M
                        E_used = target_S_used - S
                        H_val = F_sampled * E_used
                        
                        self.H_history.append(H_val.cpu().numpy().copy())
                        self.S_history.append((S + F_sampled).cpu().numpy().copy())
                    
                    S += F_sampled
                    valid_steps += 1
            
            # 将当前状态写回缓存
            self.history_cache[i] = (current_len, valid_steps, S)
            
            if current_len < self.prompt_len + self.window_size:
                continue # context 不够，跳过提权
            
            # ====== 2. 闭环负反馈核心 (Delta-Sigma Modulation) ======
            target_S = (valid_steps + 1) * 0.4 * self.M 
            # 误差向量：当前最急需弥补的方向
            E = target_S - S
            
            # 将误差保存到 Batch 列表中，留给后方的 Value Head
            E_t_batch[i] = E

            # 按步数记录所有的误差模长
            if torch.norm(S, p=2).item() == 0:
                current_e_norm = 0
            else:
                current_e_norm = torch.norm(E, p=2).item() 
            self.error_dict[valid_steps].append(current_e_norm)

            # ====== 3. 动态自适应 Delta ======
            delta_0 = 0.8
            gamma = 0.99
            min_delta = 0.1
            if valid_steps < 40:
                current_delta = delta_0
            else:
                current_delta = delta_0 * (gamma ** (valid_steps - 40))
                current_delta = max(current_delta, min_delta)
                
            base_seed = int(c_seeds[i])
            
            # ====== 4. Top-K 极速过滤与贪心打分 ======
            if self.top_k > 0 and self.top_k < self.vocab_size:
                # 1. 找出分数最高的前 K 个词汇的索引
                _, top_k_indices = torch.topk(scores[i], self.top_k)
                top_k_list = top_k_indices.tolist()
                
                # 2. 只为这 K 个候选词生成指纹（从生成 128256 次降维打击到只生成 50 次）
                F_candidates_list = []
                for idx in top_k_list:
                    token_seed = (base_seed + idx * 2654435761) % (2**32)
                    token_rng = np.random.default_rng(token_seed)
                    F_candidates_list.append(token_rng.integers(0, 2, size=self.L) * 2 - 1)
                    
                F_candidates_k = torch.tensor(np.array(F_candidates_list), device=self.device, dtype=torch.float32)
                
                # 3. 计算这 K 个词的调整分数
                score_k = torch.matmul(F_candidates_k, E) / self.L
                
                # 4. 直接把算好的分数加到原 Logits 对应的 Top-K 位置上
                new_scores[i][top_k_indices] += current_delta * score_k

            else:
                # 全词表处理备选
                rng = np.random.default_rng(base_seed)
                F_np = rng.integers(0, 2, size=(self.vocab_size, self.L)) * 2 - 1
                F_candidates = torch.tensor(F_np, device=self.device, dtype=torch.float32)
                score = torch.matmul(F_candidates, E) / self.L
                new_scores[i] += current_delta * score

        # =========================================================
        # 🌟 ForesightMark 核心：注入 Value Head 的长远预测！
        # =========================================================
        if self.value_head is not None and self.extractor is not None:
            h_t_batch = self.extractor.h_t  # 抓取当前 batch 的隐状态
            
            if h_t_batch is not None:
                with torch.no_grad():
                    # 传入隐状态 (h_t) 和误差 (E_t)，网络内部会自动进行 Padding 适配
                    value_logits = self.value_head(h_t_batch.to(torch.bfloat16), E_t_batch.to(torch.bfloat16))
                    
                    # value_logits shape: (batch_size, vocab_size)
                    # 将长期价值直接叠加到全体词表的 Logits 上
                    # alpha (权重) 可以根据需要调节，例如 1.0 或 2.0
                    alpha = 1.0 
                    new_scores += alpha * value_logits

        return new_scores

    def export_and_plot(self, save_name="mean_error"):
        """等所有文本生成完毕后，调用此方法导出数据和画图"""
        if not self.error_dict:
            print("警告：没有收集到任何误差数据！")
            return

        avg_errors = []
        for step in sorted(self.error_dict.keys()):
            norms_at_step = self.error_dict[step]
            avg_errors.append({
                "valid_steps": step,
                "mean_error_norm": np.mean(norms_at_step),
                "std_error_norm": np.std(norms_at_step),
                "sample_count": len(norms_at_step)
            })

        df_errors = pd.DataFrame(avg_errors)
        df_errors.to_csv(f"{save_name}.csv", index=False)

        plt.figure(figsize=(8, 5))
        plt.plot(df_errors["valid_steps"], df_errors["mean_error_norm"], label="Mean Error Norm ||E||", color="#1f77b4")
        plt.fill_between(
            df_errors["valid_steps"], 
            df_errors["mean_error_norm"] - df_errors["std_error_norm"], 
            df_errors["mean_error_norm"] + df_errors["std_error_norm"], 
            color="#1f77b4", alpha=0.2
        )
        plt.xlabel("Generation Steps")
        plt.ylabel("Average Error Norm")
        plt.title("Convergence of Mean Error Norm")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_name}.png", dpi=300)
        print(f"✅ 误差数据已保存为 {save_name}.csv 和 {save_name}.png")
        plt.close() # 释放内存