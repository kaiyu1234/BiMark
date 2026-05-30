import torch
import numpy as np
from transformers import LogitsProcessor
from utils import prf
import time
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import collections
import numpy as np
class WatermarkOmniMark(LogitsProcessor):
    def __init__(self, tokenizer, device, vocab_size, c_key=530773, delta=2.5, window_size=2, bits='0'*16,top_k=50):
        self.tokenizer = tokenizer
        self.device = device
        self.vocab_size = vocab_size
        self.c_key = c_key
        self.delta = delta
        self.window_size = window_size
        self.bits = bits
        self.L = len(bits)
        self.top_k = top_k  # 记录 top_k 阈值
        
        # 将 "0101" 映射为 [-1, 1, -1, 1] 的目标方向向量 M
        m_list = [1 if b == '1' else -1 for b in self.bits]
        self.M = torch.tensor(m_list, device=device, dtype=torch.float32)
        self.error_dict = collections.defaultdict(list) # 键为 valid_steps，值为一个包含该步数下所有样本误差模长的列表
        self.prompt_len = None
        self.history_cache = {}  # 记录每个 batch 的 (生成的长度, 有效步数, 累加向量S)
        # ================= 新增：用于记录画图数据的变量 =================
        self.record_history = True # 开启记录
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
                    token_seed = (int(seed) + int(token_sampled) * 2654435761) % (2**32)
                    token_rng = np.random.default_rng(token_seed)
                    # 直接只生成这 1 个词的 L 维向量 (长度16)
                    F_sampled_np = token_rng.integers(0, 2, size=self.L) * 2 - 1
                    F_sampled = torch.tensor(F_sampled_np, device=self.device, dtype=torch.float32)
                    # ================= 新增：计算并记录画图数据 =================
                    # 只记录 batch 中第一条序列的数据，方便画图
                    if i == 0 and self.record_history:
                        # 还原出采样当前 token 时，模型面向的 target 和 E
                        target_S_used = (valid_steps + 1) * 0.4 * self.M
                        E_used = target_S_used - S
                        # Heatmap 需要的值: F(c_t, v_t)[i] * E_t[i]
                        H_val = F_sampled * E_used
                        
                        self.H_history.append(H_val.cpu().numpy().copy())
                        self.S_history.append((S + F_sampled).cpu().numpy().copy())
                    # ==========================================================
                    
                    S += F_sampled
                    valid_steps += 1
            
            # 将当前状态写回缓存
            self.history_cache[i] = (current_len, valid_steps, S)
            
            if current_len < self.prompt_len + self.window_size:
                continue # context 不够，跳过提权
            
            # ====== 2. 闭环负反馈核心 (Delta-Sigma Modulation) ======
            # 目标轨迹：我们希望经过 valid_steps+1 步后，累加和达到多少？
            # 乘以 0.5 是一个工程调优技巧，表示不需要完美追求绝对值，只要方向一致即可
            # target_S = (valid_steps + 1) * 0.4 * self.M 
            target_S = (valid_steps + 1) * 0.4 * self.M 
            # 误差向量：当前最急需弥补的方向
            E = target_S - S



            # =================== 【修改代码：按步数记录所有的误差模长】 ===================
            if torch.norm(S, p=2).item() ==0:
                current_e_norm=0
            else:
                current_e_norm = torch.norm(E, p=2).item() 
            # 将当前样本在当前 valid_steps 下的误差，放入对应的列表中
            self.error_dict[valid_steps].append(current_e_norm)
            # ==============================================================================

            
     
            # 3. 对词表进行打分：谁能最大程度修正当前的误差 E，谁就获得最高的提权
            # rng = np.random.default_rng(c_seeds[i])
            # F_np = rng.integers(0, 2, size=(self.vocab_size, self.L)) * 2 - 1
            # F_candidates = torch.tensor(F_np, device=self.device, dtype=torch.float32)
            
            # 分数 = 候选特征与误差向量的内积 (归一化防止数值爆炸)
            # score = torch.matmul(F_candidates, E) / self.L
            # score = torch.matmul(F_candidates, E_bounded) / self.L

            # -------------------------------消融--------------------------------------------------------------------
            # score = torch.matmul(F_candidates, E ) / self.L



            # -------------------------------消融--------------------------------------------------------------------
            
            # ====== 新增：基于等效积分的动态自适应 Delta ======
            # 初始最高强度为1.3，每生成一个token衰减2%
            # delta_0 = 1.2 
            # gamma = 0.99
            # # 设定一个保底强度，确保即使是极长文本(如1000词)，水印也不会完全中断
            # min_delta = 0.1
            # if valid_steps<30:
            #     current_delta=0.9
            # # valid_steps 是当前已经生成的有水印的 token 数量
            # elif 30<=valid_steps<=100:
            #     current_delta = delta_0 * (gamma ** valid_steps)
            # elif 100<valid_steps<180:
            #     current_delta=0.2
            # else:
            #     current_delta = delta_0 * (gamma ** valid_steps)
            #     current_delta = max(current_delta, min_delta)
            #---------------------------------------------------------------
            delta_0=0.8
            gamma=0.99
            min_delta=0.1
            if valid_steps<40:
                current_delta=delta_0
            else:
                current_delta=delta_0*(gamma**(valid_steps-40))
                current_delta = max(current_delta, min_delta)


                
            base_seed = int(c_seeds[i])
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
                
                # 3. 只计算这 K 个词的调整分数
                score_k = torch.matmul(F_candidates_k, E) / self.L
                
                # 4. 直接把算好的分数加到原 Logits 对应的 Top-K 位置上
                new_scores[i][top_k_indices] += current_delta * score_k

            else:
                # 如果 top_k 设置为 0，意味着必须处理全词表（退回到老路子）
                rng = np.random.default_rng(base_seed)
                F_np = rng.integers(0, 2, size=(self.vocab_size, self.L)) * 2 - 1
                F_candidates = torch.tensor(F_np, device=self.device, dtype=torch.float32)
                score = torch.matmul(F_candidates, E) / self.L
                new_scores[i] += current_delta * score

            # delta_0=1.2
            # gamma=0.99
            # min_delta=0.1
            # current_delta=delta_0*(gamma**(valid_steps))
            # current_delta = max(current_delta, min_delta)
                

            
            
            # 使用 current_delta 代替原有的 self.delta
            #new_scores[i] += current_delta * score
            # new_scores[i] += self.delta * score
            
            
            #**************************只修改 Top-K 的 Logits**********************************************************
            # ====== 核心修改：只修改 Top-K 的 Logits ======
            # if self.top_k > 0 and self.top_k < self.vocab_size:
            #     # 获取当前样本原始打分前 k 高的索引
            #     _, top_k_indices = torch.topk(scores[i], self.top_k)
                
            #     # 构造一个与词表等长的布尔掩码（Mask）
            #     mask = torch.zeros_like(scores[i], dtype=torch.bool)
            #     mask[top_k_indices] = True
                
            #     # 利用掩码，只给前 k 个词汇加上 watermark 偏移
            #     new_scores[i][mask] += current_delta * score[mask]
            # else:
            #     new_scores[i] += current_delta * score

            # ====== 核心修改：只修改 Top-K 的 Logits ======


            
            # if self.top_k > 0 and self.top_k < self.vocab_size:
            #     # 获取当前样本原始打分前 k 高的索引
            #     _, top_k_indices = torch.topk(scores[i], self.top_k)
                
            #     # 构造一个与词表等长的布尔掩码（Mask）
            #     mask = torch.zeros_like(scores[i], dtype=torch.bool)
            #     mask[top_k_indices] = True
                
            #     # 利用掩码，只给前 k 个词汇加上 watermark 偏移
            #     new_scores[i][mask] += self.delta * score[mask]
            # else:
            #     new_scores[i] += self.delta * score





            
            # -------------------------------------------------------------------------------
            # current_logits = scores[i]
            
            # # 策略 A：合理性掩码 (Top-K Filter)
            # # 强制水印只能在原分布的前 K 个合理候选词中起作用
            # K = 50 
            # top_k_values, _ = torch.topk(current_logits, K)
            # threshold = top_k_values[-1]
            # plausible_mask = (current_logits >= threshold).float()
            
            # # 策略 B：动态香农熵缩放 (Entropy Scaling)
            # # 算出当前分布的概率和熵
            # probs = torch.softmax(current_logits, dim=-1)
            # entropy = -torch.sum(probs * torch.log(probs + 1e-8))
            
            # # 经验映射：熵越低（模型越确信，比如专有名词），允许的 delta 越小
            # # 熵越高（模型在多个近义词中犹豫），发挥全部 delta 强度
            # # 分母 2.0 是一个基准熵值，可根据 Llama-3 的平均表现微调
            # entropy_scale = torch.clamp(entropy / 2.0, 0.0, 1.0)
            
            # # 最终施加偏置：
            # # 1. 只有 plausible_mask 为 1 的词才会被加上水印偏置
            # # 2. 整体强度受模型当前的确定性(entropy_scale)严格控制
            # new_scores[i] += self.delta * entropy_scale * (score * plausible_mask)
            # -------------------------------------------------------------------------------




            

            # -----------------------------------------------------------------------
            # step = valid_steps + 1
            
            # if step <= 50:
            #     current_delta = 0.8
            # else:
            #     # 使用 1.5 次方反比衰减函数，积分结果完美吻合 200步平均值为0.4
            #     current_delta = 0.8 * ((50.0 / step) ** 5.0)
            
            # # 限制最小衰减值为 0.1
            # current_delta = max(0.05, current_delta)

            # # 应用当前的动态 delta
            # new_scores[i] += current_delta * score
            # -----------------------------------------------------------------------
            
            
           # new_scores[i] += self.delta * score

        

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
        plt.title("Convergence of Mean Error Norm (100 Samples)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_name}.png", dpi=300)
        print(f"✅ 误差数据已保存为 {save_name}.csv 和 {save_name}.png")
        plt.close() # 释放内存

# class WatermarkOmniMark(LogitsProcessor):
#     def __init__(self, tokenizer, device, vocab_size, c_key=530773, delta=2.5, window_size=2, bits='0'*16,gamma=0.95):
#         self.tokenizer = tokenizer
#         self.device = device
#         self.vocab_size = vocab_size
#         self.c_key = c_key
#         self.delta = delta
#         self.window_size = window_size
#         self.bits = bits
#         self.gamma = gamma # 存储衰减系数
#         self.L = len(bits)
#         print(self.L)
#         print(self.L)
#         # 将 "0101" 映射为 [-1, 1, -1, 1] 的目标方向向量 M
#         m_list = [1 if b == '1' else -1 for b in self.bits]
#         self.M = torch.tensor(m_list, device=device, dtype=torch.float32)
        
#         self.prompt_len = None
#         self.history_cache = {}  # 记录每个 batch 的 (生成的长度, 有效步数, 累加向量S)
        
#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
#         batch_size = input_ids.shape[0]
#         current_len = input_ids.shape[1]
        
#         # 记录初始 prompt 的长度，作为生成的起点
#         if self.prompt_len is None:
#             self.prompt_len = current_len
            
#         new_scores = scores.clone()
#         prefix = input_ids[:, -self.window_size:]
#         c_seeds = prf(prefix, self.c_key)
        
#         for i in range(batch_size):
#             input_seq = input_ids[i]
            
#             # 1. 状态恢复与更新
#             if i not in self.history_cache:
#                 S = torch.zeros(self.L, device=self.device)
#                 effective_steps = 0  # 将 valid_steps 改名为 effective_steps 以防混淆
#                 last_len = self.prompt_len
#             else:
#                 last_len, effective_steps, S = self.history_cache[i]
            
#             for t in range(last_len, current_len):
#                 if t >= self.prompt_len + self.window_size: 
#                     prev_prefix = input_seq[t - self.window_size : t]
#                     token_sampled = input_seq[t].item()
                    
#                     seed = prf(prev_prefix.unsqueeze(0), self.c_key)[0]
#                     rng = np.random.default_rng(seed)
#                     F_np = rng.integers(0, 2, size=(self.vocab_size, self.L)) * 2 - 1
#                     F_sampled = torch.tensor(F_np[token_sampled], device=self.device, dtype=torch.float32)
                    
#                     # ====== 核心修改：积分泄漏 (Leaky Integration) ======
#                     S = S * self.gamma + F_sampled
#                     effective_steps = effective_steps + 1
            
#             # 将当前状态写回缓存
#             self.history_cache[i] = (current_len, effective_steps, S)
            
#             if current_len < self.prompt_len + self.window_size:
#                 continue
            
#             # ====== 2. 闭环负反馈核心 ======
#             # 使用 effective_steps 替代 valid_steps 计算目标
#             target_S = effective_steps * 0.4 * self.M 
#             E = target_S - S
            
#             # (可选) 积分限幅 Anti-Windup：如果希望更加保守，可以直接对 E 截断
#             # max_E_val = self.L * 1.5
#             # E = torch.clamp(E, min=-max_E_val, max=max_E_val)

#             # 3. 对词表进行打分
#             rng = np.random.default_rng(c_seeds[i])
#             F_np = rng.integers(0, 2, size=(self.vocab_size, self.L)) * 2 - 1
#             F_candidates = torch.tensor(F_np, device=self.device, dtype=torch.float32)
            
#             score = torch.matmul(F_candidates, E) / self.L
            
#             new_scores[i] += self.delta * score
#         return new_scores