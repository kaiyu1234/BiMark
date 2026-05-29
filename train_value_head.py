import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import numpy as np
from tqdm import tqdm

from value_head import WatermarkValueHead
from utils import prf  # 从你现有的 utils.py 导入

# ==========================================
# 1. 动态复原误差序列的函数 (核心逻辑)
# ==========================================
def compute_E_t_sequence(input_ids, prompt_len, c_key, window_size, bits, vocab_size, device):
    """
    通过确定性的文本和 PRF，还原生成每一步时的误差向量 E_t
    """
    L = len(bits)
    m_list = [1 if b == '1' else -1 for b in bits]
    M = torch.tensor(m_list, device=device, dtype=torch.float32)
    
    S = torch.zeros(L, device=device)
    valid_steps = 0
    seq_len = input_ids.shape[0]
    
    E_t_seq = []
    
    # 模拟文本生成的过程
    for t in range(seq_len):
        # 1. 计算当前的 E (这是准备用来预测下一个 Token t+1 的状态)
        if t < prompt_len + window_size - 1:
            E = torch.zeros(L, device=device)
        else:
            target_S = (valid_steps + 1) * 0.5 * M
            E = target_S - S
        E_t_seq.append(E.clone())
        
        # 2. 推进状态 S (假设模型实际生成了 token input_ids[t])
        if t >= prompt_len + window_size - 1:
            prev_prefix = input_ids[t - window_size + 1 : t + 1]
            token_sampled = input_ids[t].item()
            
            # 这里必须跟你 WatermarkOmniMark.py 里的 PRF 逻辑一模一样！
            seed = prf(prev_prefix.unsqueeze(0), c_key)[0]
            rng = np.random.default_rng(seed)
            F_np = rng.integers(0, 2, size=(vocab_size, L)) * 2 - 1
            F_sampled = torch.tensor(F_np[token_sampled], device=device, dtype=torch.float32)
            S += F_sampled
            valid_steps += 1
            
    return torch.stack(E_t_seq) # Shape: (seq_len, 16)


# ==========================================
# 2. 主训练循环
# ==========================================
# def main():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     # 1. 加载主模型 (极其重要：使用 bfloat16 并且不需要梯度，省显存！)
#     model_name = "meta-llama/Meta-Llama-3.1-8B" # 替换成你的模型路径
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     print("Loading frozen LLM...")
#     llm = AutoModelForCausalLM.from_pretrained(
#         model_name, 
#         torch_dtype=torch.bfloat16,
#         device_map="auto"
#     )
#     llm.eval() # 冻结 LLM

#     # 2. 初始化你的 Value Head (需要梯度)
#     print("Initializing Value Head...")
#     # 修改后 (假设我们定最大长度为 64)
#     value_head = WatermarkValueHead(hidden_dim=4096, max_error_dim=64, vocab_size=128256).to(device)
#     optimizer = AdamW(value_head.parameters(), lr=1e-4)
#     value_head.train()

#     # 3. 加载带 Reward 的文本数据
#     data_path = "training_data.jsonl"
#     with open(data_path, "r", encoding="utf-8") as f:
#         dataset = [json.loads(line) for line in f.readlines()]

#     epochs = 3
#     for epoch in range(epochs):
#         total_loss = 0
#         pbar = tqdm(dataset, desc=f"Epoch {epoch+1}/{epochs}")
        
#         for item in pbar:
#             full_text = item["prompt_text"] + item["generated_text"]
#             prompt_len = item["prompt_len"]
#             reward_val = item["reward"] # 这条轨迹的最终全局评分
            
#             # 转为 Tensor
#             input_ids = tokenizer.encode(full_text, return_tensors="pt")[0].to(device)
#             seq_len = input_ids.shape[0]
            
#             if seq_len <= prompt_len: continue # 没生成内容跳过

#             # ========================================================
#             # 魔法发生的地方：一键提取全部隐藏状态 (h) 和误差状态 (E)
#             # ========================================================
#             with torch.no_grad():
#                 # 要求 LLM 输出隐藏层状态
#                 outputs = llm(input_ids.unsqueeze(0), output_hidden_states=True)
#                 # 取最后一层的隐藏状态, 去掉 batch 维度 -> Shape: (seq_len, 4096)
#                 h_all = outputs.hidden_states[-1].squeeze(0).to(torch.float32) 
            
#             # 动态还原所有步骤的 E_t -> Shape: (seq_len, 16)
#             E_all = compute_E_t_sequence(
#                 input_ids=input_ids, 
#                 prompt_len=prompt_len, 
#                 c_key=530773, 
#                 window_size=2, 
#                 bits='0'*16, 
#                 vocab_size=len(tokenizer), 
#                 device=device
#             )

#             # ========================================================
#             # 截取“生成部分”进行训练 (序列对其)
#             # LLM 中，t 时刻的隐藏状态 h_t 用来预测 t+1 时刻的 Token。
#             # ========================================================
#             # 输入状态：预测从 prompt_len 到最后生成的一个 token
#             h_train = h_all[prompt_len - 1 : seq_len - 1] 
#             E_train = E_all[prompt_len - 1 : seq_len - 1]
            
#             # 实际发生的动作 (真实生成的 Tokens)
#             actions = input_ids[prompt_len : seq_len] 
            
#             # 我们的目标是：让网络拟合“这个句子带来的全局 Reward”
#             # (也可以加入 discount factor 进行 step 级别的衰减，这里以全局相同 reward 举例)
#             rewards = torch.full_like(actions, fill_value=reward_val, dtype=torch.float32)

#             # ========================================================
#             # 更新 Value Head
#             # ========================================================
#             optimizer.zero_grad()
            
#             # 一次性算出该句子生成过程中，每一步所有 128256 个词的价值预测
#             predicted_logits = value_head(h_train, E_train) # Shape: (gen_len, 128256)
            
#             # 提取出当时【实际选中】的那些 Token 的预测价值
#             action_values = predicted_logits.gather(1, actions.unsqueeze(1)).squeeze(1) # Shape: (gen_len,)
            
#             # 算 MSE Loss: 预测价值 vs 真实结局
#             loss = F.mse_loss(action_values, rewards)
#             loss.backward()
#             optimizer.step()
            
#             total_loss += loss.item()
#             pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

#         print(f"Epoch {epoch+1} Average Loss: {total_loss / len(dataset):.4f}")

#     # 保存训练成果
#     torch.save(value_head.state_dict(), "foresight_value_head.pt")
#     print("Value Head training complete! Saved to foresight_value_head.pt")

# if __name__ == "__main__":
#     main()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_name = "meta-llama/Meta-Llama-3.1-8B" # 替换成你的模型路径
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # 必须设置 pad_token，因为我们要打包多个不同长度的句子
    tokenizer.pad_token = tokenizer.eos_token 
    
    print("Loading frozen LLM...")
    llm = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    llm.eval() # 冻结主模型

    # 根据我们之前的策略，设置 max_error_dim=64
    print("Initializing Value Head...")
    value_head = WatermarkValueHead(hidden_dim=4096, max_error_dim=64, vocab_size=len(tokenizer)).to(device)
    optimizer = AdamW(value_head.parameters(), lr=1e-4)
    value_head.train()

    data_path = "training_data.jsonl"
    with open(data_path, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f.readlines()]

    epochs = 3
    # 💡 核心优化：设置 Batch Size，根据你的显存调节（24G 显存可以设为 8 或 16）
    batch_size = 16 

    for epoch in range(epochs):
        total_loss = 0
        # 改用按 batch 步进的进度条
        pbar = tqdm(range(0, len(dataset), batch_size), desc=f"Epoch {epoch+1}/{epochs}")
        
        for i in pbar:
            batch_data = dataset[i : i + batch_size]
            
            # 1. 把一个批次的文本打包，自动 Padding 对齐长度
            texts = [item["prompt_text"] + item["generated_text"] for item in batch_data]
            inputs = tokenizer(texts, padding=True, return_tensors="pt").to(device)
            
            # ========================================================
            # 魔法发生的地方：一次性算出 16 句话的隐藏状态！(提速 10 倍以上)
            # ========================================================
            with torch.no_grad():
                outputs = llm(inputs.input_ids, attention_mask=inputs.attention_mask, output_hidden_states=True)
                # h_batch shape: (batch_size, max_seq_len, 4096)
                h_batch = outputs.hidden_states[-1] 
            
            # 准备收集这 16 句话中“有效的训练片段” (去除 Padding 和 Prompt 部分)
            batch_h_train = []
            batch_E_train = []
            batch_actions = []
            batch_rewards = []
            
            # 2. 遍历批次中的每一句话，剔除 Padding，还原 E_t
            for j, item in enumerate(batch_data):
                prompt_len = item["prompt_len"]
                reward_val = item["reward"]
                
                # 利用 attention_mask 提取真实的序列 (去 Padding)
                valid_mask = inputs.attention_mask[j] == 1
                input_ids_j = inputs.input_ids[j][valid_mask]
                seq_len = input_ids_j.shape[0]
                
                if seq_len <= prompt_len: continue # 没生成内容跳过
                
                # 取出该句子的隐状态并转为 float32
                h_all = h_batch[j][:seq_len].to(torch.float32)
                
                # 计算这条轨迹的误差序列 (由于计算很快，可以在 CPU/GPU 单独算)
                E_all = compute_E_t_sequence(
                    input_ids=input_ids_j, 
                    prompt_len=prompt_len, 
                    c_key=530773, 
                    window_size=2, 
                    bits='0'*16, # 注意：如果你之前用了混合 bits，这里应从 item 里读取
                    vocab_size=len(tokenizer), 
                    device=device
                )
                
                # 截取训练部分
                batch_h_train.append(h_all[prompt_len - 1 : seq_len - 1])
                batch_E_train.append(E_all[prompt_len - 1 : seq_len - 1])
                batch_actions.append(input_ids_j[prompt_len : seq_len])
                
                # 填充该句子的 Reward
                batch_rewards.append(torch.full_like(batch_actions[-1], fill_value=reward_val, dtype=torch.float32))

            # 防止遇到全是无效数据的极端情况
            if len(batch_h_train) == 0:
                continue

            # ========================================================
            # 3. 把这个批次里所有的 Token 拉平，做一次集中的梯度更新
            # ========================================================
            # 例如 16 句话，每句生成 200 词，这里拼接后相当于一次性训练 3200 个样本对！
            h_train_flat = torch.cat(batch_h_train, dim=0)
            E_train_flat = torch.cat(batch_E_train, dim=0)
            actions_flat = torch.cat(batch_actions, dim=0)
            rewards_flat = torch.cat(batch_rewards, dim=0)

            optimizer.zero_grad()
            
            # 前向传播：一次性给 3200 个 Token 算预测价值 (极其高效)
            predicted_logits = value_head(h_train_flat, E_train_flat) 
            
            # 取出实际动作的得分并算 MSE Loss
            action_values = predicted_logits.gather(1, actions_flat.unsqueeze(1)).squeeze(1)
            loss = F.mse_loss(action_values, rewards_flat)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        print(f"Epoch {epoch+1} Average Loss: {total_loss / (len(dataset) // batch_size):.4f}")

    # 保存训练成果
    torch.save(value_head.state_dict(), "foresight_value_head.pt")
    print("Value Head training complete! Saved to foresight_value_head.pt")

if __name__ == "__main__":
    main()