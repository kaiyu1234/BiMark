import numpy as np
import random
from math import sqrt
from utils import prf
from scipy import stats
import copy
import torch
class WatermarkDetector:
    def __init__(self, tokenizer, vocab_size, window_size, gamma):
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.gamma = gamma

    def _compute_z_score(self, observed_green_count, total_count, proportion=False):
        if not proportion:
            proportion = self.gamma
        numer = observed_green_count - proportion * total_count
        denom = sqrt(total_count * proportion * (1 - proportion))
        z = numer / denom
        return z
    
    def _compute_p_value(self, z):
        p_value = stats.norm.sf(z)
        return p_value

    def _z_test(self, COUNT):
        if len(COUNT[0]) == 1:
            observed_green_count = np.max(COUNT, axis=1)
        elif len(COUNT[0]) == 2:
            observed_green_count = np.min(np.max(COUNT, axis=1))
        total_count = np.sum(COUNT)
        score = self._compute_z_score(observed_green_count, total_count)
        p_value = self._compute_p_value(score)
        return score, p_value, observed_green_count, total_count
    
    def _zerobit_watermark_detector_stride(self, green_count, valid_tokens, stride):
        z_score_list, z_p_value_list = [], []
        stride_list = []
        for i in range(green_count.shape[0]):
            z_score = self._compute_z_score(green_count[i], valid_tokens[i])
            z_p_value = self._compute_p_value(z_score)
            z_score_list.append(z_score)
            z_p_value_list.append(z_p_value)
            stride_list.append(stride * i)
        print('z_score_list:', z_score_list)
        return z_score_list, z_p_value_list, stride_list


    def decode_stealthink_multibit(self, inputs, c_key, bits, R=0.25, window_size=2, start=0, stride=50):
        import torch
        import math
        import random
        
        chunk_capacity = int(math.log2(1/R))
        num_value = int(1/R) 
        
        if len(bits) % chunk_capacity != 0:
            bits += '0' * (chunk_capacity - len(bits) % chunk_capacity)
        true_msg_ints = [int(bits[i:i+chunk_capacity], 2) for i in range(0, len(bits), chunk_capacity)]
        converted_msg_length = len(true_msg_ints)
        
        cl_total = {i: [0 for _ in range(num_value)] for i in range(converted_msg_length)}
        valid_tokens = 0
        hist = set()

        for t in range(window_size, len(inputs)):
            prefix = inputs[t - window_size: t]
            pref_tuple = tuple(prefix.tolist())
            if pref_tuple in hist:
                continue
            hist.add(pref_tuple)
            
            c_seed = prf(prefix, c_key)
            if isinstance(c_seed, list): c_seed = c_seed[0]
                
            rng_bp = torch.Generator(device='cpu')
            rng_bp.manual_seed(int(c_seed) % (2**64 - 1))
            bit_pos = torch.randint(low=0, high=converted_msg_length, size=(1,), generator=rng_bp).item()
            
            rng_vocab = torch.Generator(device='cpu')
            rng_vocab.manual_seed(int(c_seed) % (2**64 - 1))
            vocab_perm = torch.randperm(self.vocab_size, device='cpu', generator=rng_vocab)
            colorlist = torch.chunk(vocab_perm, num_value)
            
            token_idx = inputs[t].item()
            for guessed_info in range(num_value):
                if token_idx in colorlist[guessed_info]:
                    cl_total[bit_pos][guessed_info] += 1
                    valid_tokens += 1
                    break
                    
        decode_bits = ''
        hit_bits = 0
        total_bits = converted_msg_length * chunk_capacity
        
        for i in range(converted_msg_length):
            # 核心修复：StealthInk 找最小值(Red list)。若遇到票数相同的平局(Tie)，随机选择避免偏置
            min_count = min(cl_total[i])
            candidates = [idx for idx, count in enumerate(cl_total[i]) if count == min_count]
            pred_val = random.choice(candidates) 
            
            pred_bin = bin(pred_val)[2:].zfill(chunk_capacity)
            true_bin = bin(true_msg_ints[i])[2:].zfill(chunk_capacity)
            
            decode_bits += pred_bin
            hit_bits += sum(1 for p, t in zip(pred_bin, true_bin) if p == t)
                
        hit_rate = hit_bits / total_bits if total_bits > 0 else 0
        print(f"StealthInk Decoding (R={R}) -> Extracted: {decode_bits}, Target: {bits}, Hit Rate: {hit_rate:.2f}")
        return cl_total, decode_bits, hit_bits, hit_rate, valid_tokens

    def decode_watermod_multibit(self, prompt_tokens, gen_tokens, c_key, bits, model, k=4, window_size=1, start=0, stride=50):
        import torch
        from scipy import stats
        import numpy as np
        
        log2k = int(np.log2(k))
        pad_len = (log2k - len(bits) % log2k) % log2k
        padded_bits = bits + '0' * pad_len
        m_true = []
        for i in range(0, len(padded_bits), log2k):
            chunk = padded_bits[i:i+log2k]
            m_true.append(int(chunk, 2))
        b_tilde = len(m_true)
        
        C = np.zeros((b_tilde, k))
        valid_tokens = 0
        G = 0 # 命中次数
        
        full_tokens = torch.cat([prompt_tokens, gen_tokens])
        prompt_len = len(prompt_tokens)
        
        # 一次性获取所有 tokens 的原始 logits
        with torch.no_grad():
            outputs = model(full_tokens.unsqueeze(0))
            logits = outputs.logits[0] # shape: [seq_len, vocab_size]
            
        # 【关键修复】从 start 处开始检测，并彻底移除 hist 去重
        for t in range(start, len(gen_tokens)):
            full_t = prompt_len + t
            if full_t < window_size:
                continue
                
            prefix = full_tokens[full_t - window_size : full_t]
            
            # 获取预测当前 token 所对应的 logit
            l_t = logits[full_t - 1]
            
            c_seed = prf(prefix, c_key)
            # 【关键修复】确保检测端的 seed 与生成端绝对一致
            if isinstance(c_seed, torch.Tensor):
                c_seed = int(c_seed.item())
            elif isinstance(c_seed, list):
                c_seed = int(c_seed[0])
            else:
                c_seed = int(c_seed)
                
            rng = np.random.default_rng(c_seed)
            u = rng.random()
            p = min(int(u * b_tilde), b_tilde - 1)
            
            # 排序获取当前生成词的 Rank
            sorted_indices = torch.argsort(l_t, descending=True)
            token_idx = full_tokens[full_t].item()
            
            rank_tensor = torch.where(sorted_indices == token_idx)[0]
            if len(rank_tensor) == 0:
                continue
            rank = rank_tensor.item()
            
            d = rank % k
            C[p][d] += 1
            valid_tokens += 1
            
            if d == m_true[p]:
                G += 1
                
        # 多数投票解码
        decode_bits = ""
        for p in range(b_tilde):
            if np.sum(C[p]) == 0:
                d_hat = 0 # 避免该位置没有任何有效 token 时的报错
            else:
                d_hat = np.argmax(C[p])
            bin_str = bin(d_hat)[2:].zfill(log2k)
            decode_bits += bin_str
            
        decode_bits = decode_bits[:len(bits)]
        
        # 计算命中率
        hit_bits = sum(1 for i, b in enumerate(bits) if i < len(decode_bits) and decode_bits[i] == b)
        hit_rate = hit_bits / len(bits) if len(bits) > 0 else 0
        
        # Z-score 计算
        p_0 = 1.0 / k
        if valid_tokens > 0:
            z_score = (G - valid_tokens * p_0) / np.sqrt(valid_tokens * p_0 * (1 - p_0))
            p_value = stats.norm.sf(z_score)
        else:
            z_score = 0.0
            p_value = 1.0
            
        return C.tolist(), decode_bits, hit_bits, hit_rate, valid_tokens, z_score, p_value
    
    def decode_mpac_multibit(self, inputs, c_key, bits, gamma=0.25):
        """
        MPAC 多比特盲检测器
        """
        import torch
        import math
        
        r = int(1 / gamma)
        bits_per_symbol = int(math.log2(r))
        effective_b = len(bits) // bits_per_symbol
        
        # 统计矩阵 W[p][m]
        W = [[0 for _ in range(r)] for _ in range(effective_b)]
        hist = set()
        valid_tokens = 0
        
        for t in range(self.window_size, len(inputs)):
            prefix = inputs[t - self.window_size: t]
            pref_tuple = tuple(prefix.tolist())
            
            if pref_tuple in hist:
                continue
            hist.add(pref_tuple)
            
            c_seed_list = prf(prefix.unsqueeze(0), c_key)
            c_seed = c_seed_list[0] if isinstance(c_seed_list, list) else c_seed_list
            
            rng = torch.Generator(device='cpu')
            rng.manual_seed(c_seed % (2**64 - 1))
            
            # 重建分配位置和词表切分，顺序必须与编码时严格一致
            p = torch.randint(low=0, high=effective_b, size=(1,), generator=rng).item()
            vocab_perm = torch.randperm(self.vocab_size, device='cpu', generator=rng)
            colorlists = torch.chunk(vocab_perm, r)
            
            token_idx = inputs[t].item()
            
            # 寻找当前 token 属于哪个 colorlist
            for m_idx in range(r):
                if token_idx in colorlists[m_idx]:
                    W[p][m_idx] += 1
                    break
                    
            valid_tokens += 1

        # 还原比特信息并计算匹配度
        decode_bits = ''
        hit = 0
        z_scores_symbols = []
        
        for p in range(effective_b):
            # 取票数最多的 colorlist 作为该位置的符号预测
            predicted_symbol = max(range(r), key=lambda x: W[p][x])
            
            # 将 symbol 转回二进制字符串并补齐
            symbol_bits = bin(predicted_symbol)[2:].zfill(bits_per_symbol)
            decode_bits += symbol_bits
            
            # 统计 Z-score (以 1/r 作为均值期望，评估分布是否显著偏向某一色表)
            total_votes_p = sum(W[p])
            max_votes = W[p][predicted_symbol]
            z = self._compute_z_score(max_votes, total_votes_p, proportion=gamma) if total_votes_p > 0 else 0
            z_scores_symbols.append(z)
            
        # 计算 Bit 准确率 (Hit Rate)
        for i in range(len(bits)):
            if decode_bits[i] == bits[i]:
                hit += 1
                
        hit_rate = hit / len(bits) if len(bits) > 0 else 0
        
        print(f"MPAC Decoding -> Extracted Bits: {decode_bits}, Hit Rate: {hit_rate:.2f}")
        return W, decode_bits, hit, hit_rate, valid_tokens, z_scores_symbols


    def decode_xmark_multibit(self, inputs, bits, hash_key=15485863, window_size=2):
        """
        XMark 的解码器。
        严格按照 XMark 的两级 hash block 计算逻辑还原。
        """
        bits_len = len(bits)
        num_blocks = bits_len // 2
        counts = torch.zeros((num_blocks, 4), dtype=torch.long)
        
        valid_tokens = 0
        hash_key_2 = 12345
        
        # 遍历生成的 token
        for t in range(window_size, len(inputs)):
            prev_prev = int(inputs[t-2].item())
            prev      = int(inputs[t-1].item())
            curr      = int(inputs[t].item())

            bidx = (prev + prev_prev) % num_blocks

            # 还原 rng_1
            seed_1 = (hash_key * prev * prev_prev) % (2**64)
            rng_1 = torch.Generator(device='cpu').manual_seed(seed_1)
            vocab_perm_1 = torch.randperm(self.vocab_size, generator=rng_1, device='cpu')
            parts_1 = torch.chunk(vocab_perm_1, chunks=4, dim=0)
            
            # 还原 rng_2
            seed_2 = (hash_key_2 * prev * prev_prev) % (2**64)
            rng_2 = torch.Generator(device='cpu').manual_seed(seed_2)
            vocab_perm_2 = torch.randperm(self.vocab_size, generator=rng_2, device='cpu')
            parts_2 = torch.chunk(vocab_perm_2, chunks=4, dim=0)

            # 统计词频
            for q, part in enumerate(parts_1):
                if curr in part:
                    counts[bidx, q] += 1
                    break
            
            for q, part in enumerate(parts_2):
                if curr in part:
                    # cTMM 逻辑
                    if curr not in parts_1[q]:  
                        counts[bidx, q] += 1
                        break
                        
            valid_tokens += 1
            
        # 解码过程 (取数量最少的 index)
        block_indices = torch.argmin(counts, dim=1).tolist()  # 0..3
        bits_per_block = [format(idx, "02b") for idx in block_indices]
        decode_bits = "".join(bits_per_block)

        # 计算 hit rate
        hit = sum([1 for i in range(bits_len) if decode_bits[i] == bits[i]])
        hit_rate = hit / bits_len if bits_len > 0 else 0
        
        print(f"XMark Decoding -> Extracted Bits: {decode_bits}, Hit Rate: {hit_rate:.2f}")

        # XMark 不使用 Z-score 检测架构，为了兼容原有 dataframe，其它用 0 填充
        return counts.tolist(), decode_bits, hit, hit_rate, valid_tokens

    
    
    def decode_omnimark_multibit(self, inputs, c_key, bits, window_size=2, start=0, stride=50):
        """
        全息多比特水印（OmniMark）的解码器。
        利用所有生成的 token 对所有 bit 的联合累积，在短文本下达到极高的检出率。
        """
        bits_len = len(bits)
        # 初始化一个 L 维的积分向量，收集所有 token 投出的票
        V_decode = np.zeros(bits_len)
        hist = set()
        valid_tokens = 0

        for t in range(window_size, len(inputs)):
            prefix = inputs[t - window_size: t]
            pref_tuple = tuple(prefix.tolist())
            
            if pref_tuple in hist:
                continue
            hist.add(pref_tuple)

            # 生成与生成端完全一致的伪随机种子
            c_seed = prf(prefix, c_key)
            if isinstance(c_seed, list):
                c_seed = c_seed[0] # 处理 prf 返回 list 的情况
            
            token_idx = inputs[t].item()
            token_seed = (int(c_seed) + int(token_idx) * 2654435761) % (2**32)
            token_rng = np.random.default_rng(token_seed)
            F_v = token_rng.integers(0, 2, size=bits_len) * 2 - 1
                        
            V_decode += F_v
            valid_tokens += 1

        # for t in range(window_size, len(inputs)):
        #     prefix = inputs[t - window_size: t]
        #     pref_tuple = tuple(prefix.tolist())
            
        #     if pref_tuple in hist:
        #         continue
        #     hist.add(pref_tuple)

        #     # 生成与生成端完全一致的伪随机种子
        #     c_seed = prf(prefix, c_key)
        #     if isinstance(c_seed, list):
        #         c_seed = c_seed[0] # 处理 prf 返回 list 的情况
            
        #     rng = np.random.default_rng(c_seed)
        #     # 重建当前上下文的整个词表的 L 维指纹
        #     F_np = rng.integers(0, 2, size=(self.vocab_size, bits_len)) * 2 - 1
            
        #     # 获取语言模型实际生成的词
        #     token_idx = inputs[t].item()
        #     # 提取该生成词所附带的 L 维特征向量，并叠加到全局积分器上
        #     F_v = F_np[token_idx]
            
        #     V_decode += F_v
        #     valid_tokens += 1
            

        # 1. 盲解码提取 (Blind Decoding)
        decode_bits = ''
        hit = 0
        for i in range(bits_len):
            if V_decode[i] > 0:
                decode_bits += '1'
                if bits[i] == '1': hit += 1
            elif V_decode[i] < 0:
                decode_bits += '0'
                if bits[i] == '0': hit += 1
            else:
                decode_bits += 'x'
                
        # 2. 统计显著性计算 (Z-scores)
        # 在原假设下，每一维度都是 N 个均匀的 {-1, 1} 的随机游走累加，方差为 N
        if valid_tokens > 0:
            z_scores_bits = V_decode / np.sqrt(valid_tokens)
            # p-value 计算 (双侧检验)
            p_values_bits = [stats.norm.sf(abs(z)) * 2 for z in z_scores_bits]
            
            # 这是一个可用于论文大吹特吹的“联合校验分数 (Global Z-score)”
            # 它衡量了整段文本与特定 message 的全局对齐程度，其方差扩展为 N * L
            M_array = np.array([1 if b == '1' else -1 for b in bits])
            S_total = np.sum(M_array * V_decode)
            global_z_score = S_total / np.sqrt(valid_tokens * bits_len)
            global_p_value = stats.norm.sf(global_z_score)
        else:
            z_scores_bits = np.zeros(bits_len)
            p_values_bits = [1.0] * bits_len
            global_z_score = 0.0
            global_p_value = 1.0

        hit_rate = hit / bits_len if bits_len > 0 else 0

        # 为了兼容你原本的接口输出格式，封装返回值
        print(f"OmniMark Decoding -> Extracted Bits: {decode_bits}, Hit Rate: {hit_rate:.2f}")
        print(f"Global Z-Score: {global_z_score:.2f}, Global P-Value: {global_p_value:.2e}")
        
        return V_decode.tolist(), decode_bits, hit, hit_rate, valid_tokens, z_scores_bits.tolist(), p_values_bits, global_z_score, global_p_value
    
    def decode_bimark_multibit_watermark(self, inputs, partition_key, d, c_key,  bit_idx_key, bits, bits_len=0, weight=0,
                               start=0, stride=50):
        if bits_len == 0:
            bits_len = len(bits)

        if weight == 0:
            weight = [1 for _ in range(d)]

        stride_idx_list = [start + stride * i for i in range((len(inputs) - start)//stride +1)]       
    
        COUNTS = [[[0, 0] for _ in range(bits_len)] for _ in range(len(stride_idx_list) )]
    
        hist = set() 
        generate_counts = [0 for _ in range(len(stride_idx_list))]
        idx_s = 0
        for t in range(self.window_size, len(inputs)):
            try:
                generate_counts[idx_s] += 1
            except:
                continue
            if idx_s < len(stride_idx_list):
                if (t-self.window_size) == stride_idx_list[idx_s]:
                    idx_s += 1
                    try:
                        COUNTS[idx_s] = copy.deepcopy(COUNTS[idx_s-1])
                        generate_counts[idx_s] = generate_counts[idx_s-1]
                    except:
                        continue
            prefix = inputs[t - self.window_size: t]
            
            p_seed=prf(prefix, partition_key)
            c_seed=prf(prefix, c_key) # seed
            rng_idx_seed = prf(prefix, bit_idx_key)
            
            if prefix not in hist:  # do not watermarking the same seed
                hist.add(prefix)
            else:
                continue
            
            rng_p = np.random.default_rng(p_seed)
            partition_masks = []
            for j in range(d):
                num_V0 = int(self.vocab_size * 0.5)
                mask = np.zeros(self.vocab_size, dtype=bool)
                mask[rng_p.choice(self.vocab_size, num_V0, replace=False)] = True
                partition_masks.append(mask)

            rng_c = np.random.default_rng(c_seed)
            
            rng_bit_idx = np.random.default_rng(rng_idx_seed)

            c_list = rng_c.integers(0, 2, size=len(partition_masks))

            bit_idx = rng_bit_idx.integers(0, bits_len)

            token_idx = inputs[t].item()

            for i in range(len(partition_masks)):
                mask = partition_masks[i]
                if ((c_list[i] == 1 and (mask[token_idx].item() is False)) or (c_list[i] == 0 and (mask[token_idx].item() is True))):
                    COUNTS[idx_s][bit_idx][1] += 1 * weight[i]
                elif ((c_list[i] == 1 and (mask[token_idx].item() is True)) or (c_list[i] == 0 and (mask[token_idx].item() is False))):
                    COUNTS[idx_s][bit_idx][0] += 1 * weight[i]
                else:
                    COUNTS[idx_s][bit_idx][random.randint(0, 1)] += 1 * weight[i]

        print('COUNTS:', COUNTS)

        green_counts = []
        valid_counts = []
        z_scores = []
        p_values = []
        decode_bits = ['' for _ in range(len(COUNTS))]
        hit = [0 for _ in range(len(COUNTS))]
        hit_rate = []
        for i in range(len(COUNTS)):
            count = COUNTS[i]
            for j in range(len(count)):
                if count[j][0] > count[j][1]:
                    decode_bits[i] += '0'
                    if bits[j] == '0':
                        hit[i] += 1
                elif count[j][0] < count[j][1]:
                    decode_bits[i] += '1'
                    if bits[j] == '1':
                        hit[i] += 1
                else:
                    decode_bits[i] += 'x'
            hit_rate.append(hit[i]/bits_len)
            green_count = np.max(count, axis=-1).sum()
            valid_count = np.sum(count)
            z_score = self._compute_z_score(green_count, valid_count)
            p_value = self._compute_p_value(z_score)
            green_counts.append(green_count)
            valid_counts.append(valid_count)
            z_scores.append(z_score)
            p_values.append(p_value)

        print('green_counts:', green_counts)
        print('valid_counts:', valid_counts)
        print('z_scores:', z_scores)
        print('p_values:', p_values)
        print('decode_bits', decode_bits)
        return COUNTS, generate_counts, green_counts, valid_counts, z_scores, p_values, decode_bits, hit, hit_rate
    
    
    def decode_bimark_multibit_watermark(self, inputs, partition_seeds, c_key,  bit_idx_key, bits, bits_len=0, weight=0,
                               start=0, stride=50):
        if bits_len == 0:
            bits_len = len(bits)

        if weight == 0:
            weight = [1 for _ in range(len(partition_seeds))]

        stride_idx_list = [start + stride * i for i in range((len(inputs) - start)//stride +1)]       
    
        COUNTS = [[[0, 0] for _ in range(bits_len)] for _ in range(len(stride_idx_list) )]
        
        partition_masks = []
        for key in partition_seeds:
            num_V0 = int(self.vocab_size * 0.5)
            rng = np.random.default_rng(key)
            mask = np.zeros(self.vocab_size, dtype=bool)
            mask[rng.choice(self.vocab_size, num_V0, replace=False)] = True
            partition_masks.append(mask)
        print('len(partition_masks):', len(partition_masks))
    
        hist = set() 
        generate_counts = [0 for _ in range(len(stride_idx_list))]
        idx_s = 0
        for t in range(self.window_size, len(inputs)):
            try:
                generate_counts[idx_s] += 1
            except:
                continue
            if idx_s < len(stride_idx_list):
                if (t-self.window_size) == stride_idx_list[idx_s]:
                    idx_s += 1
                    try:
                        COUNTS[idx_s] = copy.deepcopy(COUNTS[idx_s-1])
                        generate_counts[idx_s] = generate_counts[idx_s-1]
                    except:
                        continue
            prefix = inputs[t - self.window_size: t]
            
            c_seed=prf(prefix, c_key) # seed
            # partition_idx_seed=prf(prefix, partition_idx_key)
            rng_idx_seed = prf(prefix, bit_idx_key)
            
            if prefix not in hist:  # do not watermarking the same seed
                hist.add(prefix)
            else:
                continue
            rng_c = np.random.default_rng(c_seed)
            
            rng_bit_idx = np.random.default_rng(rng_idx_seed)

            c_list = rng_c.integers(0, 2, size=len(partition_masks))


            bit_idx = rng_bit_idx.integers(0, bits_len)

            token_idx = inputs[t].item()

            for i in range(len(partition_masks)):
                mask = partition_masks[i]
                if ((c_list[i] == 1 and (mask[token_idx].item() is False)) or (c_list[i] == 0 and (mask[token_idx].item() is True))):
                    COUNTS[idx_s][bit_idx][1] += 1 * weight[i]
                elif ((c_list[i] == 1 and (mask[token_idx].item() is True)) or (c_list[i] == 0 and (mask[token_idx].item() is False))):
                    COUNTS[idx_s][bit_idx][0] += 1 * weight[i]
                else:
                    COUNTS[idx_s][bit_idx][random.randint(0, 1)] += 1 * weight[i]

        print('COUNTS:', COUNTS)

        green_counts = []
        valid_counts = []
        z_scores = []
        p_values = []
        decode_bits = ['' for _ in range(len(COUNTS))]
        hit = [0 for _ in range(len(COUNTS))]
        hit_rate = []
        for i in range(len(COUNTS)):
            count = COUNTS[i]
            for j in range(len(count)):
                if count[j][0] > count[j][1]:
                    decode_bits[i] += '0'
                    if bits[j] == '0':
                        hit[i] += 1
                elif count[j][0] < count[j][1]:
                    decode_bits[i] += '1'
                    if bits[j] == '1':
                        hit[i] += 1
                else:
                    decode_bits[i] += 'x'
            hit_rate.append(hit[i]/bits_len)
            green_count = np.max(count, axis=-1).sum()
            valid_count = np.sum(count)
            z_score = self._compute_z_score(green_count, valid_count)
            p_value = self._compute_p_value(z_score)
            green_counts.append(green_count)
            valid_counts.append(valid_count)
            z_scores.append(z_score)
            p_values.append(p_value)

        print('green_counts:', green_counts)
        print('valid_counts:', valid_counts)
        print('z_scores:', z_scores)
        print('p_values:', p_values)
        print('decode_bits', decode_bits)
        return COUNTS, generate_counts, green_counts, valid_counts, z_scores, p_values, decode_bits, hit, hit_rate
    

    def verify_bimark_multibit(self, detect_gen_tokens, partition_seeds,  c_key,  bit_idx_key, 
                               bits, start=0, weight=0, stride=50):
        if weight == 0:
            weight = [1 for _ in range(partition_seeds)]
        
        partition_masks = []
        for key in partition_seeds:
            num_V0 = int(self.vocab_size * 0.5)
            rng = np.random.default_rng(key)
            mask = np.zeros(self.vocab_size, dtype=bool)
            mask[rng.choice(self.vocab_size, num_V0, replace=False)] = True
            partition_masks.append(mask)
        stride_idx_list = [start + stride * i for i in range((len(detect_gen_tokens) - start)//stride +1 )]       
        
        print('stride_idx_list:', stride_idx_list)
        green_count = [0 for _ in range(len(stride_idx_list) )]
        generate_count = [0 for _ in range(len(stride_idx_list) )]
        valid_count = [0 for _ in range(len(stride_idx_list) )]
        bits_green_count = [[0 for _ in range(len(bits))] for _ in range(len(stride_idx_list) )]
        bits_valid_count = [[0 for _ in range(len(bits))] for _ in range(len(stride_idx_list) )]

        hist = set()
        idx_s = 0
        for t in range(self.window_size, detect_gen_tokens.shape[-1]):
            try:
                generate_count[idx_s] += len(partition_masks)
            except:
                continue
            if idx_s < len(stride_idx_list):
                if (t-self.window_size) == stride_idx_list[idx_s]:
                    idx_s += 1
                    try:
                        generate_count[idx_s] = int(generate_count[idx_s-1])
                        green_count[idx_s] = int(green_count[idx_s-1])
                        valid_count[idx_s] = int(valid_count[idx_s-1])
                        bits_green_count[idx_s] = [item for item in bits_green_count[idx_s-1]]
                        bits_valid_count[idx_s] = [item for item in bits_valid_count[idx_s-1]]
                    except:
                        continue

            prefix = detect_gen_tokens[t - self.window_size: t]

            c_seed=prf(prefix, c_key) # seed
            rng_idx_seed = prf(prefix, bit_idx_key)
            
            if prefix not in hist:  # do not watermarking the same seed
                hist.add(prefix)
            else:
                continue
            rng_c = np.random.default_rng(c_seed)
            rng_bit_idx = np.random.default_rng(rng_idx_seed)

            c_list = rng_c.integers(0, 2, size=len(partition_masks))

            bit_idx = rng_bit_idx.integers(0, len(bits))
            bit = int(bits[bit_idx])

            token_idx = detect_gen_tokens[t].item()

            for i in range(len(partition_masks)):
                mask = partition_masks[i]
                if ((c_list[i] == 1 and bit == 0) or (c_list[i] == 0 and bit == 1)):
                    if mask[token_idx].item() is True:
                        green_count[idx_s] += 1 * weight[i]
                        bits_green_count[idx_s][bit_idx] += 1 * weight[i]
                elif ((c_list[i] == 1 and bit == 1) or (c_list[i] == 0 and bit == 0)):
                    if mask[token_idx].item() is False:
                        green_count[idx_s] += 1 * weight[i]
                        bits_green_count[idx_s][bit_idx] += 1 * weight[i]
                bits_valid_count[idx_s][bit_idx] += 1 * weight[i]
                valid_count[idx_s] += 1 * weight[i]
        print('bits_green_count:', bits_green_count)
        print('bits_valid_count:', bits_valid_count)
        z_score, z_p_value, stride_list = self._zerobit_watermark_detector_stride(np.array(green_count), valid_count, stride)
        
        z_score_bits = [[0 for _ in range(len(bits))] for _ in range(len(stride_idx_list))]
        z_p_value_bits = [[0 for _ in range(len(bits))] for _ in range(len(stride_idx_list))]
        for i in range(len(stride_idx_list)):
            for j in range(len(bits)):
                try:
                    z_score_bit = self._compute_z_score(bits_green_count[i][j], bits_valid_count[i][j])
                    z_p_value_bit = self._compute_p_value(z_score_bit)
                    z_score_bits[i][j] = z_score_bit
                    z_p_value_bits[i][j] = z_p_value_bit
                except:
                    z_score_bits[i][j] = 0
                    z_p_value_bits[i][j] = 1
        print('z_score_bits:', z_score_bits)
        print('z_p_value_bits:', z_p_value_bits)
        z_score = list(map(float, z_score))
        z_p_value = list(map(float, z_p_value))
        
        print('z_score_bits:', z_score_bits)
        return z_score, z_p_value, green_count,  generate_count, valid_count, stride_list, bits_green_count, bits_valid_count, z_score_bits, z_p_value_bits
            