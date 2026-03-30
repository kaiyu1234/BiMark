import numpy as np
import random
from math import sqrt
from utils import prf
from scipy import stats
import copy

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
            
            rng = np.random.default_rng(c_seed)
            # 重建当前上下文的整个词表的 L 维指纹
            F_np = rng.integers(0, 2, size=(self.vocab_size, bits_len)) * 2 - 1
            
            # 获取语言模型实际生成的词
            token_idx = inputs[t].item()
            # 提取该生成词所附带的 L 维特征向量，并叠加到全局积分器上
            F_v = F_np[token_idx]
            
            V_decode += F_v
            valid_tokens += 1

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
            