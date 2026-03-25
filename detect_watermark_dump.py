import numpy as np
import random
from math import sqrt
from utils import prf, init_eh_state, eh_select_bit, eh_should_embed
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
                               start=0, stride=50, eh_enable=False, eh_state_key=99431, eh_sched_key=137631,
                               eh_candidate_width=8, eh_min_credit=0.35, eh_allow_skip=False):
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
        eh_states = [0]
        bit_counts = [[0 for _ in range(bits_len)]]
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
                        eh_states.append(eh_states[idx_s-1])
                        bit_counts.append(copy.deepcopy(bit_counts[idx_s-1]))
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
            
            c_list = rng_c.integers(0, 2, size=len(partition_masks))
            if eh_enable:
                if t == self.window_size:
                    eh_states[idx_s] = init_eh_state(prefix, eh_state_key, bits_len)
                bit_idx, next_state, credit = eh_select_bit(
                    prefix=prefix,
                    state=eh_states[idx_s],
                    bit_counts=bit_counts[idx_s],
                    bits_len=bits_len,
                    state_key=eh_state_key,
                    sched_key=eh_sched_key,
                    candidate_width=eh_candidate_width
                )
                eh_states[idx_s] = next_state
                should_embed = eh_should_embed(
                    credit=credit,
                    bit_counts=bit_counts[idx_s],
                    bit_idx=bit_idx,
                    step_idx=t,
                    total_steps=len(inputs),
                    min_credit=eh_min_credit
                )
                if (not should_embed) and (not eh_allow_skip):
                    rng_bit_idx = np.random.default_rng(rng_idx_seed)
                    bit_idx = rng_bit_idx.integers(0, bits_len)
                    should_embed = True
            else:
                rng_bit_idx = np.random.default_rng(rng_idx_seed)
                bit_idx = rng_bit_idx.integers(0, bits_len)
                should_embed = True
            if not should_embed:
                continue
            if eh_enable:
                bit_counts[idx_s][bit_idx] += 1

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
                               bits, start=0, weight=0, stride=50, eh_enable=False,
                               eh_state_key=99431, eh_sched_key=137631, eh_candidate_width=8, eh_min_credit=0.35,
                               eh_allow_skip=False):
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
        eh_states = [0]
        bit_counts_track = [[0 for _ in range(len(bits))]]
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
                        eh_states.append(eh_states[idx_s-1])
                        bit_counts_track.append([item for item in bit_counts_track[idx_s-1]])
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
            c_list = rng_c.integers(0, 2, size=len(partition_masks))
            if eh_enable:
                if t == self.window_size:
                    eh_states[idx_s] = init_eh_state(prefix, eh_state_key, len(bits))
                bit_idx, next_state, credit = eh_select_bit(
                    prefix=prefix,
                    state=eh_states[idx_s],
                    bit_counts=bit_counts_track[idx_s],
                    bits_len=len(bits),
                    state_key=eh_state_key,
                    sched_key=eh_sched_key,
                    candidate_width=eh_candidate_width
                )
                eh_states[idx_s] = next_state
                should_embed = eh_should_embed(
                    credit=credit,
                    bit_counts=bit_counts_track[idx_s],
                    bit_idx=bit_idx,
                    step_idx=t,
                    total_steps=detect_gen_tokens.shape[-1],
                    min_credit=eh_min_credit
                )
                if (not should_embed) and (not eh_allow_skip):
                    rng_bit_idx = np.random.default_rng(rng_idx_seed)
                    bit_idx = rng_bit_idx.integers(0, len(bits))
                    should_embed = True
            else:
                rng_bit_idx = np.random.default_rng(rng_idx_seed)
                bit_idx = rng_bit_idx.integers(0, len(bits))
                should_embed = True
            if not should_embed:
                continue
            if eh_enable:
                bit_counts_track[idx_s][bit_idx] += 1
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
            
