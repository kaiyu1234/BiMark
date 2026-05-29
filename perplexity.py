
import logging
from typing import List
from tqdm import tqdm
import tiktoken
from statistics import mean
from math import exp
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
    

class LocalModel:
    """ Local Language Model. """
    
    def __init__(self, model_name: str, device):
        """ Local Language Model.
        
        @param model_path: Path to the local model.
        """
        logging.info(f'Loading Model from: `{model_name}`')
        auth_token = "****"
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype=torch.bfloat16, use_auth_token=auth_token)
        # # self.tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        if 'gpt2' in model_name.lower():
            model_dtype = torch.float32 # GPT-2 原生精度
        else:
            model_dtype = torch.bfloat16

        # 2. 加载 Tokenizer
        # GPT-2 不需要 auth_token，但传了也不会报错（如果库版本不严格）。为了严谨可以按需传。
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype=model_dtype, token=auth_token)
        except TypeError:
            # 兼容老版本的 transformers (use_auth_token)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype=model_dtype, token=auth_token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, torch_dtype=model_dtype, token=auth_token)
        except TypeError:
             self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, torch_dtype=model_dtype, use_auth_token=auth_token)
            
        # self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, torch_dtype=torch.bfloat16, use_auth_token=auth_token)
        # # self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, torch_dtype=torch.bfloat16)
        self.model.eval()  # Set the model to evaluation mode
        
    def get_perplexity(self, prompt: str, input_texts: List[str], *args, **kwargs):
        """ Compute the perplexity on the local language model.
        
        :param prompt: The prompt to be prepended to each input text.
        :param input_texts: A list of input texts for evaluation.
        :return: A list of perplexity values.
        """
        ppl_list = []
        prompt_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.model.device)
        prompt_len = prompt_ids.size(1)
        
        #--------------------------------------------------------------------------------------------------
        original_prompt_len = prompt_ids.size(1)
        max_len = getattr(self.model.config, 'max_position_embeddings', 1024)
        if hasattr(self.model.config, 'n_positions'):
            max_len = min(max_len, self.model.config.n_positions)
        #--------------------------------------------------------------------------------------------------

        
        for text in tqdm(input_texts):
            full_text = prompt + text
            inputs = self.tokenizer(full_text, return_tensors='pt').to(self.model.device)

            #--------------------------------------------------------------------------------------------------
            seq_len = inputs.input_ids.size(1)
            prompt_len = original_prompt_len
            if seq_len > max_len:
                truncate_len = seq_len - max_len
                # 从左侧截掉超出的部分
                inputs.input_ids = inputs.input_ids[:, truncate_len:]
                if 'attention_mask' in inputs:
                    inputs.attention_mask = inputs.attention_mask[:, truncate_len:]
                    
                # 同步调整 prompt_len，确保在切片计算 logits 时不越界
                # max(1, ...) 保证至少留 1 个 token 给 prompt，防止出现空切片
                prompt_len = max(1, original_prompt_len - truncate_len)
                #--------------------------------------------------------------------------------------------------


            
            with torch.no_grad():
                outputs = self.model(**inputs)
            # print('outputs.logits.shape', outputs.logits.shape)
            print('otuptus.logits', outputs.logits)
            logits = outputs.logits[0, prompt_len-1:-1, :]
            target_ids = inputs.input_ids[0, prompt_len:]
            
            log_probs = torch.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            
            if len(token_log_probs) > 0:
                nll = -token_log_probs.mean().item()
                ppl = exp(nll)
                ppl_list.append(ppl)
            else:
                ppl_list.append(-1)  # Error case
        
        return ppl_list