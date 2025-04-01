
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
        auth_token = "xxx"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype=torch.bfloat16, use_auth_token=auth_token)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, torch_dtype=torch.bfloat16, use_auth_token=auth_token)
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
        
        for text in tqdm(input_texts):
            full_text = prompt + text
            inputs = self.tokenizer(full_text, return_tensors='pt').to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            print('outputs.logits.shape', outputs.logits.shape)
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