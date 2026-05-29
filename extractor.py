# extractor.py
import torch

class HiddenStateExtractor:
    def __init__(self, model):
        self.h_t = None
        # Llama-3 的倒数第一层 transformer block 是 model.model.layers[-1]
        self.hook = model.model.layers[-1].register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # 截获当前正在生成的这最后一个 token 的隐状态
        self.h_t = output[0][:, -1, :].detach().clone()
        
    def remove(self):
        self.hook.remove()