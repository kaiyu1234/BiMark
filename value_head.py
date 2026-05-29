import torch
import torch.nn as nn
import torch.nn.functional as F

class WatermarkValueHead(nn.Module):
    def __init__(self, hidden_dim=4096, max_error_dim=64, vocab_size=128256):
        super().__init__()
        self.max_error_dim = max_error_dim # 定义模型支持的最大水印长度
        
        # 输入维度固定为：隐藏层维度 + 最大误差维度
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + max_error_dim, 1024),
            nn.SiLU(), 
            nn.Linear(1024, 256),
            nn.SiLU(),
            nn.Linear(256, vocab_size) 
        )

    def forward(self, h_t, E_t):
        """
        h_t: (batch_size, seq_len, 4096) 或是推理时的 (batch_size, 4096)
        E_t: (batch_size, seq_len, L) 或是推理时的 (batch_size, L) 
             这里的 L 可以是 16, 32，只要不大于 max_error_dim 即可
        """
        current_L = E_t.shape[-1]
        
        # ==========================================
        # 魔法发生的地方：动态零填充 (Zero-Padding)
        # ==========================================
        if current_L < self.max_error_dim:
            # 计算需要补多少个 0
            pad_len = self.max_error_dim - current_L
            
            # F.pad 的参数 (0, pad_len) 表示：
            # 在最后一个维度的左边补 0 个，右边补 pad_len 个
            # 这样不管 E_t 是 2 维还是 3 维张量，都能完美兼容
            E_t_padded = F.pad(E_t, (0, pad_len), "constant", 0)
            
        elif current_L > self.max_error_dim:
            raise ValueError(f"当前水印长度 {current_L} 超过了模型支持的最大长度 {self.max_error_dim}")
        else:
            # 长度正好等于最大长度，无需填充
            E_t_padded = E_t
            
        # 拼接固定维度的张量并输入网络
        x = torch.cat([h_t, E_t_padded], dim=-1)
        return self.mlp(x)