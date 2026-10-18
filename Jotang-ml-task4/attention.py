import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ================= 1. 手写单头 Scaled Dot-Product Attention =================
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K, V 的形状均为: (batch_size, seq_len, d_k)
    """
    d_k = Q.size(-1) # 特征维度，比如 64
    
    # 1. Q 和 K^T 点积，除以 sqrt(d_k)
    # 维度变化: (B, L, d_k) * (B, d_k, L) -> (B, L, L)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    
    # 2. 如果有 mask（比如 Decoder 防止看未来），就把该遮的地方填成负无穷
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
    # 3. Softmax 归一化，得到注意力权重
    attn_weights = F.softmax(scores, dim=-1)
    
    # 4. 乘以 V
    # 维度变化: (B, L, L) * (B, L, d_k) -> (B, L, d_k)
    output = torch.matmul(attn_weights, V)
    
    return output, attn_weights

# ================= 2. 手写多头注意力 (Multi-Head Attention) =================
class MyMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MyMultiHeadAttention, self).__init__()
        # 确保维度能被头数整除
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 定义四个线性层 (Q, K, V 的映射，以及最后的输出映射)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # 线性映射并拆分成多头
        # (B, L, d_model) -> (B, L, num_heads, d_k) -> permute -> (B, num_heads, L, d_k)
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 对每个头计算 Attention (形状: B, num_heads, L, d_k)
        out, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        
        # 拼接多头结果
        # (B, num_heads, L, d_k) -> transpose -> (B, L, num_heads, d_k) -> view -> (B, L, d_model)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # 最后的输出线性层
        out = self.W_o(out)
        
        return out, attn_weights

# ================= 3. 测试并绘制热力图 =================
if __name__ == '__main__':
    # 构造随机数据 (batch_size=1, 序列长度=10, 特征维度=512)
    seq_len = 10
    d_model = 512
    x = torch.randn(1, seq_len, d_model)
    
    # 实例化模型 (8个头)
    mha = MyMultiHeadAttention(d_model, num_heads=8)
    out, attn_weights = mha(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")
    print(f"注意力权重形状 (B, Heads, Seq, Seq): {attn_weights.shape}")
    
    # 画热力图 (取第一个头的权重矩阵)
    plt.figure(figsize=(6, 5))
    plt.imshow(attn_weights[0, 0].detach().numpy(), cmap='viridis')
    plt.colorbar(label='Attention Weight')
    plt.title("Multi-Head Attention 权重热力图 (Head 1)")
    plt.xlabel("Key Index")
    plt.ylabel("Query Index")
    plt.savefig('attention_heatmap.png', dpi=150)
    print("热力图已保存")