# 第 7 课：_ResamplerBlock 与注意力机制

## 课程目标

- 深入理解 MultiheadAttention 的使用细节
- 掌握分离 LayerNorm 的设计原因
- 理解前馈网络 (FFN) 的结构与作用
- 理解残差连接在深度学习中的重要性

---

## 7.1 _ResamplerBlock 概述

### 7.1.1 位置与定义

**位置**：`methods/vision_latent_mas_codec_new.py:522`

```python
class _ResamplerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ln_q = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )
```

### 7.1.2 结构图解

```
输入: q [B, Q_len, D], kv [B, KV_len, D]
           │                              │
           │                              │
           ▼                              ▼
    ┌───────────────┐              ┌───────────────┐
    │    ln_q       │              │    ln_kv     │
    │  (Query LN)   │              │  (Key-Value LN) │
    └───────────────┘              └───────────────┘
           │                              │
           │                              │
           ▼                              ▼
    ┌─────────────────────────────────────────────┐
    │         MultiheadAttention                   │
    │         (q, kv, kv)                        │
    │         need_weights=False                  │
    └─────────────────────────────────────────────┘
           │
           ▼
    ┌───────────────┐
    │      +        │  ← 第一次残差连接
    │     (q)       │
    └───────────────┘
           │
           ▼
    ┌───────────────┐
    │      ff       │  ← 前馈网络
    │   (MLP + LN)  │
    └───────────────┘
           │
           ▼
    ┌───────────────┐
    │      +        │  ← 第二次残差连接
    │     (q)       │
    └───────────────┘
           │
           ▼
    输出: q [B, Q_len, D]
```

---

## 7.2 MultiheadAttention 详解

### 7.2.1 PyTorch 的 MultiheadAttention

```python
nn.MultiheadAttention(
    embed_dim,      # D，嵌入维度
    num_heads,      # n_heads，注意力头数
    dropout=0.0,
    batch_first=True # 输入/输出格式：[batch, seq, dim]
)
```

### 7.2.2 前向传播

```python
out, _ = self.attn(
    query,    # Q 查询
    key,      # K 键
    value,    # V 值
    key_padding_mask=kv_key_padding_mask,
    need_weights=False  # 不返回注意力权重
)
```

### 7.2.3 数学原理

**注意力分数计算**：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**多头版本**：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

其中每个 $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$

### 7.2.4 在 ResamplerBlock 中的使用

```python
out, _ = self.attn(q2, kv2, kv2,
                   key_padding_mask=kv_key_padding_mask,
                   need_weights=False)
#             ↑  ↑
#             │  └── Key 和 Value 都来自 kv2
#             └── Query 来自 q2
```

**关键特性**：

| 参数 | 值 | 说明 |
|------|-----|------|
| `query` | q2 | 来自可学习 Query |
| `key` | kv2 | 来自输入序列 |
| `value` | kv2 | 来自输入序列 |
| `key_padding_mask` | 有效位置掩码 | 忽略 padding |
| `need_weights` | False | 不需要注意力权重 |

---

## 7.3 分离 LayerNorm 的设计

### 7.3.1 为什么分离 Q 和 KV 的 LN？

```python
self.ln_q = nn.LayerNorm(dim)   # Query 独立归一化
self.ln_kv = nn.LayerNorm(dim)  # Key-Value 独立归一化
```

**原因分析**：

| 输入 | 来源 | 分布特点 |
|------|------|----------|
| q | 可学习参数（q_sem, q_global, q_style） | 初始化为均值为0、标准差0.02的正态分布 |
| kv | 输入序列（经过 in_proj 投影） | 分布取决于输入数据 |

由于 Q 和 KV 来自**不同的分布**，独立归一化可以：

1. **独立适配**：各自学习最优的归一化参数
2. **训练稳定**：避免不同分布导致的梯度问题
3. **表达灵活**：增加模型容量

### 7.3.2 LayerNorm 的作用

$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

其中：
- $\mu = \frac{1}{d}\sum_i x_i$：均值
- $\sigma^2 = \frac{1}{d}\sum_i (x_i - \mu)^2$：方差
- $\gamma, \beta$：可学习的缩放和偏移参数

**效果**：将激活值归一化到均值为0、方差为1的分布，再通过 $\gamma, \beta$ 调整。

### 7.3.3 对比：为什么不只用一个大 LN？

```python
# 方案 A：分离 LN（当前使用）
self.ln_q = nn.LayerNorm(dim)
self.ln_kv = nn.LayerNorm(dim)

# 方案 B：共享 LN
self.ln_shared = nn.LayerNorm(dim)
q2 = self.ln_shared(q)
kv2 = self.ln_shared(kv)
```

**分离 LN 的优势**：

| 方面 | 分离 LN | 共享 LN |
|------|---------|---------|
| 参数数量 | 2 × (γ + β) | 1 × (γ + β) |
| 表达能力 | 更强 | 较弱 |
| 训练稳定性 | 更稳定 | 可能不稳定 |

---

## 7.4 前馈网络 (Feed-Forward Network)

### 7.4.1 结构

```python
self.ff = nn.Sequential(
    nn.LayerNorm(dim),
    nn.Linear(dim, 4 * dim),
    nn.GELU(),
    nn.Linear(4 * dim, dim),
)
```

**数学形式**：

$$\text{FFN}(x) = \text{Linear}_2(\text{GELU}(\text{Linear}_1(x)))$$

展开为：

$$h' = W_2 \cdot \text{GELU}(W_1 \cdot x + b_1) + b_2$$

其中 $W_1 \in \mathbb{R}^{d \times 4d}$，$W_2 \in \mathbb{R}^{4d \times d}$

### 7.4.2 维度变化

```
输入: [B, Q_len, D]
         │
         ▼
    LN(dim)
         │
         ▼
    Linear(D → 4D)
         │
         ▼
    GELU 激活
         │
         ▼
    Linear(4D → D)
         │
         ▼
输出: [B, Q_len, D]
```

### 7.4.3 为什么扩展因子是 4？

这是 Transformer 的标准设置（见《Attention is All You Need》）。

**考虑**：

| 扩展因子 | 参数量 | 计算量 | 表达能力 |
|----------|--------|--------|----------|
| 2 | D×2D + 2D×D = 4D² | 较低 | 较弱 |
| 4 | D×4D + 4D×D = 8D² | 中等 | 标准 |
| 6 | D×6D + 6D×D = 12D² | 较高 | 很强 |

**选择 4 的理由**：在表达能力和计算效率之间取得平衡。

### 7.4.4 GELU 激活函数

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \tanh\left(\sqrt{\frac{2}{\pi}}\left(x + 0.044715x^3\right)\right)\right]$$

**对比其他激活函数**：

| 激活函数 | 公式 | 特点 |
|----------|------|------|
| ReLU | $\max(0, x)$ | 简单，稀疏激活 |
| GELU | $x \cdot \Phi(x)$ | 平滑，概率解释 |
| SwiGLU | $\text{swish}(x) \cdot \sigma(x)$ | 门控，更强表达 |

**GELU 的优势**：
- 平滑连续，梯度流动更好
- 近似概率加权，理论意义好
- 在 Transformer 中表现优异

---

## 7.5 残差连接 (Residual Connection)

### 7.5.1 两次残差连接

```python
# 第一次残差：注意力之后
q = q + out

# 第二次残差：前馈网络之后
q = q + self.ff(q)
```

### 7.5.2 残差的数学意义

**没有残差**：
$$h' = f(h)$$

**有残差**：
$$h' = h + f(h)$$

这意味着：
- $f(h)$ 学习的是**残差**（相对于输入的变化）
- 当 $f(h) \approx 0$ 时，$h' \approx h$（恒等映射）
- 梯度可以直接流过加法操作

### 7.5.3 残差的优势

| 优势 | 说明 |
|------|------|
| **梯度流畅** | 梯度可以直接从输出传到输入 |
| **训练稳定** | 缓解梯度消失/爆炸问题 |
| **易于优化** | 底层可以学习恒等映射 |
| **特征保留** | 原始信息不会被破坏 |

### 7.5.4 恒等映射的直觉

```
深层网络中的问题：
  如果底层需要学习恒等映射，但中间层复杂
  f(x) = x 需要精确学习

残差解决：
  只需要学习 f(x) ≈ 0
  h' = h + f(h) ≈ h
```

---

## 7.6 完整前向传播解析

### 7.6.1 代码

```python
def forward(self, q, kv, kv_key_padding_mask=None):
    # 1. Query 归一化
    q2 = self.ln_q(q)

    # 2. Key-Value 归一化
    kv2 = self.ln_kv(kv)

    # 3. 多头注意力
    out, _ = self.attn(q2, kv2, kv2,
                       key_padding_mask=kv_key_padding_mask,
                       need_weights=False)

    # 4. 第一次残差连接
    q = q + out

    # 5. 前馈网络
    q = q + self.ff(q)

    return q
```

### 7.6.2 计算顺序

```
输入 q, kv
    │
    ├──▶ ln_q ──▶ attn(q2, kv2, kv2) ──▶ + q ──▶ ff ──▶ + q ──▶ 输出
    │                                                    ↑
    └──▶ ln_kv ──────────────────────────────────────────┘
```

### 7.6.3 与标准 Transformer Block 的对比

| 组件 | 标准 Transformer | ResamplerBlock |
|------|------------------|----------------|
| LayerNorm 位置 | Pre-LN 或 Post-LN | Pre-LN（分离） |
| Q, K, V 处理 | Q, K, V 共享输入 | Q, KV 分离 LN |
| 残差连接 | 2 个 | 2 个 |
| FFN | 标准 | 标准 |

**Pre-LN vs Post-LN**：

```
Post-LN (标准):
  x → LN → Attention → + → LN → FFN → + → out

Pre-LN (ResamplerBlock 使用):
  x → [+ → Attention → + → FFN →] with LN at start
```

---

## 7.7 在编解码器中的使用

### 7.7.1 Encoder 中的使用

```python
# Encoder: q 是 Query，kv 是输入序列
for blk in self.blocks:
    q = blk(q, kv, kv_key_padding_mask=latents_key_padding_mask)
```

**场景**：用 Query 从输入序列中提取信息

### 7.7.2 Decoder 中的使用

```python
# Decoder: q 是可学习 Query，kv 是通用表示
for blk in self.blocks:
    q = blk(q, kv, kv_key_padding_mask=U_key_padding_mask)
```

**场景**：用 Query 从通用表示中生成输出

---

## 7.8 设计哲学总结

### 7.8.1 为什么这样设计？

| 设计选择 | 理由 |
|----------|------|
| **MultiheadAttention** | 捕捉多种关联模式 |
| **分离 LN** | Q 和 KV 分布不同 |
| **FFN 扩展 4x** | 标准 Transformer 设置 |
| **GELU** | 平滑激活，理论好 |
| **两次残差** | 梯度流畅，信息保留 |

### 7.8.2 核心思想

> **通过 Query-based 注意力机制，让固定数量的可学习 Query 从输入序列中提取、重组信息**

---

## 7.9 本课小结

### 关键要点

1. **MultiheadAttention**：
   - query 来自可学习 Query
   - key-value 来自输入序列
   - `need_weights=False` 不返回注意力权重
2. **分离 LayerNorm**：Q 和 KV 分布不同，独立归一化
3. **FFN**：扩展因子 4，GELU 激活
4. **残差连接**：两次残差，梯度流畅 + 信息保留
5. **Pre-LN 风格**：LN 在操作之前

### 思考题

1. 如果 `need_weights=True`，返回的注意力权重有什么用途？在这个设计中有帮助吗？
2. 为什么 FFN 的扩展因子选择 4 而不是其他值？太小或太大会有什么影响？
3. 残差连接要求输入输出维度相同。在这个项目中是如何保证的？

### 下一课预告

**第 8 课：编解码器的协同工作流程**

- Sender → Encoder → 传输 → Decoder → Receiver 完整流程
- 维度转换的完整追踪
- 虚拟图像的注入机制
- 端到端通信的示例

---

*返回 [课程大纲](./课程大纲.md)*
