# 第 9 课：ModelWrapper 设计与实现

## 课程目标

- 理解 ModelWrapper 在项目中的核心作用
- 掌握 VLM 的加载与管理机制
- 理解 KV Cache 操作函数
- 掌握 `generate_latent_batch` 潜在向量生成流程
- 理解多模态模型的检测与适配

---

## 9.1 ModelWrapper 概述

### 9.1.1 类的位置与作用

**位置**：`models.py:279`

```python
class ModelWrapper:
    def __init__(self, model_name: str, device: torch.device, use_vllm: bool = False, args=None):
        # 负责加载和管理 VLM
```

### 9.1.2 核心职责

| 职责 | 说明 |
|------|------|
| **模型加载** | 从 Hugging Face 加载 VLM |
| **Tokenizer 管理** | 处理不同模型的 tokenizer |
| **文本生成** | `generate_text_batch` 方法 |
| **潜在向量生成** | `generate_latent_batch` 方法 |
| **潜在对齐** | `_apply_latent_realignment` 方法 |

### 9.1.3 核心属性

```python
class ModelWrapper:
    def __init__(self, model_name: str, device: torch.device, ...):
        self.model_name = model_name          # 模型名称
        self.device = device                  # 设备 (cuda:0, etc.)
        self.use_vllm = use_vllm              # 是否使用 vLLM
        self.vllm_engine = None               # vLLM 引擎
        self.model = None                    # Hugging Face 模型
        self.tokenizer = None                # Tokenizer

        # 模型类型标记
        self.is_internvl = False             # 是否 InternVL
        self.is_smolvlm = False              # 是否 SmolVLM
        self.is_lfm2_vl = False             # 是否 LFM2-VL
        self.is_minicpm_v = False           # 是否 MiniCPM-V

        # 潜在空间对齐
        self.latent_space_realign = False
        self._latent_realign_matrices: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
```

---

## 9.2 VLM 加载机制

### 9.2.1 加载流程

```python
def __init__(self, model_name: str, device: torch.device, use_vllm: bool = False, args=None):
    # 1. 检测模型类型
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    is_multimodal = _is_multimodal_config(cfg)

    # 2. 标记特定模型
    self.is_internvl = _is_internvl_config(cfg)
    self.is_smolvlm = _is_smolvlm_config(cfg)
    self.is_lfm2_vl = _is_lfm2_vl_config(cfg)
    self.is_minicpm_v = _is_minicpm_v_config(cfg)

    # 3. vLLM 或 Hugging Face 加载
    if use_vllm:
        # 使用 vLLM 引擎
        self.vllm_engine = LLM(model=model_name, ...)
    else:
        # 使用 Hugging Face
        if is_multimodal:
            model_cls = AutoModelForImageTextToText
        else:
            model_cls = AutoModelForCausalLM
        self.model = model_cls.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
```

### 9.2.2 多模态模型检测

```python
# models.py:110-133
def _is_multimodal_config(cfg) -> bool:
    model_type = str(getattr(cfg, "model_type", "")).lower()
    class_name = cfg.__class__.__name__.lower()

    # Qwen-VL 系列
    if any(key in model_type for key in ("qwen", "gemma3")) and \
       any(key in model_type for key in ("vl", "_vl")):
        return True

    # LFM2-VL
    if "lfm2_vl" in model_type or "lfm2vl" in class_name:
        return True

    # SmolVLM
    if "smolvlm" in model_type or "smolvlm" in class_name:
        return True

    # InternVL
    if "internvl" in model_type or "internvl" in class_name:
        return True

    # MiniCPMV
    if "minicpmv" in model_type or "minicpmv" in class_name:
        return True

    # ... 其他检测
```

### 9.2.3 特定模型适配

```python
# models.py:384-389
def _get_text_model(self):
    """获取文本解码器，处理不同模型的差异"""
    if self.is_internvl and hasattr(self.model, "language_model"):
        return self.model.language_model
    if self.is_minicpm_v and hasattr(self.model, "llm"):
        return self.model.llm
    return self.model
```

---

## 9.3 KV Cache 操作

### 9.3.1 缓存结构

KV Cache 是 LLM 推理优化的核心技术：

```
无 Cache:
  Token 1 → Forward → Hidden 1
  Token 2 → Forward → Hidden 2 (需要重新计算 Token 1)
  Token 3 → Forward → Hidden 3 (需要重新计算 Token 1, 2)
  ...

有 Cache:
  Token 1 → Forward → Hidden 1 + KV Cache
  Token 2 → Forward → Hidden 2 + KV Cache (复用 Token 1 的 KV)
  Token 3 → Forward → Hidden 3 + KV Cache (复用 Token 1, 2 的 KV)
  ...
```

### 9.3.2 缓存转换函数

**位置**：`models.py:44-77`

```python
def _layer_to_kv(layer: Any) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """从不同格式的 layer 中提取 key-value"""
    if isinstance(layer, (tuple, list)) and len(layer) == 2:
        return layer[0], layer[1]
    if hasattr(layer, "key_cache") and hasattr(layer, "value_cache"):
        return layer.key_cache, layer.value_cache
    if hasattr(layer, "k_cache") and hasattr(layer, "v_cache"):
        return layer.k_cache, layer.v_cache
    # ... 其他格式
    return None

def _cache_to_legacy_list(past_key_values: Any) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """将各种格式的 KV Cache 转为统一格式"""
    if past_key_values is None:
        return []
    if isinstance(past_key_values, (list, tuple)):
        # 已经是 list 格式
        out = []
        for layer in past_key_values:
            kv = _layer_to_kv(layer)
            if kv is not None:
                out.append(kv)
        return out
    if hasattr(past_key_values, "key_cache"):
        # 某些模型的格式
        return list(zip(past_key_values.key_cache, past_key_values.value_cache))
    return []
```

### 9.3.3 缓存长度计算

```python
# models.py:80-87
def _past_length(past_key_values: Any) -> int:
    """获取 KV Cache 的序列长度"""
    legacy = _cache_to_legacy_list(past_key_values)
    if not legacy:
        return 0
    k = legacy[0][0]
    if not torch.is_tensor(k) or k.ndim < 3:
        return 0
    return int(k.shape[-2])
```

### 9.3.4 缓存切片

```python
# models.py:90-100
def _slice_past(past_key_values: Any, idx: int) -> Optional[Tuple]:
    """从 KV Cache 中切片出特定样本的数据"""
    if past_key_values is None:
        return None
    legacy = _cache_to_legacy_list(past_key_values)
    if not legacy:
        return None
    sliced = []
    for k, v in legacy:
        if torch.is_tensor(k) and torch.is_tensor(v):
            sliced.append((k[idx : idx + 1].contiguous(),
                          v[idx : idx + 1].contiguous()))
    return tuple(sliced) if sliced else None
```

---

## 9.4 generate_latent_batch 详解

### 9.4.1 功能概述

`generate_latent_batch` 是 Vision Wormhole 的核心方法之一，用于**生成潜在向量序列**。

### 9.4.2 方法签名

```python
# models.py:696-788
@torch.no_grad()
def generate_latent_batch(
    self,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    *,
    latent_steps: int,        # 生成多少步潜在向量
    past_key_values: Optional[Tuple] = None,
    return_latent_embeds: bool = False,
) -> Tuple:
```

### 9.4.3 核心逻辑

```python
def generate_latent_batch(self, input_ids, attention_mask=None, *, latent_steps, ...):
    # 1. 首次 forward：获取初始隐藏状态
    outputs = text_model(**fwd)
    past = outputs.past_key_values
    last_hidden = outputs.hidden_states[-1][:, -1, :]  # [B, H]

    latent_embeds: List[torch.Tensor] = []

    # 2. 多步潜在向量生成
    for step in range(latent_steps):
        # 2.1 应用潜在对齐
        latent_vec = self._apply_latent_realignment(last_hidden, text_model)

        # 2.2 保存潜在向量
        latent_embeds.append(latent_vec.detach().clone())

        # 2.3 用潜在向量作为输入继续 forward
        # 这样可以生成更多潜在状态
        latent_embed = latent_vec.unsqueeze(1)  # [B, 1, H]
        outputs = text_model(inputs_embeds=latent_embed, ...)
        past = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1, :]

    # 3. 返回
    if return_latent_embeds:
        latent_stack = torch.stack(latent_embeds, dim=1)  # [B, latent_steps, H]
        return past, latent_stack
    return past
```

### 9.4.4 维度追踪

```
输入: input_ids [B, L]
         │
         ▼
    首次 forward
         │
         ▼
    last_hidden [B, H]  ← 最后一个 token 的隐藏状态
         │
         ├──────────────────────┐
         │                      │
         ▼                      ▼
    latent_steps 次循环:
         │
         │  latent_vec [B, H]
         │       │
         │       ├───▶ 保存到 latent_embeds
         │       │
         │       └───▶ unsqueeze(1) → [B, 1, H]
         │                              │
         │                              ▼
         │                         forward
         │                              │
         │                              ▼
         │                      last_hidden [B, H]
         │                              │
         └──────────────────────────────┘

输出:
  - past: KV Cache
  - latent_embeds: [B, latent_steps, H]
```

### 9.4.5 为什么需要 latent_steps > 1？

单个隐藏状态可能不足以代表完整的推理轨迹。通过多次迭代：

```python
for step in range(latent_steps):
    # 每次迭代都用前一次的输出作为新的输入
    # 这相当于模拟一个"继续思考"的过程
    # 每个步骤的隐藏状态可能捕捉不同的推理方面
```

---

## 9.5 潜在对齐 (Latent Realignment)

### 9.5.1 为什么需要对齐？

不同模型的 embedding 空间不同：

```
Model A: "猫" → [0.2, 0.8, ...] (维度 4096)
Model B: "猫" → [0.5, 0.1, ...] (维度 3072)
```

直接传递会出错，需要对齐到统一空间。

### 9.5.2 _apply_latent_realignment

**位置**：`models.py:520-529`

```python
def _apply_latent_realignment(self, hidden: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    # 1. 获取对齐矩阵
    matrix, target_norm = self._ensure_latent_realign_matrix(model, hidden.device, self.args)

    # 2. 对齐矩阵相乘
    hidden_fp32 = hidden.to(torch.float32)
    aligned = torch.matmul(hidden_fp32, matrix)

    # 3. 归一化到目标尺度
    aligned_norm = aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    aligned = aligned * (target_norm / aligned_norm)

    return aligned.to(hidden.dtype)
```

### 9.5.3 对齐矩阵的计算

```python
# models.py:475-500
def _build_latent_realign_matrix(self, model, device, args) -> Tuple[torch.Tensor, torch.Tensor]:
    # 获取 embedding 层权重
    input_embeds = model.get_input_embeddings().weight   # [vocab, dim]
    output_embeds = model.get_output_embeddings().weight # [vocab, dim]

    # 论文公式 (8):
    # W_a = (W_out^T @ W_out + λI)^-1 @ W_out^T @ W_in
    gram = torch.matmul(output_embeds.T, output_embeds)  # [dim, dim]
    reg = 1e-5 * torch.eye(gram.shape[0], device=device)
    gram_reg = gram + reg

    rhs = torch.matmul(output_embeds.T, input_embeds)   # [dim, dim]
    realign_matrix = torch.linalg.solve(gram_reg, rhs)

    # 目标归一化尺度
    target_norm = input_embeds.norm(dim=1).mean()

    return realign_matrix, target_norm
```

### 9.5.4 对齐的物理意义

```
原始隐藏状态:
  h ∈ R^H  (来自 LLM)

对齐过程:
  1. h @ W_cross → h_aligned (投影到目标空间)
  2. 归一化尺度 → 匹配目标 embedding 分布

本质:
  找到一个线性变换 W，使得
  LLM 的 hidden space 和 embedding space 对齐
```

---

## 9.6 文本生成

### 9.6.1 generate_text_batch

```python
# models.py:531-686
@torch.no_grad()
def generate_text_batch(
    self,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    past_key_values: Optional[Tuple] = None,
    stop_regexes: Optional[List[str]] = None,
    return_metadata: bool = False,
) -> Tuple:
```

### 9.6.2 生成流程

```python
# 1. 处理 KV Cache
if past_key_values is not None:
    past_len = _past_length(past_key_values)
    cache_position = torch.arange(past_len, past_len + input_len, ...)

# 2. 调用模型生成
outputs = text_model.generate(**gen_kwargs)

# 3. 解码输出
generations = []
for idx, length in enumerate(prompt_lengths):
    generated_ids = sequences[idx, input_len:]
    text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
    generations.append(text)
```

### 9.6.3 停止条件

```python
# 支持多种停止条件
if stop_regexes:
    # 正则表达式匹配
    stop_checker = _StopOnRegex(...)

# 最大 token 数
if gen_len >= max_new_tokens:
    stop_reasons.append("max_tokens")

# EOS token
if generated_ids[-1] == eos_token_id:
    stop_reasons.append("eos")
```

---

## 9.7 vLLM 集成

### 9.7.1 为什么使用 vLLM？

vLLM 是一个高效的 LLM 推理引擎，支持：
- PagedAttention（内存优化）
- 连续批处理
- 张量并行

### 9.7.2 初始化

```python
if use_vllm and _HAS_VLLM:
    self.vllm_engine = LLM(
        model=model_name,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=gpu_util,
        enable_prefix_caching=True,
    )
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
```

### 9.7.3 vLLM 生成

```python
def vllm_generate_text_batch(self, prompts: List[str], *, ...):
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
    )
    outputs = self.vllm_engine.generate(prompts, sampling_params)
    generations = [out.outputs[0].text.strip() for out in outputs]
    return generations
```

---

## 9.8 设计哲学

### 9.8.1 统一接口

ModelWrapper 的设计目标是为不同 VLM 提供**统一接口**：

```python
# 无论底层是 Qwen-VL、InternVL 还是其他
wrapper = ModelWrapper(model_name, device)
response = wrapper.generate_text_batch(input_ids)
latents = wrapper.generate_latent_batch(input_ids, latent_steps=8)
```

### 9.8.2 适配器模式

通过检测模型类型，动态适配：

```python
def _get_text_model(self):
    if self.is_internvl:
        return self.model.language_model
    elif self.is_minicpm_v:
        return self.model.llm
    else:
        return self.model
```

---

## 9.9 本课小结

### 关键要点

1. **ModelWrapper**：统一封装不同 VLM 的加载和管理
2. **多模态检测**：通过 `model_type` 字符串匹配
3. **KV Cache 操作**：
   - `_cache_to_legacy_list`：统一格式
   - `_past_length`：获取长度
   - `_slice_past`：切片
4. **generate_latent_batch**：
   - 首次 forward 获取初始隐藏状态
   - latent_steps 次迭代生成潜在向量序列
   - 每次迭代用潜在向量作为输入继续 forward
5. **潜在对齐**：`_apply_latent_realignment` 将隐藏状态投影到统一空间
6. **vLLM 集成**：高效的 LLM 推理支持

### 思考题

1. 为什么需要 `latent_steps > 1`？只用最后一个隐藏状态不够吗？
2. `_apply_latent_realignment` 中的对齐矩阵是基于什么数学原理？如果两个模型的词表完全不相同，这个对齐还能工作吗？
3. ModelWrapper 为什么需要处理这么多特殊模型（InternVL、MiniCPMV 等）？这些模型有什么共同特点？

### 下一课预告

**第 10 课：Agent 角色定义**

- 四种角色：Planner、Critic、Refiner、Judger
- 提示词模板设计
- 层次化 vs 顺序式消息构建

---

*返回 [课程大纲](./课程大纲.md)*
