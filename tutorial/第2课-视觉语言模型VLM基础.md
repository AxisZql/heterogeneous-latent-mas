# 第 2 课：视觉-语言模型 (VLM) 基础

## 课程目标

- 理解 VLM 的基本架构：视觉编码器 + 语言解码器
- 掌握常见的 VLM 模型系列及特点
- 了解多模态推理轨迹的表示方式
- 理解项目中如何检测和适配不同的 VLM

---

## 2.1 VLM 的基本架构

**视觉-语言模型 (Vision-Language Model, VLM)** 是一类能够同时理解和生成文本与图像的深度学习模型。与纯文本 LLM 不同，VLM 需要处理**两种截然不同的模态**：视觉（图像）和语言（文本）。

### 2.1.1 核心组件

典型的 VLM 由三个主要组件构成：

```
┌─────────────────────────────────────────────────────────┐
│                    Vision-Language Model                 │
│                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌────────────┐ │
│  │   Vision    │───▶│   Bridge    │───▶│   Language │ │
│  │   Encoder    │    │  (Projector)│    │   Decoder  │ │
│  └─────────────┘    └─────────────┘    └────────────┘ │
│       │                                     │           │
│       ▼                                     ▼           │
│  [Image Patch                            [Text Token   │ │
│   Features]                               Output]     │
└─────────────────────────────────────────────────────────┘
```

| 组件 | 功能 | 示例技术 |
|------|------|----------|
| **视觉编码器 (Vision Encoder)** | 将图像转换为特征向量 | ViT, CLIP, SigLIP |
| **桥接器 (Bridge/Projector)** | 将视觉特征映射到语言模型空间 | MLP, Cross-Attention |
| **语言解码器 (Language Decoder)** | 生成文本输出 | LLaMA, Qwen, Gemma |

### 2.1.2 视觉编码器

视觉编码器负责将图像转换为 token 序列。典型流程：

```
图像 (H × W × 3)
    │
    ▼
[Patchification]  切分为 P × P 的 patches
    │
    ▼
线性投影 + 位置编码
    │
    ▼
Transformer Encoder (ViT)
    │
    ▼
视觉 Token 序列 [B, N, D_vision]
```

### 2.1.3 桥接器 (Projector)

桥接器是 VLM 的关键组件，它将视觉编码器的输出（通常是 CLIP/SigLIP 特征）映射到语言模型的 embedding 空间。

**常见类型**：

| 类型 | 结构 | 示例 |
|------|------|------|
| **MLP 桥接器** | 2-3 层 MLP | LLaVA, Qwen-VL |
| **Cross-Attention** | 交叉注意力层 | Flamingo, IDEFICS |
| **Perceiver Resampler** | 可学习的 Query + 交叉注意力 | GPT-4V, LFM |

### 2.1.4 语言解码器

语言解码器是一个标准的因果语言模型 (Causal LM)，负责生成文本：

- **输入**：文本 token 序列 + 视觉 token 序列（注入）
- **输出**：下一个 token 的概率分布
- **注意力**：因果注意力（只看前面的 token）

---

## 2.2 常见 VLM 模型系列

本项目支持多种 VLM 模型，以下是主要的模型系列：

### 2.2.1 Qwen-VL 系列

| 模型 | 特点 | 代码检测 |
|------|------|----------|
| Qwen2-VL | 阿里开源，支持多图、灵活的帧率 | `"qwen2vl"` in model_type |
| Qwen3-VL | 最新版本，支持长上下文 | `"qwen3vl"` in model_type |
| Qwen3-VL-2B-Thinking | 小型化思维模型 | `"qwen"` + `"vl"` |

**Qwen-VL 的架构特点**：
- 视觉编码器：ViT + SigLIP
- 桥接器：MLP
- 支持图像、视频、多语言

### 2.2.2 Gemma3-VL 系列

| 模型 | 特点 | 代码检测 |
|------|------|----------|
| Gemma3-VL | Google 开源，Gemini 技术下放 | `"gemma3"` in model_type |
| Gemma3-2B/7B-VL | 不同规模的视觉版本 | `"gemma"` + `"vl"` |

### 2.2.3 LFM2.5-VL 系列

| 模型 | 特点 | 代码检测 |
|------|------|----------|
| LFM2.5-VL-1.6B | Liquid AI 的液态神经网络 | `"lfm2_vl"` in model_type |
| LFM2.5-VL | 高效的视觉理解 | `"lfm2vl"` in class_name |

**LFM 的特殊架构**：
- 使用液态神经网络 (Liquid Neural Networks)
- 状态空间模型 (SSM) 而非传统 Transformer

### 2.2.4 InternVL 系列

| 模型 | 特点 | 代码检测 |
|------|------|----------|
| InternVL2 | 上海 AI Lab 开源 | `"internvl"` in model_type |
| InternVL3 | 最新的第三代 | `"internvl"` in class_name |

**InternVL 的特点**：
- 视觉编码器：InternViT (6B)
- 支持多种视觉任务
- 强大的文档理解能力

### 2.2.5 MiniCPMV 系列

| 模型 | 特点 | 代码检测 |
|------|------|----------|
| MiniCPM-V | 晓庄科技开源，小而强 | `"minicpmv"` in model_type |
| MiniCPM-V 2.x | 性能大幅提升 | `"minicpm"` + `"vl"` |

### 2.2.6 SmolVLM 系列

| 模型 | 特点 | 代码检测 |
|------|------|----------|
| SmolVLM | Hugging Face 小型 VLM | `"smolvlm"` in model_type |
| SmolVLM-100M/500M | 极致轻量级 | `"smolvlm"` in class_name |

### 2.2.7 其他支持模型

| 模型系列 | 检测关键词 | 备注 |
|----------|-----------|------|
| PaliGemma | `"paligemma"` | Google 的 PaLI 延续 |
| LLaVA | `"llava"` | 早期的开源 VLM |
| IDEFICS | `"idefics"` | Hugging Face 多模态 |
| mLlama | `"mllama"` | Meta 的多模态 LLaMA |

---

## 2.3 项目中的 VLM 检测机制

项目的 `models.py` 中实现了完善的 VLM 检测逻辑：

### 2.3.1 多模态模型检测

```python
# models.py:110-133
def _is_multimodal_config(cfg) -> bool:
    model_type = str(getattr(cfg, "model_type", "")).lower()
    class_name = cfg.__class__.__name__.lower()

    # Qwen-VL 检测
    if any(key in model_type for key in ("qwen", "gemma3")) and \
       any(key in model_type for key in ("vl", "_vl")):
        return True

    # LFM2-VL 检测
    if "lfm2_vl" in model_type or "lfm2vl" in class_name:
        return True

    # SmolVLM 检测
    if "smolvlm" in model_type or "smolvlm" in class_name:
        return True

    # InternVL 检测
    if "internvl" in model_type or "internvl" in class_name:
        return True

    # MiniCPMV 检测
    if "minicpmv" in model_type or "minicpmv" in class_name:
        return True
    # ...
```

### 2.3.2 特定模型检测

```python
# models.py:136-159
def _is_internvl_config(cfg) -> bool: ...
def _is_smolvlm_config(cfg) -> bool: ...
def _is_lfm2_vl_config(cfg) -> bool: ...
def _is_minicpm_v_config(cfg) -> bool: ...
```

### 2.3.3 ModelWrapper 中的模型类型标记

```python
# models.py:279-297
class ModelWrapper:
    def __init__(self, model_name: str, device: torch.device, ...):
        # 检测多模态
        is_multimodal = _is_multimodal_config(cfg)

        # 标记特定模型类型
        self.is_internvl = _is_internvl_config(cfg)
        self.is_smolvlm = _is_smolvlm_config(cfg)
        self.is_lfm2_vl = _is_lfm2_vl_config(cfg)
        self.is_minicpm_v = _is_minicpm_v_config(cfg)
```

这些标记在后续的模型操作中非常重要，用于区分处理不同模型的特殊需求。

---

## 2.4 多模态推理轨迹的表示

### 2.4.1 什么是推理轨迹 (Reasoning Trace)？

**推理轨迹**是智能体在解决问题过程中产生的中间推理步骤。例如：

```
问题：计算 15 + 27 = ?

推理轨迹：
Step 1: 我需要计算 15 + 27
Step 2: 15 + 20 = 35
Step 3: 35 + 7 = 42
Step 4: 答案是 42
```

在 LLM 中，这个推理轨迹被编码为**隐藏状态序列**：

$$h = [h_1, h_2, h_3, ..., h_L]$$

其中每个 $h_i \in \mathbb{R}^d$ 是模型在第 $i$ 步的隐藏状态。

### 2.4.2 多模态推理轨迹的组成

VLM 的推理轨迹可能包含：

| 组成部分 | 描述 | 维度 |
|----------|------|------|
| **文本隐藏状态** | 文本 token 对应的隐藏状态 | $[L_{text}, d]$ |
| **视觉隐藏状态** | 图像 token 对应的隐藏状态 | $[L_{image}, d]$ |
| **交叉模态状态** | 视觉-语言交互产生的状态 | varies |

### 2.4.3 项目中的潜在向量生成

项目中通过 `generate_latent_batch` 方法生成潜在向量：

```python
# models.py:696-...
def generate_latent_batch(
    self,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    *,
    latent_steps: int,  # 生成多少步潜在向量
    past_key_values: Optional[Tuple] = None,
    return_latent_embeds: bool = False,
    ...
) -> Tuple:
```

**核心逻辑**：

1. 将输入转换为 `inputs_embeds`
2. 多次执行 `model.forward()` 获取隐藏状态
3. 对最后的隐藏状态应用 `latent_realignment`
4. 返回潜在向量序列

---

## 2.5 VLM 的特殊处理

### 2.5.1 不同 VLM 的差异

不同 VLM 系列在以下方面存在差异：

| 方面 | Qwen-VL | InternVL | MiniCPMV | LFM |
|------|---------|----------|----------|-----|
| **视觉编码器** | SigLIP | InternViT | Vit | SigLIP |
| **桥接器** | MLP | MLP | MLP | Perceiver |
| **文本模型** | Qwen2 | Qwen2/LLaMA | Qwen2 | 定制 SSM |
| **图像处理** | 动态分辨率 | 动态分辨率 | 固定 | 固定 |

### 2.5.2 特殊模型的适配代码

项目中针对不同模型有专门的适配：

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

```python
# models.py:360-362
if self.is_internvl or self.is_minicpm_v:
    model_cls = AutoModelForCausalLM  # 这些模型需要特殊处理
else:
    model_cls = AutoModelForImageTextToText if is_multimodal else AutoModelForCausalLM
```

### 2.5.3 Tokenizer 的特殊处理

```python
# models.py:393-405
def render_chat(self, messages: List[Dict], add_generation_prompt: bool = True) -> str:
    if self.is_smolvlm:
        # SmolVLM 需要特殊的消息格式
        norm_messages: List[Dict[str, Any]] = []
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, list):
                norm_content = content
            else:
                norm_content = [{"type": "text", "text": str(content)}]
            norm_messages.append({"role": role, "content": norm_content})
        messages = norm_messages
```

---

## 2.6 视觉 Token 的工作机制

### 2.6.1 图像到 Token 的转换

```
原始图像 (224 × 224 × 3)
        │
        ▼
[Patchification]  16×16 patches → 14×14 = 196 patches
        │
        ▼
[Vision Encoder]  ViT/SigLIP → 196 个 embedding
        │
        ▼
[Projector]       MLP → 映射到 LLM embedding 空间
        │
        ▼
视觉 Token 序列 [1, N_visual, D_llm]
```

### 2.6.2 视觉 Token 的注入位置

在 VLM 中，视觉 token 通常被注入到文本 token 序列的**特定位置**：

```
[Text Tokens]  [SOS] 我 想 计 算 [VISION] 15 + 27 = ?
                  ↑
            视觉 token 插入位置
```

### 2.6.3 项目中的 Dummy Image

项目使用 **Dummy Image（虚拟图像）** 作为视觉通道注入的占位符：

```python
# methods/vision_latent_mas_codec_new.py:177-178
def _make_dummy_image(size: int = 224) -> Image.Image:
    return Image.new("RGB", (size, size), color=(255, 255, 255))
```

这允许在没有真实图像的情况下，通过虚拟图像注入潜在表示。

---

## 2.7 本课小结

### 关键要点

1. **VLM 架构**：视觉编码器 + 桥接器 + 语言解码器
2. **桥接器类型**：MLP、Cross-Attention、Perceiver Resampler
3. **支持模型**：Qwen-VL、Gemma3-VL、LFM2.5-VL、InternVL、MiniCPMV、SmolVLM 等
4. **检测机制**：通过 `model_type` 和 `class_name` 字符串匹配
5. **推理轨迹**：隐藏状态序列 $[h_1, h_2, ..., h_L]$
6. **潜在向量生成**：通过 `generate_latent_batch` 方法

### 思考题

1. 为什么不同 VLM 需要不同的桥接器设计？这与它们的训练方式有什么关系？
2. 如果要在新的 VLM 上应用 Vision Wormhole，需要修改哪些代码？
3. Dummy Image 的设计有什么优势？有什么潜在问题？

### 下一课预告

**第 3 课：潜在空间通信 (Latent Space Transfer)**

- 潜在空间 vs 文本空间的对比
- 连续表示 vs 离散表示
- 现有方法的局限性分析
- Hub-and-Spoke 拓扑的数学基础

---

*返回 [课程大纲](./课程大纲.md)*
