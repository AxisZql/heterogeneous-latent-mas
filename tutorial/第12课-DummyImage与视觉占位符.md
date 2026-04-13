# 第 12 课：Dummy Image 与视觉占位符

## 课程目标

- 理解虚拟图像 (Dummy Image) 的设计原理
- 掌握 `_DummyImageSpec` 配置系统
- 理解视觉 token 提取机制
- 掌握不同 VLM 的处理差异

---

## 12.1 问题背景

### 12.1.1 视觉通道注入的挑战

Vision Wormhole 的核心是将信息通过**视觉通道**注入到接收模型。但这里有一个问题：

> **接收模型的视觉编码器需要真实的图像输入，我们如何"伪造"一个图像来承载注入的信息？**

### 12.1.2 解决方案：Dummy Image

**Dummy Image（虚拟图像）** 是一个纯色的空白图像：

```python
# methods/vision_latent_mas_codec_new.py:177-178
def _make_dummy_image(size: int = 224) -> Image.Image:
    return Image.new("RGB", (size, size), color=(255, 255, 255))
    # 白色图像，不携带任何有意义的信息
```

**关键洞察**：Dummy Image 的视觉 token 位置是**空的**，可以被我们的 `delta` 替换！

---

## 12.2 Dummy Image 的作用机制

### 12.2.1 注入原理

```
Dummy Image 的处理流程:

1. 创建白色图像
   ┌────────────────┐
   │                │
   │  (空白)        │
   │                │
   └────────────────┘

2. 通过视觉编码器提取 token
   dummy_tokens: [t_1, t_2, t_3, ..., t_256]

3. Decoder 生成 delta
   delta: [d_1, d_2, d_3, ..., d_256]

4. 注入
   injected = dummy_tokens + gate * delta

5. 输入到接收模型
   接收模型认为在看图像，实际接收了注入的信息
```

### 12.2.2 为什么有效？

VLM 的视觉编码器处理图像时：
1. 将图像分割成 patches
2. 每个 patch 转换为 embedding
3. 这些 embedding 进入 Transformer

Dummy Image 提供了一个**合法的图像结构**，但其内容是空白的。当我们替换其 token 后，接收模型不知道这是"假图像"，只会按照正常流程处理这些"增强的视觉 token"。

---

## 12.3 _DummyImageSpec 配置

### 12.3.1 类定义

**位置**：`methods/vision_latent_mas_codec_new.py:97-100`

```python
@dataclass
class _DummyImageSpec:
    count: int   # 虚拟图像数量
    size: int    # 虚拟图像尺寸（正方形边长）
```

### 12.3.2 配置解析

**位置**：`methods/vision_latent_mas_codec_new.py:103-151`

```python
def _resolve_dummy_image_specs(model_names: Sequence[str], args: argparse.Namespace) -> Dict[str, _DummyImageSpec]:
    n = len(model_names)

    # 默认配置
    default_count = max(1, _safe_int(getattr(args, "vision_codec_dummy_image_count", 1), 1))
    default_size = max(32, _safe_int(getattr(args, "vision_codec_dummy_image_size", 224), 224))
```

### 12.3.3 支持的配置方式

| 参数 | 说明 | 示例 |
|------|------|------|
| `--vision_codec_dummy_image_count` | 全局默认图像数量 | `1` |
| `--vision_codec_dummy_image_size` | 全局默认图像尺寸 | `224` |
| `--vision_codec_dummy_image_counts` | 每个模型的图像数量 | `"1,2,1"` |
| `--vision_codec_dummy_image_sizes` | 每个模型的图像尺寸 | `"224,336,224"` |
| `--vision_codec_dummy_image_spec_json` | JSON 格式完整配置 | `{"model_name": {"count": 2, "size": 336}}` |

### 12.3.4 配置示例

```bash
# 全局配置
--vision_codec_dummy_image_count=1
--vision_codec_dummy_image_size=224

# 每个模型不同配置
--vision_codec_dummy_image_counts=1,2,1
--vision_codec_dummy_image_sizes=224,336,224

# JSON 完整配置
--vision_codec_dummy_image_spec_json='{"Qwen/Qwen3-VL-2B": {"count": 1, "size": 224}}'
```

---

## 12.4 视觉 Token 提取

### 12.4.1 _extract_dummy_image_tokens 函数

**位置**：`methods/vision_latent_mas_codec_new.py:869`

```python
def _extract_dummy_image_tokens(
    wrapper: ModelWrapper,
    processor: Any,
    tokenizer: Any,
    special_ids: SpecialTokenIds,
    dummy_imgs: List[Image.Image],
) -> Optional[torch.Tensor]:
```

### 12.4.2 提取流程

```python
def _extract_dummy_image_tokens(...):
    # 1. 构建消息
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": _build_mm_user_content(num_images=n_imgs, text=" "),
        },
    ]

    # 2. 应用 chat template
    if hasattr(tokenizer, "apply_chat_template"):
        chat = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    else:
        chat = "You are a helpful assistant.\n\n[IMAGE]\n "

    # 3. 对于不同模型有特殊处理
    if _is_internvl_wrapper(wrapper):
        # InternVL 特殊处理
        enc = _internvl_prepare_multimodal_batch(...)
    elif _is_minicpm_wrapper(wrapper):
        # MiniCPM 特殊处理
        ...
```

### 12.4.3 Token 长度验证

**位置**：`methods/vision_latent_mas_codec_new.py:1222-1231`

```python
# 检查不同模型的 dummy token 长度是否一致
if int(getattr(self.args, "vision_codec_check_dummy_img_tokens", 0)) == 1 and len(token_lens) >= 2:
    lens_set = set(token_lens.values())
    if len(lens_set) != 1:
        msg = "Dummy image token counts do not match across models: ..."
        if int(getattr(args, "vision_codec_require_dummy_img_tokens_match", 0)) == 1:
            raise RuntimeError(msg)
        print(f"[codec-new][warn] {msg}")
```

**为什么需要一致？** 因为 `delta` 的维度 (`k_img`) 是固定的，如果不同模型的视觉 token 数量不同，注入会出问题。

---

## 12.5 不同 VLM 的处理差异

### 12.5.1 VLM 类型检测

```python
# methods/vision_latent_mas_codec_new.py:181-198
def _is_internvl_wrapper(wrapper: ModelWrapper) -> bool:
    if bool(getattr(wrapper, "is_internvl", False)):
        return True
    cfg = getattr(wrapper.model, "config", None)
    model_type = str(getattr(cfg, "model_type", "")).lower()
    cls_name = wrapper.model.__class__.__name__.lower()
    return ("internvl" in model_type) or ("internvl" in cls_name)

def _is_minicpm_wrapper(wrapper: ModelWrapper) -> bool:
    if bool(getattr(wrapper, "is_minicpm_v", False)):
        return True
    cfg = getattr(wrapper.model, "config", None)
    model_type = str(getattr(cfg, "model_type", "")).lower()
    cls_name = wrapper.model.__class__.__name__.lower()
    return ("minicpmv" in model_type) or ...
```

### 12.5.2 各模型的特殊处理

| 模型 | 特殊处理 | 原因 |
|------|----------|------|
| **InternVL** | `_internvl_prepare_multimodal_batch` | 特殊的视觉编码器接口 |
| **MiniCPM** | MiniCPMV 特定的 chat template | 不同的图像占位符格式 |
| **Qwen-VL** | 标准处理 | 通用 VLM 设计 |
| **Gemma3-VL** | 优先使用 fast processor | 性能优化 |
| **SmolVLM** | 特殊消息格式 | 小型化设计 |

### 12.5.3 Text Backbone 获取

```python
# methods/vision_latent_mas_codec_new.py:201-206
def _get_text_backbone(wrapper: ModelWrapper):
    """获取不同模型的文本解码器"""
    if _is_internvl_wrapper(wrapper) and hasattr(wrapper.model, "language_model"):
        return wrapper.model.language_model
    if _is_minicpm_wrapper(wrapper) and hasattr(wrapper.model, "llm"):
        return wrapper.model.llm
    return wrapper.model
```

---

## 12.6 Dummy Image 在系统中的角色

### 12.6.1 完整生命周期

```
1. 初始化阶段
   ┌─────────────────────────────────────────┐
   │ VisionLatentMASMethodCODECNew.__init__   │
   │                                          │
   │ for each model:                          │
   │   dummy_imgs = [_make_dummy_image(size)] │
   │   dummy_tokens = extract(dummy_imgs)    │
   │   self.dummy_img_tokens[i] = dummy_tokens│
   └─────────────────────────────────────────┘

2. 运行阶段（Sender → Receiver）
   ┌─────────────────────────────────────────┐
   │ Sender:                                 │
   │   h = model.get_hidden_states()         │
   │   U = encoder(h)                        │
   └─────────────────────────────────────────┘
                     ↓ U
   ┌─────────────────────────────────────────┐
   │ Receiver:                               │
   │   delta, gate = decoder(U)              │
   │   injected = dummy_tokens + gate * delta │
   │   model输入 = [text_tokens + injected]   │
   └─────────────────────────────────────────┘
```

### 12.6.2 优势

| 优势 | 说明 |
|------|------|
| **无需真实图像** | 纯白图像即可 |
| **可预测的 token 槽位** | 固定位置可供注入 |
| **模型无关** | 各模型都支持 |
| **可调试** | 可以检查 dummy token 数量 |

### 12.6.3 局限性

| 局限 | 说明 |
|------|------|
| **Token 数量匹配** | 需要不同模型的 dummy token 数量一致 |
| **尺寸敏感性** | 某些模型对图像尺寸敏感 |
| **注入位置固定** | 只能在 dummy token 位置注入 |

---

## 12.7 实际使用示例

### 12.7.1 创建 Dummy Image

```python
from PIL import Image

# 创建 224x224 的白色图像
dummy = Image.new("RGB", (224, 224), color=(255, 255, 255))
```

### 12.7.2 配置多个模型的不同 Dummy Image

```bash
# Qwen 使用 1 个 224x224 图像
# Gemma 使用 2 个 336x336 图像
--vision_codec_dummy_image_counts=1,2
--vision_codec_dummy_image_sizes=224,336
```

### 12.7.3 检查 Token 数量

```bash
# 启用检查
--vision_codec_check_dummy_img_tokens=1

# 如果不匹配则报错
--vision_codec_require_dummy_img_tokens_match=1
```

---

## 12.8 本课小结

### 关键要点

1. **Dummy Image** 是纯色空白图像，用于占据视觉 token 槽位
2. **注入原理**：Dummy token + gate * delta → 注入信息
3. **配置系统**：`count` 和 `size` 可分别配置
4. **Token 提取**：通过 processor 将图像转为 token
5. **模型适配**：不同 VLM 需要不同的处理方式
6. **Token 匹配**：不同模型的 dummy token 数量需要一致

### 思考题

1. 如果 dummy image 不是白色的，而是有颜色的（如黑色），会影响注入效果吗？为什么？
2. 为什么 dummy token 数量需要匹配？能否设计一个自适应方案来处理不同长度的 token？
3. 如果一个模型支持动态分辨率（不同图像尺寸产生不同数量的 token），dummy image 应该如何设计？

### 下一课预告

**第 13 课：教师-学生蒸馏目标**

- MSE 损失
- KL 损失
- CKA 损失
- 统计损失

---

*返回 [课程大纲](./课程大纲.md)*
