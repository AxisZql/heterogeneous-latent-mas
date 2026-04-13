# 第 16 课：Anchor Text 与数据预处理

## 课程目标

- 理解锚点文本在编解码器训练中的作用
- 掌握锚点文本的构建方法
- 理解数据预处理脚本的使用
- 掌握 JSONL 格式的数据处理

---

## 16.1 锚点文本的作用

### 16.1.1 为什么需要锚点文本？

锚点文本 (Anchor Text) 是用于**跨模型对齐**的特殊文本：

```
锚点文本的作用：

Model A 的锚点表示: U_A("1+1=2") ──┐
                                          ├── 对齐到同一空间
Model B 的锚点表示: U_B("1+1=2") ──┘

锚点文本覆盖各种语义类型：
- 数学推理: "1+1=2", "求 x: 2x+3=7"
- 代码生成: "def quick_sort():", "for i in range"
- 常识推理: "猫是动物", "水往低处流"
- 批判分析: "这个论证的问题在于..."
```

### 16.1.2 锚点文本的要求

| 要求 | 说明 |
|------|------|
| **语义覆盖** | 覆盖各种任务类型和领域 |
| **长度适中** | 不太长也不太短，适合模型处理 |
| **无特殊格式** | 避免复杂的 markdown、LaTeX |
| **独立同分布** | 与训练数据的分布相似 |

### 16.1.3 默认锚点文本

**位置**：`train_vision_latent_mas_codec_new.py:2291-2303`

```python
default = [
    "Summarize the following in one sentence: The mitochondrion is the powerhouse of the cell.",
    "Give a step-by-step plan to solve a two-digit multiplication problem.",
    "Explain what 'gradient descent' is in simple terms.",
    "Write a short critique of an argument that confuses correlation with causation.",
    "List three potential failure modes in multi-agent reasoning systems.",
    "State the Pythagorean theorem and one practical use-case.",
    "You are given: A=17, B=5. Compute A*B and show your reasoning.",
    "Define Bayes' rule and describe one intuition for it.",
    "Provide a counterexample to the claim: 'All prime numbers are odd.'",
    "Explain what an embedding is in machine learning.",
]
```

---

## 16.2 锚点文本的构建

### 16.2.1 构建策略

| 策略 | 方法 | 优缺点 |
|------|------|--------|
| **手动编写** | 人工编写代表性文本 | 质量高，但耗时 |
| **数据集采样** | 从训练集采样 | 代表性强，但可能泄露 |
| **合成生成** | 使用 LLM 生成 | 可扩展，但可能同质化 |
| **混合策略** | 结合多种方法 | 最佳实践 |

### 16.2.2 数据集来源

项目中支持从多个数据集构建锚点文本：

| 数据集 | 类型 | 用途 |
|--------|------|------|
| `salesforce/cos_e` | Chain-of-Thought 推理 | 数学、常识推理 |
| `nvidia/opencodereasoning` | 代码推理 | 编程任务 |
| `openai/prm800k` | 过程监督 | 步骤级推理 |

---

## 16.3 数据集预处理脚本

### 16.3.1 脚本位置

**位置**：`scripts/preprocess_dataset.py`

### 16.3.2 使用方法

```bash
python scripts/preprocess_dataset.py \
  --datasets "salesforce/cos_e,nvidia/opencodereasoning,openai/prm800k" \
  --splits "validation,split_0,train" \
  --limit_per_dataset 1000 \
  --shuffle \
  --format jsonl \
  --out data/vision_codec_anchor_text/mixed_custom.jsonl
```

### 16.3.3 参数说明

| 参数 | 说明 |
|------|------|
| `--datasets` | 数据集名称（逗号分隔） |
| `--splits` | 数据集分割（逗号分隔） |
| `--limit_per_dataset` | 每个数据集的最大样本数 |
| `--shuffle` | 是否打乱数据 |
| `--format` | 输出格式（jsonl） |
| `--out` | 输出文件路径 |

---

## 16.4 预处理函数解析

### 16.4.1 PRM800k 格式处理

**位置**：`scripts/preprocess_dataset.py:71-100`

```python
def _build_text_from_prm800k(item: Dict[str, Any]) -> Optional[str]:
    # 提取问题
    question = item.get("question", {})
    problem = question.get("problem") or question.get("question")

    # 提取步骤
    label = item.get("label", {})
    steps_raw = label.get("steps", None)

    # 构建步骤文本
    steps_out: List[str] = []
    for step in steps_raw:
        # 提取人类完成的步骤
        human = step.get("human_completion", None)
        if human:
            steps_out.append(human.strip())
            continue

        # 或使用选中的完成
        comps = step.get("completions", [])
        if comps:
            chosen = step.get("chosen_completion", 0)
            comp = comps[chosen]
            steps_out.append(comp.get("text", "").strip())

    # 格式化为带编号的步骤
    steps = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps_out))
    return f"{problem}\n\n{steps}"
```

### 16.4.2 CoSE 数据集处理

```python
# 提取推理步骤
"cos_e" 数据集包含:
- question: 问题
- reasoning: 推理步骤
- answer: 答案
```

### 16.4.3 OpenCodeReasoning 数据集处理

```python
# 代码推理数据
"nvidia/opencodereasoning" 数据集包含:
- problem: 代码问题描述
- solution: 解决方案
- explanation: 解释
```

---

## 16.5 锚点文本加载

### 16.5.1 加载函数

**位置**：`train_vision_latent_mas_codec_new.py:2291-2330`

```python
def _load_anchor_texts(path: str) -> List[str]:
    # 如果路径为空或不存在，返回默认文本
    if not path or not os.path.exists(path):
        return default

    # 加载 JSONL 格式
    if path.endswith(".jsonl"):
        out: List[str] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line.strip())
                if isinstance(obj, str):
                    out.append(obj)
                elif isinstance(obj, dict):
                    # 尝试多个 key
                    for k in ("text", "prompt", "message"):
                        if k in obj:
                            out.append(obj[k])
                            break
        return out

    return default
```

### 16.5.2 支持的格式

| 格式 | 示例 | 加载方式 |
|------|------|----------|
| JSONL (字符串) | `"这是一条锚点文本"` | 直接读取为字符串 |
| JSONL (对象) | `{"text": "锚点文本"}` | 读取 text 字段 |
| JSONL (对象) | `{"prompt": "锚点文本"}` | 读取 prompt 字段 |

---

## 16.6 JSONL 格式说明

### 16.6.1 什么是 JSONL？

**JSON Lines (JSONL)** 是每行一个 JSON 的格式：

```
{"text": "锚点文本 1"}
{"text": "锚点文本 2"}
{"text": "锚点文本 3"}
```

### 16.6.2 与 JSON 数组的区别

```json
// JSON 数组
["文本1", "文本2", "文本3"]

// JSONL (每行一个 JSON)
{"text": "文本1"}
{"text": "文本2"}
{"text": "文本3"}
```

### 16.6.3 JSONL 的优势

| 优势 | 说明 |
|------|------|
| **流式处理** | 可以边读边处理，无需全部加载到内存 |
| **易于追加** | 直接 append 新行即可 |
| **容错性** | 一行损坏不影响其他行 |
| **大文件友好** | 适合处理大规模数据 |

---

## 16.7 完整流程示例

### 16.7.1 Step 1: 创建锚点文本目录

```bash
mkdir -p data/vision_codec_anchor_text
```

### 16.7.2 Step 2: 预处理数据集

```bash
python scripts/preprocess_dataset.py \
  --datasets "salesforce/cos_e,nvidia/opencodereasoning,openai/prm800k" \
  --splits "validation,train" \
  --limit_per_dataset 500 \
  --shuffle \
  --format jsonl \
  --out data/vision_codec_anchor_text/mixed.jsonl
```

### 16.7.3 Step 3: 训练时指定锚点文本

```bash
python train_vision_latent_mas_codec_new.py \
  --model_name "Qwen/Qwen3-VL-2B-Thinking" \
  --vision_codec_path checkpoints/codec_qwen3vl2b.pt \
  --vision_codec_anchor_texts_path data/vision_codec_anchor_text/mixed.jsonl \
  --vision_codec_align_max_anchors 300 \
  ...
```

### 16.7.4 Step 4: 合并时使用

```bash
python merge_vision_codec_checkpoints.py \
  --codec_paths checkpoints/codec_qwen3vl2b.pt checkpoints/codec_lfm25vl16b.pt \
  --vision_codec_anchor_texts_path data/vision_codec_anchor_text/mixed.jsonl \
  ...
```

---

## 16.8 锚点文本数量与质量

### 16.8.1 数量选择

| 场景 | 建议数量 |
|------|----------|
| 快速实验 | 10-50 |
| 标准训练 | 100-500 |
| 高质量对齐 | 500-1000+ |

### 16.8.2 质量评估

检查锚点文本的质量：

```python
# 1. 长度分布
texts = _load_anchor_texts("path/to/anchor.jsonl")
lengths = [len(t) for t in texts]
print(f"平均长度: {sum(lengths)/len(lengths):.1f}")
print(f"最短: {min(lengths)}, 最长: {max(lengths)}")

# 2. 语义多样性（可通过嵌入的方差估计）
# 如果方差很小，说明文本同质化严重
```

### 16.8.3 质量改进

如果锚点文本质量不够好：

1. **增加多样性**：从更多数据集采样
2. **过滤短文本**：排除太短的文本
3. **去重**：移除重复或相似的文本
4. **手动审核**：检查并修正问题文本

---

## 16.9 本课小结

### 关键要点

1. **锚点文本作用**：作为跨模型对齐的桥梁
2. **构建策略**：手动编写、数据集采样、合成生成、混合策略
3. **数据集支持**：CoSE、OpenCodeReasoning、PRM800k
4. **预处理脚本**：`scripts/preprocess_dataset.py`
5. **JSONL 格式**：每行一个 JSON，适合流式处理
6. **加载函数**：`_load_anchor_texts()` 支持多种格式

### 思考题

1. 如果锚点文本只包含数学推理，对齐后的编解码器在代码生成任务上表现会如何？
2. 为什么 JSONL 格式比 JSON 数组更适合大数据集处理？
3. 如何判断锚点文本的数量是否足够？有没有一个有效的估计方法？

### 下一课预告

**第 17 课：数据集与评测指标**

- 支持的数据集：GSM8K、GPQA、ARC、AIME 等
- 数据加载器实现
- 答案提取与规范化
- 评测指标计算

---

*返回 [课程大纲](./课程大纲.md)*
