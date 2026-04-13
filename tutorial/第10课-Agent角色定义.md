# 第 10 课：Agent 角色定义

## 课程目标

- 理解多智能体系统中 Agent 的角色划分
- 掌握四种角色（Planner、Critic、Refiner、Judger）的职责
- 理解提示词模板的设计逻辑
- 掌握顺序式与层次化消息构建的区别

---

## 10.1 多 Agent 架构概述

### 10.1.1 为什么需要多 Agent？

单个 LLM 在复杂任务上可能表现不佳，多 Agent 系统通过**分工协作**提升效果：

| 问题 | 单 Agent | 多 Agent |
|------|----------|----------|
| 复杂推理 | 容易出错 | 分步验证 |
| 错误纠正 | 难以发现 | Critic 发现 |
| 任务泛化 | 单一能力 | 角色专业化 |

### 10.1.2 角色划分原则

Vision Wormhole 采用**四角色**分工：

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│Planner  │───▶│ Critic  │───▶│ Refiner │───▶│ Judger  │
│ (计划者) │    │ (批评者) │    │ (改进者) │    │ (评判者) │
└─────────┘    └─────────┘    └─────────┘    └─────────┘
     │              │              │              │
     ▼              ▼              ▼              ▼
  生成计划        评估计划        改进计划       最终裁决
```

---

## 10.2 四种角色详解

### 10.2.1 Planner（计划者）

**职责**：理解问题，制定解决计划

```python
# prompts.py:19-26
role == "planner"
user_prompt = f"""You are a Planner Agent. Given an input question,
design a clear, step-by-step plan for how to solve the question.

Question: {question}

Your outlined plan should be concise with a few bulletpoints for each step.
Do not produce the final answer.
Now output your plan to solve the question below:
"""
```

**特点**：
- 不直接给出答案
- 输出结构化的步骤计划
- 为 Critic 提供评估对象

**输出示例**：

```
Step 1: 理解问题要求 - 计算 15 + 27
Step 2: 进行加法运算 - 15 + 20 = 35, 35 + 7 = 42
Step 3: 验证结果 - 42
```

### 10.2.2 Critic（批评者）

**职责**：评估计划质量，提供改进建议

```python
# prompts.py:28-42
role == "critic"
user_prompt = f"""
Question: {question}

You are a Critic Agent to evaluate the correctness of the input plan
for the given question and provide helpful feedback for improving the plan.
The plan information is provided in latent KV representation format.

Format your response as follows:
Original Plan: [Copy the provided Planner Agent's plan here]
Feedback: [Your detailed feedback to improve the plan here]
"""
```

**特点**：
- 评估计划的**正确性**和**完整性**
- 提供**建设性反馈**
- 不直接改进，只指出问题

**输出示例**：

```
Original Plan: 步骤1... 步骤2...
Feedback: 计划缺少验证步骤，建议添加...
```

### 10.2.3 Refiner（改进者）

**职责**：基于反馈，生成改进后的计划

```python
# prompts.py:44-56
role == "refiner"
user_prompt = f"""
Question: {question}

You are a Refiner Agent to provide a refined step-by-step plan for solving
the given question.
You are provided with:
(1) latent-format information: a previous plan with feedback
(2) text-format information: the input question

Based on the input, write a refined and improved plan to solve the question.
Make sure your output plan is correct and concise.
"""
```

**特点**：
- 结合**原始计划 + Critic 反馈**
- 整合**文本问题 + 潜在信息**
- 输出改进后的完整计划

**输出示例**：

```
Step 1: 理解问题...
Step 2: 执行计算...
Step 3: 验证结果... (新增)
Step 4: 输出最终答案...
```

### 10.2.4 Judger（评判者）

**职责**：综合所有信息，输出最终答案

```python
# prompts.py:58-70 (GSM8K 示例)
role == "judger"
user_prompt = f"""
Target Question: {question}

You are a helpful assistant. You are provided with latent information
for reference and a target question to solve.

The latent information might contain irrelevant contents.
Ignore it if it is not helpful for solving the target question.

You must reason step-by-step to solve the provided Target Question
without outputting other irrelevant information.

At the end, output the final answer on a single line as: #### <number>.
"""
```

**特点**：
- 是**最终决策者**
- 接收 Planner/Critic/Refiner 的信息
- 负责**格式化输出最终答案**

**输出格式**：

| 任务类型 | 输出格式 |
|----------|----------|
| GSM8K | `#### <number>` |
| AIME | `\boxed{YOUR_FINAL_ANSWER}` |
| ARC/GPQA/MedQA | `\boxed{A/B/C/D}` |
| MBPP+/HumanEval+ | 代码块 |

---

## 10.3 提示词模板设计

### 10.3.1 系统提示词

```python
system_message = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
if 'qwen' not in args.model_name.lower():
    system_message = "You are a helpful assistant."
```

### 10.3.2 任务适配

提示词根据**任务类型**和**数据集**动态调整：

```python
if args.task == "gsm8k":
    # GSM8K 格式：#### <number>
    ...
elif args.task in ['aime2024', 'aime2025']:
    # AIME 格式：\boxed{YOUR_FINAL_ANSWER}
    ...
elif args.task in ["arc_easy", "arc_challenge", "gpqa", 'medqa']:
    # 选择题格式：\boxed{A/B/C/D}
    ...
elif args.task in ["mbppplus", "humanevalplus"]:
    # 编程题格式
    ...
```

### 10.3.3 角色适配

同一角色在不同任务中可能表现不同：

```python
# 层次化模式中，Planner 在 GSM8K 任务中的提示
if role == "planner" and args.task == "gsm8k":
    user_content = f"""
You are a math agent. Given the input question,
reason step-by-step and put the final answer on a single line as: #### <number>.

Input Question: {question}
"""
```

---

## 10.4 顺序式消息构建

### 10.4.1 build_agent_message_sequential_latent_mas

**位置**：`prompts.py:9`

```python
def build_agent_message_sequential_latent_mas(role, question, context="", method=None, args=None):
```

### 10.4.2 信息流

```
问题 → Planner → 计划
            ↓
        Critic → 评估 + 反馈
            ↓
        Refiner → 改进计划
            ↓
        Judger → 最终答案
```

### 10.4.3 各角色接收的信息

| 角色 | 接收的信息 | 来源 |
|------|-----------|------|
| **Planner** | 问题 | 直接输入 |
| **Critic** | 问题 + 计划 | Planner 输出（通过视觉通道） |
| **Refiner** | 问题 + 计划 + 反馈 | Planner + Critic（混合） |
| **Judger** | 问题 + 所有中间结果 | 所有 Agent |

---

## 10.5 层次化消息构建

### 10.5.1 build_agent_message_hierarchical_latent_mas

**位置**：`prompts.py:141`

```python
def build_agent_message_hierarchical_latent_mas(role, question, context="", method=None, args=None):
```

### 10.5.2 信息流

```
┌─────────────────────────────────────────────────────────┐
│                     问题输入                             │
└─────────────────────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
   ┌─────────┐      ┌─────────┐      ┌─────────┐
   │ Planner │      │ Critic  │      │ Refiner │
   │ (数学)  │      │ (科学)  │      │ (代码)  │
   └─────────┘      └─────────┘      └─────────┘
        │                │                │
        └────────────────┼────────────────┘
                         ▼
                  ┌─────────┐
                  │ Judger  │
                  │ (总结)  │
                  └─────────┘
```

### 10.5.3 角色专业化

层次化模式中，每个角色被赋予**专业化人设**：

| 角色 | 人设 | 说明 |
|------|------|------|
| **Planner** | 数学 Agent | 专注数学推理 |
| **Critic** | 科学 Agent | 批判性分析 |
| **Refiner** | 代码 Agent | 精确执行 |
| **Judger** | 总结 Agent | 综合判断 |

---

## 10.6 顺序式 vs 层次化对比

### 10.6.1 对比表

| 方面 | 顺序式 (Sequential) | 层次化 (Hierarchical) |
|------|---------------------|----------------------|
| **信息流** | 线性传递 | 并行汇总 |
| **角色设计** | 通用 | 专业化 |
| **通信模式** | A→B→C→D | A,B,C → D |
| **复杂度** | O(N) | O(1) 到 D |
| **适用场景** | 简单推理 | 复杂多角度分析 |

### 10.6.2 通信开销

```
顺序式:
  Planner → Critic → Refiner → Judger
  总通信次数 = N-1 = 3

层次化:
  Planner ─┐
  Critic ──┼──→ Judger
  Refiner ─┘
  总通信次数 = 3 (并行)
```

---

## 10.7 潜在信息传递

### 10.7.1 潜在信息的角色

在 Vision Wormhole 中，信息传递有两种形式：

| 形式 | 说明 | 使用场景 |
|------|------|----------|
| **文本传递** | 显式文本输出 | 通用信息 |
| **潜在传递** | 通过视觉通道注入 | 隐藏状态 |

### 10.7.2 Critic 接收潜在信息

```python
# prompts.py:33
"The plan information is provided in latent KV representation format."
```

这意味着 Critic **不是通过读取文本**来获取 Planner 的计划，而是通过**视觉通道注入**来接收潜在信息。

### 10.7.3 潜在信息的优势

| 优势 | 说明 |
|------|------|
| **信息完整** | 不经过文本量化 |
| **带宽高** | 连续向量直接传递 |
| **保密** | 潜在信息难以直接解读 |

---

## 10.8 实际执行流程

### 10.8.1 完整流程示例

```python
# 1. Planner 生成计划
planner_output = model.generate("问题", role="planner")
plan_latent = model.get_hidden_states()  # 获取潜在表示

# 2. Critic 评估（接收潜在信息）
critic_input = {
    "question": question,
    "latent_plan": plan_latent  # 通过视觉通道注入
}
critic_output = model.generate(critic_input, role="critic")

# 3. Refiner 改进
refiner_input = {
    "question": question,
    "latent_plan": plan_latent,
    "feedback": critic_output.text
}
refiner_output = model.generate(refiner_input, role="refiner")

# 4. Judger 最终裁决
judger_input = {
    "question": question,
    "latent_all": all_latents  # 所有 Agent 的潜在信息
}
final_answer = model.generate(judger_input, role="judger")
```

---

## 10.9 本课小结

### 关键要点

1. **四角色分工**：
   - Planner：制定计划
   - Critic：评估反馈
   - Refiner：改进计划
   - Judger：最终裁决
2. **提示词适配**：根据任务类型（GSM8K、AIME、ARC 等）调整格式
3. **两种消息构建**：
   - 顺序式：线性传递 A→B→C→D
   - 层次化：并行汇总 A,B,C → D
4. **潜在信息传递**：通过视觉通道而非文本传递隐藏状态

### 思考题

1. 为什么 Judger 需要说"The latent information might contain irrelevant contents. Ignore it if it is not helpful"？这反映了潜在传递的什么问题？
2. 如果 Refiner 收到的反馈与原始计划矛盾，应该如何处理？
3. 层次化模式中，如果 Planner、Critic、Refiner 同时出错，Judger 如何保证最终答案的正确性？

### 下一课预告

**第 11 课：多智能体方法对比**

- text_mas：标准文本通信
- latent_mas：纯潜在空间转移
- vision_latent_mas_codec_new：Vision Wormhole
- vision_latent_mas_ocr：OCR 基线

---

*返回 [课程大纲](./课程大纲.md)*
