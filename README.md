# LLM Pro - 大模型课程作业

本仓库为大模型（LLM）课程的全部作业代码与辅导资料。
a
---

## 目录

- [1. 大模型基础](#1-大模型基础)
  - [1.1 API编程课后作业](#11-api编程课后作业)
  - [1.2 行业分类数据打标](#12-行业分类数据打标)
  - [1.3 记忆作业](#13-记忆作业)
  - [1.4 ReAct作业](#14-react作业)
- [2. 私有化部署](#2-私有化部署)
  - [2.1 Ollama本地部署](#21-ollama本地部署)
- [辅导资料](#辅导资料)

---

## 1. 大模型基础

### 1.1 API编程课后作业

> 目录：[作业/1-大模型基础/api编程课后作业](作业/1-大模型基础/api编程课后作业)

**作业要求：**

1. 多轮对话的上下文高效管理（3种方式 + 代码演示）
2. 发票内容解析（多模态大模型提取 + 校验）

**内容文件：**

| 文件 | 说明 |
|------|------|
| [1.多轮对话-滑动窗口.ipynb](作业/1-大模型基础/api编程课后作业/1.多轮对话-滑动窗口.ipynb) | 多轮对话 - 滑动窗口法 |
| [2.多轮对话-摘要压缩法.ipynb](作业/1-大模型基础/api编程课后作业/2.多轮对话-摘要压缩法.ipynb) | 多轮对话 - 摘要压缩法 |
| [多轮对话-Token截断法.ipynb](作业/1-大模型基础/api编程课后作业/多轮对话-Token截断法%20.ipynb) | 多轮对话 - Token截断法 |
| [发票解析.ipynb](作业/1-大模型基础/api编程课后作业/发票解析.ipynb) | 发票内容提取与校验 |
| [发票解析-openai.ipynb](作业/1-大模型基础/api编程课后作业/发票解析-openai.ipynb) | 发票解析 - OpenAI版本 |

---

### 1.2 行业分类数据打标

> 目录：[作业/1-大模型基础/行业分类数据打标](作业/1-大模型基础/行业分类数据打标)

**作业要求：**

1. 行业分类标准制定（20个分类）
2. 评测集制作（20分类 x 10条 = 200条）
3. 数据打标Prompt制作
4. 模型打标正确率 >= 92%
5. 训练数据集制作（20分类 x 2000条 = 40000条）
6. 论文阅读与代码复现
7. 模型评测（MMLU数据集）

**内容文件：**

| 文件 | 说明 |
|------|------|
| [classifier.py](作业/1-大模型基础/行业分类数据打标/行业分类数据打标/classifier.py) | 行业分类器 v1 |
| [classifier_v2.py](作业/1-大模型基础/行业分类数据打标/行业分类数据打标/classifier_v2.py) | 行业分类器 v2 |
| [classifier_v3.py](作业/1-大模型基础/行业分类数据打标/行业分类数据打标/classifier_v3.py) | 行业分类器 v3 |
| [classifier_v4.py](作业/1-大模型基础/行业分类数据打标/行业分类数据打标/classifier_v4.py) | 行业分类器 v4 |
| [classifier_v5.py](作业/1-大模型基础/行业分类数据打标/行业分类数据打标/classifier_v5.py) | 行业分类器 v5 |

**论文复现：**

| 论文 | 复现代码 |
|------|----------|
| Language Models are Few-Shot Learners (GPT-3) | [代码](作业/1-大模型基础/行业分类数据打标/论文复现/Language%20Models%20are%20Few-Shot%20Learners.ipynb) |
| Chain-of-Thought Prompting Elicits Reasoning in Large Language Models | [代码](作业/1-大模型基础/行业分类数据打标/论文复现/Chain-of-Thought.ipynb) |
| Large Language Models are Zero-Shot Reasoners | [代码](作业/1-大模型基础/行业分类数据打标/论文复现/Large%20Language%20Models%20are%20Zero-Shot%20Reasoners.ipynb) |
| 论文总结 | [总结文档](作业/1-大模型基础/行业分类数据打标/论文复现/论文总结.md) |

**模型评测（MMLU）：**

| 文件 | 说明 |
|------|------|
| [模型测试v1.0.py](作业/1-大模型基础/行业分类数据打标/模型评测/模型测试v1.0.py) | 模型评测 v1.0 |
| [模型测试v2.0.py](作业/1-大模型基础/行业分类数据打标/模型评测/模型测试v2.0.py) | 模型评测 v2.0 |
| [评测集excel.py](作业/1-大模型基础/行业分类数据打标/模型评测/评测集excel.py) | 评测集数据处理 |

**V2.0 评测结果：**

| Model | zero-shot | zero-shot-cot | few-shot | few-shot-cot |
|:------|:----------|:--------------|:---------|:-------------|
| mimo-v2-pro | 100.00% | 100.00% | 100.00% | 100.00% |
| deepseek-chat | 95.00% | 100.00% | 95.00% | 100.00% |
| openai/gpt-5-nano | 95.00% | 98.33% | 95.00% | 96.67% |

---

### 1.3 记忆作业

> 目录：[作业/1-大模型基础/记忆作业](作业/1-大模型基础/记忆作业)

**作业要求：** 将持久化存储 + 消息裁剪 + 会话压缩串起来执行，封装成通用聊天机器人。

**内容文件：**

| 文件 | 说明 |
|------|------|
| [chatbot.py](作业/1-大模型基础/记忆作业/chatbot.py) | 聊天机器人主模块 |
| [store.py](作业/1-大模型基础/记忆作业/store.py) | 持久化存储模块 |
| [context.py](作业/1-大模型基础/记忆作业/context.py) | 上下文裁剪模块 |
| [compressor.py](作业/1-大模型基础/记忆作业/compressor.py) | 会话压缩模块 |
| [models.py](作业/1-大模型基础/记忆作业/models.py) | 数据模型定义 |
| [main.py](作业/1-大模型基础/记忆作业/main.py) | 入口文件 |

---

### 1.4 ReAct作业

> 目录：[作业/1-大模型基础/ReAct作业](作业/1-大模型基础/ReAct作业)

**作业要求：**

1. 熟读 ReAct 论文，实现 ReAct 代码
2. 实现 Plan-and-Solve（规划-执行范式）
3. 实现 Reflection（反思范式）

**内容文件：**

| 文件 | 说明 |
|------|------|
| [ReAct.py](作业/1-大模型基础/ReAct作业/ReAct.py) | ReAct Agent 实现 |
| [ReAct-VibeCoding.py](作业/1-大模型基础/ReAct作业/ReAct-VibeCoding.py) | ReAct - Vibe Coding 版本 |
| [Plan_and_solve.py](作业/1-大模型基础/ReAct作业/Plan_and_solve.py) | Plan-and-Solve 范式实现 |
| [Reflexion.py](作业/1-大模型基础/ReAct作业/Reflexion.py) | Reflexion 范式实现 |

---

## 2. 私有化部署

### 2.1 Ollama本地部署

> 目录：[作业/2-私有化部署/2-1-ollama](作业/2-私有化部署/2-1-ollama)

**内容文件：**

| 文件 | 说明 |
|------|------|
| [01-ollama_test.py](作业/2-私有化部署/2-1-ollama/01-ollama_test.py) | Ollama 文本对话测试 |
| [02-ollama_img_test.py](作业/2-私有化部署/2-1-ollama/02-ollama_img_test.py) | Ollama 多模态图片测试 |
---

## 辅导资料

> 目录：[辅导资料](辅导资料)

| 目录 | 说明 |
|------|------|
| [Prompt Engineering For Developers](辅导资料/Prompt%20Engineering%20For%20Developers/) | 提示词工程教程（含9章内容） |
| [api编程](辅导资料/api编程/) | API编程教程（快速接入、文本模态、多模态、工具调用、记忆测试） |
| [提示词](辅导资料/提示词/) | 提示词评测与案例演练 |
