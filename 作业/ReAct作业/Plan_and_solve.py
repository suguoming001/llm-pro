"""
Plan-and-Solve (PS) 提示法 复现
论文: Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning
      by Large Language Models (Wang et al., 2023)

核心思想:
  1. PS 提示法: 先制定计划, 再分步执行
  2. PS+ 提示法: 在 PS 基础上增加更详细的指令(提取变量、关注计算等)
  3. 两步流程: 第一步生成推理 → 第二步提取最终答案
"""
import os
import re
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)


# ============================================================
# 1. 提示模板定义 (来自论文 Table 5/6/7)
# ============================================================

class PromptTemplates:
    """论文中所有提示模板的集合 (中文版)"""

    # Zero-shot 思维链 (基线方法)
    ZERO_SHOT_COT = "让我们一步一步地思考。"

    # PS 提示法 (基础版)
    PS_PROMPT = (
        "让我们首先理解问题并制定一个解决方案。"
        "然后, 让我们按照计划一步一步地解决问题。"
    )

    # PS+ 提示法 (增强版, 论文主推方法)
    PS_PLUS_PROMPT = (
        "让我们首先理解问题, 提取相关变量及其对应的数值, 并制定一个完整的计划。"
        "然后, 让我们执行计划, 计算中间变量(注意正确的数值计算和常识), "
        "一步一步地解决问题, 并给出答案。"
    )

    # 针对常识推理的 PS+ 变体 (来自附录 Table 14)
    PS_PLUS_COMMONSENSE = (
        "让我们首先准备相关信息并制定计划。然后, 让我们一步步回答问题"
        "(注意常识和逻辑连贯性)。"
    )

    # 针对 Coin Flip 的 PS+ 变体 (来自附录 Table 16)
    PS_PLUS_COIN_FLIP = (
        "让我们首先理解问题并制定一个完整的计划。然后, 让我们执行计划并一步步推理。"
        "每一步回答子问题: \"这个人是否翻转了硬币? 硬币当前的状态是什么?\"。"
        "根据硬币的最终状态给出最终答案(注意每次翻转都会让硬币状态反转)。"
    )

    # 答案提取提示 (第二步)
    ANSWER_EXTRACTION = {
        "number": "因此, 答案(阿拉伯数字)是",
        "option": "因此, 在选项 A 到 E 中, 答案是",
        "yes_no": "因此, 答案(是 或 否)是",
        "string": "因此, 答案是",
    }


# ============================================================
# 2. 任务类型定义
# ============================================================

@dataclass
class Task:
    """单个推理任务"""
    question: str  # 问题文本
    task_type: str  # 任务类型: math/commonsense/symbolic
    answer_format: str  # 答案格式: number/option/yes_no/string
    ground_truth: Optional[str] = None  # 标准答案 (用于评测)


# ============================================================
# 3. LLM 接口抽象 (便于替换不同模型)
# ============================================================

class LLMClient:
    """LLM 调用的抽象接口。实际使用时替换为 OpenAI / Anthropic / 国内大模型等。"""

    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.0):
        self.model_name = model_name
        self.temperature = temperature

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """
        调用 LLM 生成文本。
        实现示例 (OpenAI):
            from openai import OpenAI
            client = OpenAI()
            resp = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        """
        client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"),
                        base_url=os.getenv("DEEPSEEK_BASE_URL"))
        resp = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content
        raise NotImplementedError("请接入实际的 LLM API")


# ============================================================
# 4. 答案提取工具 (论文 Step 2 后处理)
# ============================================================

class AnswerExtractor:
    """从 LLM 生成的文本中提取最终答案"""

    @staticmethod
    def extract_number(text: str) -> Optional[str]:
        """提取数字答案 - 取最后一个数字"""
        # 处理千分位逗号 (中英文)
        text = text.replace(",", "").replace(",", "")
        # 匹配所有数字 (包括负数和小数)
        numbers = re.findall(r"-?\d+\.?\d*", text)
        if not numbers:
            return None
        # 取最后一个数字作为答案
        ans = numbers[-1]
        if ans.endswith("."):
            ans = ans[:-1]
        return ans

    @staticmethod
    def extract_option(text: str) -> Optional[str]:
        """提取选项答案 (A/B/C/D/E)"""
        # 优先匹配 (A) 这种带括号的形式
        match = re.search(r"\(([A-E])\)", text)
        if match:
            return match.group(1)
        # 其次匹配孤立的大写字母 A-E
        match = re.search(r"\b([A-E])\b", text)
        return match.group(1) if match else None

    @staticmethod
    def extract_yes_no(text: str) -> Optional[str]:
        """提取 是/否 答案 (支持中英文)"""
        text_lower = text.lower().strip()

        # 中文: 找否定词和肯定词的最早出现位置
        no_keywords = ["不", "否", "没"]
        yes_keywords = ["是", "会", "能", "可以"]

        no_pos = -1
        for kw in no_keywords:
            pos = text.find(kw)
            if pos != -1 and (no_pos == -1 or pos < no_pos):
                no_pos = pos

        yes_pos = -1
        for kw in yes_keywords:
            pos = text.find(kw)
            if pos != -1 and (yes_pos == -1 or pos < yes_pos):
                yes_pos = pos

        if no_pos != -1 and (yes_pos == -1 or no_pos < yes_pos):
            return "否"
        if yes_pos != -1:
            return "是"

        # 英文匹配
        if re.match(r"^\s*yes\b", text_lower):
            return "是"
        if re.match(r"^\s*no\b", text_lower):
            return "否"
        en_yes_pos = text_lower.find("yes")
        en_no_pos = text_lower.find("no")
        if en_yes_pos == -1 and en_no_pos == -1:
            return None
        if en_yes_pos == -1:
            return "否"
        if en_no_pos == -1:
            return "是"
        return "是" if en_yes_pos < en_no_pos else "否"

    @staticmethod
    def extract_string(text: str) -> Optional[str]:
        """提取字符串答案 (如 last letters concatenation)"""
        # 一般是引号包裹的内容 (支持中英文引号)
        match = re.search(r'["\'""''「]([^"\'""''」]+)["\'""''」]', text)
        if match:
            return match.group(1).strip()
        # 否则取最后一个非空行
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        return lines[-1] if lines else None

    @classmethod
    def extract(cls, text: str, answer_format: str) -> Optional[str]:
        """统一入口"""
        extractors = {
            "number": cls.extract_number,
            "option": cls.extract_option,
            "yes_no": cls.extract_yes_no,
            "string": cls.extract_string,
        }
        return extractors[answer_format](text)


# ============================================================
# 5. Plan-and-Solve 主流程 (核心实现)
# ============================================================

class PlanAndSolveSolver:
    """Plan-and-Solve 提示法求解器"""

    def __init__(self, llm: LLMClient, prompting_strategy: str = "ps_plus"):
        """
        参数:
            llm: LLM 客户端
            prompting_strategy: "cot" | "ps" | "ps_plus" | "ps_plus_commonsense" | "ps_plus_coin"
        """
        self.llm = llm
        self.strategy = prompting_strategy
        self.trigger_sentence = self._get_trigger(prompting_strategy)

    def _get_trigger(self, strategy: str) -> str:
        mapping = {
            "cot": PromptTemplates.ZERO_SHOT_COT,
            "ps": PromptTemplates.PS_PROMPT,
            "ps_plus": PromptTemplates.PS_PLUS_PROMPT,
            "ps_plus_commonsense": PromptTemplates.PS_PLUS_COMMONSENSE,
            "ps_plus_coin": PromptTemplates.PS_PLUS_COIN_FLIP,
        }
        if strategy not in mapping:
            raise ValueError(f"未知策略: {strategy}")
        return mapping[strategy]

    def step1_reasoning(self, question: str) -> str:
        """
        第一步: 用 PS/PS+ 提示让 LLM 生成包含计划与推理的文本
        论文 Section 2.1
        """
        prompt = f"问题: {question}\n答: {self.trigger_sentence}"
        reasoning_text = self.llm.generate(prompt, max_tokens=500)
        print(f"step1_reasoning --->模型输出 prompt is {prompt} reasoning_text is{reasoning_text} ")

        return reasoning_text

    def step2_answer_extraction(
            self,
            question: str,
            reasoning_text: str,
            answer_format: str,
    ) -> str:
        """
        第二步: 用提取提示从推理文本中拿到最终答案
        论文 Section 2.2
        """
        extraction_trigger = PromptTemplates.ANSWER_EXTRACTION[answer_format]
        print(f"extraction_trigger --->answer_format is {answer_format} extraction_trigger is{extraction_trigger} ")

        # 构建第二步的完整 prompt (包含 Step1 的输出)
        full_prompt = (
            f"问题: {question}\n"
            f"答: {self.trigger_sentence}\n"
            f"{reasoning_text}\n"
            f"{extraction_trigger}"
        )
        raw_answer = self.llm.generate(full_prompt, max_tokens=500)
        print(f"step2_answer_extraction --->模型输出 full_prompt is {full_prompt} raw_answer is{raw_answer} ")

        # 后处理提取规范化答案
        return AnswerExtractor.extract(raw_answer, answer_format) or raw_answer.strip()

    def solve(self, task: Task) -> Dict[str, Any]:
        """完整的两步求解流程"""
        # 第一步: 推理
        reasoning = self.step1_reasoning(task.question)
        # 第二步: 答案提取
        answer = self.step2_answer_extraction(
            task.question, reasoning, task.answer_format
        )
        # 判断正确性
        correct = None
        if task.ground_truth is not None:
            correct = self._is_correct(answer, task.ground_truth, task.answer_format)
        return {
            "question": task.question,
            "reasoning": reasoning,
            "predicted_answer": answer,
            "ground_truth": task.ground_truth,
            "correct": correct,
        }

    @staticmethod
    def _is_correct(pred: str, gt: str, fmt: str) -> bool:
        """判断答案是否正确"""
        pred = str(pred).strip().lower()
        gt = str(gt).strip().lower()
        if fmt == "number":
            try:
                return abs(float(pred) - float(gt)) < 1e-4
            except ValueError:
                return pred == gt
        return pred == gt


# ============================================================
# 6. Self-Consistency 增强 (论文 Section 4.2)
# ============================================================

class SelfConsistencyPSSolver(PlanAndSolveSolver):
    """带自洽性投票的 PS 求解器 - 多次采样后多数投票"""

    def __init__(
            self,
            llm: LLMClient,
            prompting_strategy: str = "ps_plus",
            n_samples: int = 10,
            temperature: float = 0.7,
    ):
        super().__init__(llm, prompting_strategy)
        self.n_samples = n_samples
        # 自洽性需要更高 temperature 以增加多样性
        self.llm.temperature = temperature

    def solve(self, task: Task) -> Dict[str, Any]:
        """采样 N 次并多数投票"""
        from collections import Counter

        answers = []
        all_reasonings = []
        for _ in range(self.n_samples):
            reasoning = self.step1_reasoning(task.question)
            answer = self.step2_answer_extraction(
                task.question, reasoning, task.answer_format
            )
            answers.append(answer)
            all_reasonings.append(reasoning)

        # 多数投票
        final_answer = Counter(answers).most_common(1)[0][0]
        correct = None
        if task.ground_truth is not None:
            correct = self._is_correct(
                final_answer, task.ground_truth, task.answer_format
            )
        return {
            "question": task.question,
            "all_answers": answers,
            "predicted_answer": final_answer,
            "ground_truth": task.ground_truth,
            "correct": correct,
        }


# ============================================================
# 7. 评估器
# ============================================================

class Evaluator:
    """在数据集上评估准确率"""

    def __init__(self, solver: PlanAndSolveSolver):
        self.solver = solver

    def evaluate(self, tasks: List[Task], verbose: bool = False) -> Dict[str, Any]:
        results = []
        n_correct = 0
        for i, task in enumerate(tasks):
            try:
                result = self.solver.solve(task)
                results.append(result)
                if result["correct"]:
                    n_correct += 1
                if verbose:
                    status = "✓" if result["correct"] else "✗"
                    print(f"[{i + 1}/{len(tasks)}] "
                          f"预测={result['predicted_answer']} "
                          f"标准={result['ground_truth']} {status}")
            except Exception as e:
                print(f"任务 {i} 执行失败: {e}")
                results.append({"error": str(e), "correct": False})

        accuracy = n_correct / len(tasks) if tasks else 0
        print(f"\n评测完成: 准确率 {accuracy:.2%} ({n_correct}/{len(tasks)})")
        return {
            "accuracy": accuracy,
            "n_correct": n_correct,
            "n_total": len(tasks),
            "results": results,
        }


# ============================================================
# 8. 中文示例任务集
# ============================================================

# 数学推理任务集 (类似 GSM8K 风格)
chinese_math_tasks = [
    Task(
        question="小明有 5 个苹果, 妈妈又给了他 3 个, 然后他吃了 2 个。请问他现在还有几个苹果?",
        task_type="math",
        answer_format="number",
        ground_truth="6",
    ),
    Task(
        question=("一个舞蹈班有 20 名学生, 其中 20% 的人选了现代舞, "
                  "剩下学生中的 25% 选了爵士舞, 其余的人都选了街舞。"
                  "请问选街舞的学生占全班人数的百分之多少?"),
        task_type="math",
        answer_format="number",
        ground_truth="60",
    ),
    Task(
        question=("小红有 125 元, 小李的钱数比小红的 4 倍少 2 元。"
                  "请问他们俩一共有多少元?"),
        task_type="math",
        answer_format="number",
        ground_truth="623",
    ),
    Task(
        question=("一辆汽车从 A 城出发, 以每小时 60 公里的速度行驶 3 小时到达 B 城。"
                  "返程时由于堵车, 平均速度变为每小时 45 公里。"
                  "请问返程用了多少小时?"),
        task_type="math",
        answer_format="number",
        ground_truth="4",
    ),
    Task(
        question=("商店原价 200 元的商品打 8 折出售, 又因会员再打 9 折。"
                  "请问会员实际支付多少元?"),
        task_type="math",
        answer_format="number",
        ground_truth="144",
    ),
]

# 常识推理任务集 (类似 StrategyQA 风格)
chinese_commonsense_tasks = [
    Task(
        question="北京和上海, 哪一个是中国的首都?",
        task_type="commonsense",
        answer_format="string",
        ground_truth="北京",
    ),
    Task(
        question="企鹅会飞吗?",
        task_type="commonsense",
        answer_format="yes_no",
        ground_truth="否",
    ),
    Task(
        question="一个人能在不借助任何工具的情况下徒手抓到一只成年老虎吗?",
        task_type="commonsense",
        answer_format="yes_no",
        ground_truth="否",
    ),
]

# 符号推理任务集 (类似 Coin Flip 风格)
chinese_symbolic_tasks = [
    Task(
        question=("一枚硬币正面朝上。小张翻转了硬币, 小李没有翻转, "
                  "小王翻转了硬币, 小赵没有翻转。请问硬币现在还是正面朝上吗?"),
        task_type="symbolic",
        answer_format="yes_no",
        ground_truth="是",  # 翻转两次相当于没翻
    ),
    Task(
        question=("一枚硬币正面朝上。小张翻转了硬币, 小李翻转了硬币, "
                  "小王没有翻转, 小赵翻转了硬币。请问硬币现在还是正面朝上吗?"),
        task_type="symbolic",
        answer_format="yes_no",
        ground_truth="否",  # 翻转三次相当于翻一次
    ),
]

if __name__ == "__main__":
    print("=" * 70)
    print("Plan-and-Solve 提示法演示")
    print("=" * 70)
    llm_client = LLMClient(model_name="deepseek-reasoner")
    print("=" * 70)
    print("PlanAndSolveSolver 提示法演示")
    planAndSolveSolver = PlanAndSolveSolver(llm_client)
    print(planAndSolveSolver.solve(chinese_symbolic_tasks[1]))
    print("=" * 70)

    print("=" * 70)
    print("selfConsistencyPSSolver 提示法演示")
    selfConsistencyPSSolver = SelfConsistencyPSSolver(llm_client)
    print(selfConsistencyPSSolver.solve(chinese_symbolic_tasks[1]))
    print("=" * 70)

    print("\n中文测试集大小:")
    print(f"  - 数学推理: {len(chinese_math_tasks)} 题")
    print(f"  - 常识推理: {len(chinese_commonsense_tasks)} 题")
    print(f"  - 符号推理: {len(chinese_symbolic_tasks)} 题")
