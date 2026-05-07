"""
Reflexion 框架复现
论文: Reflexion: Language Agents with Verbal Reinforcement Learning
      (Shinn et al., 2023)

核心思想:
  - 通过"语言反馈"而非梯度更新来强化 LLM Agent
  - 三模块架构: Actor(执行者) / Evaluator(评估者) / Self-Reflection(自我反思)
  - 双重记忆: 短期(轨迹) + 长期(反思文本)
  - 迭代循环: 生成 → 评估 → 反思 → 重试
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Callable
from collections import deque


# ============================================================
# 1. 基础数据结构
# ============================================================

@dataclass
class Trajectory:
    """轨迹: 动作和观察的序列 (短期记忆)"""
    actions: List[str] = field(default_factory=list)        # 动作列表
    observations: List[str] = field(default_factory=list)   # 观察列表
    final_output: Optional[str] = None                      # 最终输出 (代码/答案等)

    def add_step(self, action: str, observation: str):
        self.actions.append(action)
        self.observations.append(observation)

    def to_text(self) -> str:
        """格式化为可读文本, 喂给 LLM"""
        parts = []
        for a, o in zip(self.actions, self.observations):
            parts.append(f"动作: {a}")
            parts.append(f"观察: {o}")
        if self.final_output:
            parts.append(f"最终输出:\n{self.final_output}")
        return "\n".join(parts)


@dataclass
class EvaluationResult:
    """评估器的评估结果"""
    score: float          # 数值分数 (例如 0/1 或 0~1 之间)
    passed: bool          # 是否通过 (用于决定是否继续迭代)
    feedback: str = ""    # 详细反馈 (例如单元测试的输出)


# ============================================================
# 2. LLM 接口抽象
# ============================================================

class LLMBackbone:
    """LLM 调用的抽象接口"""

    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.0):
        self.model_name = model_name
        self.temperature = temperature

    def generate(self, prompt: str, system: Optional[str] = None,
                 max_tokens: int = 1024) -> str:
        """需替换为实际 API 调用"""
        raise NotImplementedError("请接入真实 LLM API")


# ============================================================
# 3. 三大核心模块
# ============================================================

class Actor(ABC):
    """
    Actor 执行者模块: 生成动作和文本
    可以是 Chain-of-Thought 或 ReAct 风格
    """

    def __init__(self, llm: LLMBackbone):
        self.llm = llm

    @abstractmethod
    def act(
        self,
        task: str,
        memory: List[str],          # 长期记忆 (反思文本列表)
        prev_trajectory: Optional[Trajectory] = None,
    ) -> Trajectory:
        """生成一条新轨迹"""
        ...


class Evaluator(ABC):
    """
    Evaluator 评估者模块: 评估 Actor 的输出
    论文中介绍了几种评估方式:
      - 精确匹配 (推理任务)
      - 启发式规则 (决策任务)
      - LLM 自评估 (任意任务)
      - 自生成单元测试 (编程任务)
    """

    @abstractmethod
    def evaluate(self, task: str, trajectory: Trajectory) -> EvaluationResult:
        ...


class SelfReflection:
    """
    Self-Reflection 自我反思模块: 将稀疏反馈转化为详细的语言反思
    论文 Section 3
    """

    DEFAULT_REFLECTION_PROMPT = """你是一个能通过自我反思不断改进的高级推理智能体。\
你将看到一次过去的尝试: 你曾被给予一个任务, 但失败了。\
请不要总结环境信息, 而是仔细思考自己究竟在哪里做错了。\
用几句话写下错误所在以及下次应该如何改正。这段反思将在你下次重试任务时作为指导。\
请只返回反思内容本身。

任务: {task}

之前的尝试:
{trajectory}

评估反馈: {feedback}

反思:"""

    def __init__(self, llm: LLMBackbone, prompt_template: Optional[str] = None):
        self.llm = llm
        self.prompt_template = prompt_template or self.DEFAULT_REFLECTION_PROMPT

    def reflect(
        self,
        task: str,
        trajectory: Trajectory,
        eval_result: EvaluationResult,
    ) -> str:
        """生成本轮的反思文本"""
        prompt = self.prompt_template.format(
            task=task,
            trajectory=trajectory.to_text(),
            feedback=eval_result.feedback,
        )
        return self.llm.generate(prompt, max_tokens=512).strip()


# ============================================================
# 4. 记忆模块 (滑动窗口)
# ============================================================

class EpisodicMemory:
    """
    长期记忆: 存储跨轮次的反思文本
    论文中容量 Ω 通常设为 1-3 (避免超出上下文长度)
    """

    def __init__(self, capacity: int = 3):
        self.capacity = capacity
        self.reflections: deque = deque(maxlen=capacity)

    def add(self, reflection: str):
        self.reflections.append(reflection)

    def get_all(self) -> List[str]:
        return list(self.reflections)

    def format_for_prompt(self) -> str:
        """格式化为可注入提示的文本"""
        if not self.reflections:
            return ""
        sections = ["你之前尝试过这个任务。以下是过去失败尝试的反思:"]
        for i, r in enumerate(self.reflections, 1):
            sections.append(f"\n第 {i} 条反思:\n{r}")
        return "\n".join(sections)

    def clear(self):
        self.reflections.clear()


# ============================================================
# 5. Reflexion Agent 主循环 (论文 Algorithm 1)
# ============================================================

@dataclass
class TrialResult:
    """单次尝试的结果"""
    trial_id: int
    trajectory: Trajectory
    evaluation: EvaluationResult
    reflection: Optional[str] = None


class ReflexionAgent:
    """
    完整的 Reflexion Agent
    实现论文 Algorithm 1: 通过自我反思进行强化学习
    """

    def __init__(
        self,
        actor: Actor,
        evaluator: Evaluator,
        self_reflector: SelfReflection,
        memory_capacity: int = 3,
        max_trials: int = 10,
    ):
        self.actor = actor
        self.evaluator = evaluator
        self.self_reflector = self_reflector
        self.memory = EpisodicMemory(capacity=memory_capacity)
        self.max_trials = max_trials

    def run(self, task: str, verbose: bool = False) -> Dict[str, Any]:
        """
        Algorithm 1 主循环:
            1. 生成初始轨迹 τ₀
            2. while 未通过 且 t < max_trials:
                 a. 生成轨迹 τₜ
                 b. 评估 → rₜ
                 c. 反思 → srₜ, 存入记忆
                 d. t++
        """
        self.memory.clear()
        trial_results: List[TrialResult] = []
        prev_trajectory: Optional[Trajectory] = None

        for t in range(self.max_trials):
            # (a) Actor 生成新轨迹 (利用记忆中的反思)
            trajectory = self.actor.act(
                task=task,
                memory=self.memory.get_all(),
                prev_trajectory=prev_trajectory,
            )

            # (b) Evaluator 评估
            eval_result = self.evaluator.evaluate(task, trajectory)

            if verbose:
                print(f"[第 {t} 轮] 通过={eval_result.passed}, 分数={eval_result.score:.3f}")

            # 提前终止: 已通过则不需要继续
            if eval_result.passed:
                trial_results.append(TrialResult(t, trajectory, eval_result))
                if verbose:
                    print(f"[第 {t} 轮] 任务已成功完成, 提前终止迭代")
                return {
                    "success": True,
                    "n_trials": t + 1,
                    "trials": trial_results,
                    "final_trajectory": trajectory,
                }

            # (c) 自我反思生成反思并写入长期记忆
            reflection = self.self_reflector.reflect(task, trajectory, eval_result)
            self.memory.add(reflection)
            trial_results.append(
                TrialResult(t, trajectory, eval_result, reflection=reflection)
            )
            prev_trajectory = trajectory

            if verbose:
                print(f"[第 {t} 轮] 反思: {reflection[:120]}...")

        # 用尽尝试次数仍未通过
        if verbose:
            print(f"达到最大尝试次数 {self.max_trials}, 仍未成功")
        return {
            "success": False,
            "n_trials": self.max_trials,
            "trials": trial_results,
            "final_trajectory": prev_trajectory,
        }


# ============================================================
# 6. 编程任务的具体实现 (HumanEval 类型)
# ============================================================

class CodeActor(Actor):
    """
    代码生成 Actor
    根据题目和过往反思生成 Python 代码
    """

    SYSTEM_PROMPT = (
        "你是一个 Python 编程助手。你将看到一个函数签名和文档字符串, "
        "你的任务是写出完整、正确的实现代码。"
    )

    PROMPT_TEMPLATE = """函数签名和文档字符串:
```python
{task}
```

{memory_section}

{previous_attempt_section}

请写出完整的函数实现。只返回有效的 Python 代码(包括函数签名), 不要其他文字。"""

    def act(self, task, memory, prev_trajectory=None) -> Trajectory:
        memory_section = ""
        if memory:
            memory_section = (
                "来自过去失败尝试的反思(请利用这些反思避免再犯同样的错误):\n"
                + "\n---\n".join(f"- {r}" for r in memory)
            )

        prev_attempt_section = ""
        if prev_trajectory and prev_trajectory.final_output:
            prev_attempt_section = (
                f"你之前(错误的)实现:\n```python\n"
                f"{prev_trajectory.final_output}\n```"
            )

        prompt = self.PROMPT_TEMPLATE.format(
            task=task,
            memory_section=memory_section,
            previous_attempt_section=prev_attempt_section,
        )
        code = self.llm.generate(prompt, system=self.SYSTEM_PROMPT, max_tokens=1024)
        # 提取代码块
        code = self._extract_code(code)

        traj = Trajectory(final_output=code)
        traj.add_step(
            action="编写 Python 实现",
            observation="代码已生成。",
        )
        return traj

    @staticmethod
    def _extract_code(text: str) -> str:
        """从 LLM 返回中提取 Python 代码块"""
        import re
        match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()


class CodeEvaluator(Evaluator):
    """
    代码评估器: 用自生成的单元测试评估
    论文 Section 4.3: 用 CoT 提示生成测试, 通过 AST 验证
    """

    TEST_GEN_PROMPT = """请为以下 Python 函数生成 {n_tests} 个多样化、高质量的单元测试, 使用 `assert` 语句。\
测试应覆盖正常情况和边界情况。只返回 assert 语句本身, 一行一个, 不要任何其他文字。

函数:
```python
{task}
```

单元测试:"""

    def __init__(self, llm: LLMBackbone, n_tests: int = 6, timeout: float = 5.0):
        self.llm = llm
        self.n_tests = n_tests
        self.timeout = timeout
        self._test_cache: Dict[str, List[str]] = {}  # 同一题目复用测试

    def generate_tests(self, task: str) -> List[str]:
        """为题目生成单元测试 (有缓存)"""
        if task in self._test_cache:
            return self._test_cache[task]

        prompt = self.TEST_GEN_PROMPT.format(task=task, n_tests=self.n_tests)
        response = self.llm.generate(prompt, max_tokens=512)

        # 过滤为合法的 assert 语句
        valid_tests = []
        for line in response.split("\n"):
            line = line.strip()
            if not line.startswith("assert"):
                continue
            if self._is_valid_python(line):
                valid_tests.append(line)

        self._test_cache[task] = valid_tests
        return valid_tests

    @staticmethod
    def _is_valid_python(code: str) -> bool:
        """用 AST 检查是否为合法 Python"""
        import ast
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def evaluate(self, task: str, trajectory: Trajectory) -> EvaluationResult:
        code = trajectory.final_output or ""
        tests = self.generate_tests(task)

        if not tests:
            return EvaluationResult(
                score=0.0, passed=False,
                feedback="没有生成有效的单元测试。",
            )

        passed_count = 0
        feedback_lines = []
        for test in tests:
            ok, err = self._run_test(code, test)
            if ok:
                passed_count += 1
                feedback_lines.append(f"通过: {test}")
            else:
                feedback_lines.append(f"失败: {test}\n   错误: {err}")

        score = passed_count / len(tests)
        all_passed = passed_count == len(tests)
        return EvaluationResult(
            score=score,
            passed=all_passed,
            feedback="\n".join(feedback_lines),
        )

    def _run_test(self, code: str, test: str) -> Tuple[bool, str]:
        """在隔离命名空间执行 code+test (生产环境应使用沙箱/子进程)"""
        namespace: Dict[str, Any] = {}
        try:
            exec(code, namespace)
            exec(test, namespace)
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"


# ============================================================
# 7. 推理/决策任务的具体实现示例
# ============================================================

class ReasoningActor(Actor):
    """简单的思维链推理 Actor (用于 HotPotQA 等)"""

    PROMPT_TEMPLATE = """请一步一步地回答以下问题。

{memory_section}

问题: {task}

让我们一步步思考。请用 "答案: <你的答案>" 的格式给出最终答案。"""

    def act(self, task, memory, prev_trajectory=None) -> Trajectory:
        memory_section = ""
        if memory:
            memory_section = (
                "你之前回答过这个问题但失败了。以下是反思:\n"
                + "\n".join(f"- {r}" for r in memory)
            )
        prompt = self.PROMPT_TEMPLATE.format(task=task, memory_section=memory_section)
        response = self.llm.generate(prompt, max_tokens=512)
        # 提取最终答案
        import re
        match = re.search(r"答案[::]\s*(.+?)(?:\n|$)", response)
        final = match.group(1).strip() if match else response.strip().split("\n")[-1]
        traj = Trajectory(final_output=final)
        traj.add_step(action="逐步推理", observation=response)
        return traj


class ExactMatchEvaluator(Evaluator):
    """精确匹配评估器 (用于 HotPotQA 等)"""

    def __init__(self, ground_truth_fn: Callable[[str], str]):
        """
        参数:
            ground_truth_fn: 输入 task, 返回标准答案
        """
        self.ground_truth_fn = ground_truth_fn

    def evaluate(self, task: str, trajectory: Trajectory) -> EvaluationResult:
        pred = (trajectory.final_output or "").lower().strip()
        gt = self.ground_truth_fn(task).lower().strip()
        passed = pred == gt or gt in pred
        return EvaluationResult(
            score=1.0 if passed else 0.0,
            passed=passed,
            feedback=f"你的答案: '{pred}'。期望答案: '{gt}'。" if not passed else "回答正确!",
        )


# ============================================================
# 8. 中文示例任务集
# ============================================================

# 中文编程任务集
chinese_code_tasks = [
    {
        "task": '''def factorial(n):
    """
    计算非负整数 n 的阶乘。
    例如:
    factorial(0) == 1
    factorial(5) == 120
    """''',
        "description": "经典递归题, 测试基础递归/循环能力",
    },
    {
        "task": '''def min_subarray_sum(nums):
    """
    给定一个整数数组 nums, 找出任意非空子数组的最小和。
    例如:
    min_subarray_sum([2, 3, 4, 1, 2, 4]) == 1
    min_subarray_sum([-1, -2, -3]) == -6
    """''',
        "description": "需要动态规划思想 (论文 Appendix C.1 的经典题)",
    },
    {
        "task": '''def is_palindrome(s):
    """
    判断字符串 s 是否为回文 (忽略大小写)。
    例如:
    is_palindrome("level") == True
    is_palindrome("Hello") == False
    """''',
        "description": "字符串处理基础题",
    },
]

# 中文推理任务集
chinese_reasoning_tasks = [
    {
        "task": "中国四大名著的作者分别是谁? 请按《三国演义》《水浒传》《西游记》《红楼梦》的顺序回答。",
        "ground_truth": "罗贯中, 施耐庵, 吴承恩, 曹雪芹",
    },
    {
        "task": "唐朝的开国皇帝和宋朝的开国皇帝分别是谁?",
        "ground_truth": "李渊, 赵匡胤",
    },
]


def demo_code_generation():
    """演示如何用 Reflexion 解决中文编程题"""

    # 假设你已实现一个真实的 LLM 客户端
    # 例如: llm = OpenAIBackbone(model_name="gpt-4")
    llm = None  # 占位

    task = chinese_code_tasks[1]["task"]  # 用第二个题作演示

    # 组装 Reflexion Agent
    agent = ReflexionAgent(
        actor=CodeActor(llm),
        evaluator=CodeEvaluator(llm, n_tests=6),
        self_reflector=SelfReflection(llm),
        memory_capacity=1,        # 论文编程任务用 Ω=1
        max_trials=5,             # 论文中通常 5-10 轮
    )

    print("=" * 70)
    print("Reflexion 中文代码生成演示 (需接入真实 LLM 才能运行)")
    print("=" * 70)
    print(f"\n[题目]:\n{task}\n")
    print("[流程]: Actor → Evaluator → SelfReflection → Memory → 重试...")
    # result = agent.run(task, verbose=True)
    # print(f"\n[结果] 成功={result['success']}, 尝试次数={result['n_trials']}")
    # print(f"\n[最终代码]:\n{result['final_trajectory'].final_output}")


def demo_reasoning():
    """演示如何用 Reflexion 解决中文推理题"""
    llm = None  # 占位

    task = chinese_reasoning_tasks[0]["task"]
    ground_truth = chinese_reasoning_tasks[0]["ground_truth"]

    agent = ReflexionAgent(
        actor=ReasoningActor(llm),
        evaluator=ExactMatchEvaluator(lambda t: ground_truth),
        self_reflector=SelfReflection(llm),
        memory_capacity=3,
        max_trials=5,
    )

    print(f"\n[推理题目]: {task}")
    print(f"[标准答案]: {ground_truth}")
    # result = agent.run(task, verbose=True)


if __name__ == "__main__":
    demo_code_generation()
    demo_reasoning()
    print("\n" + "=" * 70)
    print("使用说明:")
    print("  - 实际运行需要在 LLMBackbone.generate() 中接入真实 LLM API")
    print("  - 推荐用 GPT-4 / Claude Opus 4.7 等强模型获得论文报告的效果")
    print("  - 中文任务推荐用对中文优化的模型 (Qwen / ChatGLM / DeepSeek 等)")
    print("  - 编程任务建议在沙箱中执行 _run_test (避免恶意代码风险)")
    print("=" * 70)