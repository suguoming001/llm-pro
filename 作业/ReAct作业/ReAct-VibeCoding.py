"""
ReAct Agent 实现
参考: https://til.simonwillison.net/llms/python-react-pattern

运行前请先在 .env 中配置：
    OPENAI_API_KEY=your_api_key
    OPENAI_BASE_URL=https://api.deepseek.com/v1   # 或其他兼容 OpenAI 接口的服务
"""
from __future__ import annotations

import os
import re
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)


# =============================================================================
# Tool 定义
# =============================================================================

class Tool:
    """ReAct 可调用的工具封装"""

    def __init__(self, name: str, description: str, func: Callable[[str], str]):
        self.name = name
        self.description = description
        self.func = func

    def __call__(self, input_str: str) -> str:
        return self.func(input_str)


def _calculate(expression: str) -> str:
    """用 Python eval 计算表达式。"""
    try:
        # 只允许数学运算相关符号，防止注入
        safe_chars = set("0123456789.+-*/%() eE")
        if not all(c in safe_chars for c in expression):
            return f"Error: 表达式包含非法字符: {expression!r}"
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"


def _wikipedia(query: str) -> str:
    """查询维基百科摘要。"""
    try:
        resp = httpx.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
            },
            timeout=10,
        )
        data = resp.json()
        hits = data.get("query", {}).get("search", [])
        if not hits:
            return f"No wikipedia result for: {query}"
        return hits[0]["snippet"]
    except Exception as e:
        return f"Error: {e}"


def _python_exec(code: str) -> str:
    """执行一段 Python 代码，返回最后一行表达式的值或打印输出。"""
    import io
    import contextlib

    buf = io.StringIO()
    local_ns: Dict[str, Any] = {}
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, {"__builtins__": __builtins__}, local_ns)
        output = buf.getvalue().strip()
        if output:
            return output
        # 返回最后一个变量
        if local_ns:
            last_key = list(local_ns.keys())[-1]
            return f"{last_key} = {local_ns[last_key]}"
        return "(no output)"
    except Exception as e:
        return f"Error: {e}"


def get_default_tools() -> List[Tool]:
    """默认工具集：计算器、维基百科、Python 执行"""
    return [
        Tool(
            name="calculate",
            description=(
                "e.g. calculate: 4 * 7 / 3\n"
                "Runs a calculation using python eval and returns the number. "
                "Use floating point syntax if necessary."
            ),
            func=_calculate,
        ),
        Tool(
            name="wikipedia",
            description=(
                "e.g. wikipedia: Django\n"
                "Returns a summary from searching Wikipedia."
            ),
            func=_wikipedia,
        ),
        Tool(
            name="python",
            description=(
                "e.g. python: for i in range(3): print(i)\n"
                "Executes python code and returns the printed output or the last assigned variable. "
                "Useful for solving math/algebra problems by enumeration or using sympy-like logic."
            ),
            func=_python_exec,
        ),
    ]


# =============================================================================
# Prompt 模板
# =============================================================================

SYSTEM_PROMPT_TEMPLATE = """\
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer.

Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.
When you have enough information, output the final result with the prefix "Answer:".

Your available actions are:

{tools}

Rules:
- Each step output EXACTLY one of: an Action line followed by PAUSE, OR an Answer line.
- Action format MUST be: `Action: <tool_name>: <tool_input>`
- Do not fabricate observations; wait for the real Observation returned by the system.
- The question may be in Chinese; the Answer should be returned in the same language as the question.

Example session:

Question: 一个矩形的长是 6, 宽是 4, 面积是多少?
Thought: 我可以用 calculate 工具计算 6*4。
Action: calculate: 6 * 4
PAUSE

(system will respond)
Observation: 24

Thought: 得到了面积。
Answer: 矩形的面积是 24。
"""

ACTION_RE = re.compile(r"^Action:\s*([A-Za-z_][\w\-]*)\s*:\s*(.*)$")


# =============================================================================
# Result
# =============================================================================

@dataclass
class ReActResult:
    success: bool
    answer: Optional[str] = None
    reason: Optional[str] = None
    steps: int = 0
    trace: List[Dict[str, str]] = field(default_factory=list)


# =============================================================================
# Agent
# =============================================================================

class ReActAgent:
    def __init__(
            self,
            model: str = "gpt-4o-mini",
            max_steps: int = 5,
            tools: Optional[List[Tool]] = None,
            verbose: bool = False,
            client: Optional[OpenAI] = None,
            system_prompt: Optional[str] = None,
            temperature: float = 0.0,
    ):
        self.model = model
        self.max_steps = max_steps
        self.tools: Dict[str, Tool] = {t.name: t for t in (tools or [])}
        self.verbose = verbose
        self.temperature = temperature
        self.client = client or OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL")
        )
        self.system_prompt = system_prompt or self._render_system_prompt()

    # ---------- 内部辅助 ----------
    def _render_system_prompt(self) -> str:
        tools_block = "\n\n".join(
            f"{t.name}:\n{t.description}" for t in self.tools.values()
        )
        return SYSTEM_PROMPT_TEMPLATE.format(tools=tools_block)

    def _chat(self, messages: List[Dict[str, str]]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            stop=["Observation:"],  # 让模型在生成 Observation 之前停下
        )
        return resp.choices[0].message.content or ""

    def _log(self, *args):
        if self.verbose:
            print(*args, flush=True)

    # ---------- 核心循环 ----------
    def solve(self, question: str) -> ReActResult:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Question: {question}"},
        ]

        trace: List[Dict[str, str]] = []

        self._log("=" * 60)
        self._log(f"Question: {question}")
        self._log("=" * 60)

        for step in range(1, self.max_steps + 1):
            self._log(f"\n--- Step {step} ---")

            try:
                response = self._chat(messages).strip()
            except Exception as e:
                return ReActResult(
                    success=False,
                    reason=f"LLM 调用失败: {e}",
                    steps=step,
                    trace=trace,
                )

            self._log(response)
            messages.append({"role": "assistant", "content": response})
            trace.append({"type": "model", "content": response})

            # 1) 先看是否给出最终答案
            if "Answer:" in response:
                answer = response.split("Answer:", 1)[1].strip()
                return ReActResult(
                    success=True,
                    answer=answer,
                    steps=step,
                    trace=trace,
                )

            # 2) 解析一个 Action
            action_match = None
            for line in response.splitlines():
                m = ACTION_RE.match(line.strip())
                if m:
                    action_match = m
                    break

            if action_match is None:
                return ReActResult(
                    success=False,
                    reason="模型没有输出 Action 也没有给出 Answer。",
                    steps=step,
                    trace=trace,
                )

            action_name, action_input = action_match.groups()
            action_input = action_input.strip()

            if action_name not in self.tools:
                observation = (
                    f"Unknown action: {action_name}. "
                    f"Available: {list(self.tools.keys())}"
                )
            else:
                try:
                    observation = self.tools[action_name](action_input)
                except Exception as e:
                    observation = f"Error running {action_name}: {e}"

            self._log(f"Observation: {observation}")
            obs_msg = f"Observation: {observation}"
            messages.append({"role": "user", "content": obs_msg})
            trace.append({"type": "observation", "content": observation})

        return ReActResult(
            success=False,
            reason=f"达到最大步数 {self.max_steps} 仍未给出答案。",
            steps=self.max_steps,
            trace=trace,
        )


# =============================================================================
# Builder
# =============================================================================

class ReActAgentBuilder:
    """链式构建 ReActAgent"""

    def __init__(self):
        self._model: str = "gpt-4o-mini"
        self._max_steps: int = 5
        self._tools: List[Tool] = []
        self._verbose: bool = False
        self._client: Optional[OpenAI] = None
        self._system_prompt: Optional[str] = None
        self._temperature: float = 0.0

    def with_model(self, model: str) -> "ReActAgentBuilder":
        self._model = model
        return self

    def with_max_steps(self, max_steps: int) -> "ReActAgentBuilder":
        self._max_steps = max_steps
        return self

    def with_tools(self, tools: List[Tool]) -> "ReActAgentBuilder":
        self._tools = list(tools)
        return self

    def with_default_tools(self) -> "ReActAgentBuilder":
        self._tools = get_default_tools()
        return self

    def add_tool(self, tool: Tool) -> "ReActAgentBuilder":
        self._tools.append(tool)
        return self

    def with_verbose(self, verbose: bool = True) -> "ReActAgentBuilder":
        self._verbose = verbose
        return self

    def with_client(self, client: OpenAI) -> "ReActAgentBuilder":
        self._client = client
        return self

    def with_system_prompt(self, prompt: str) -> "ReActAgentBuilder":
        self._system_prompt = prompt
        return self

    def with_temperature(self, temperature: float) -> "ReActAgentBuilder":
        self._temperature = temperature
        return self

    def build(self) -> ReActAgent:
        return ReActAgent(
            model=self._model,
            max_steps=self._max_steps,
            tools=self._tools,
            verbose=self._verbose,
            client=self._client,
            system_prompt=self._system_prompt,
            temperature=self._temperature,
        )


if __name__ == "__main__":
    question = "笼子里有鸡和兔子工35只，脚共有94只，请问鸡和兔子各有多少只？"
    agent = ReActAgentBuilder().with_model("deepseek-chat").with_max_steps(10).with_default_tools().with_verbose(
        True).build()
    result = agent.solve(question)
    if result.success:
        print(result.answer)
    else:
        print(f"失败: {result.reason}")
