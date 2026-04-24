import os
from typing import Callable, List, Optional, Dict

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

from tavily import TavilyClient


class Tool:
    """ReAct 可调用的工具封装"""

    def __init__(self, name: str, description: str, func: Callable[[str], str]):
        self.name = name
        self.description = description
        self.func = func

    def __call__(self, input_str: str) -> str:
        return self.func(input_str)


def _tavily(query: str) -> str:
    try:
        client = TavilyClient(os.getenv("TAVILY_TOKEN"))
        response = client.search(
            query="长江有多长",
            include_answer="advanced",
            search_depth="advanced",
            time_range="month"
        )
        if response:
            return response['answer']
        return f"对于这个问题{query} tavily 中没有找到结果"
    except Exception as e:
        return f"Error: {e}"


def calculate(expression: str) -> str:
    """只支持 + - * / 和括号的四则运算计算器。"""

    class Parser:
        def __init__(self, s: str):
            # 去掉所有空白
            self.s = "".join(s.split())
            self.i = 0

        def peek(self):
            return self.s[self.i] if self.i < len(self.s) else ""

        def advance(self):
            ch = self.peek()
            self.i += 1
            return ch

        # expr   := term (('+' | '-') term)*
        def parse_expr(self):
            value = self.parse_term()
            while self.peek() in ("+", "-"):
                op = self.advance()
                rhs = self.parse_term()
                value = value + rhs if op == "+" else value - rhs
            return value

        # term   := factor (('*' | '/') factor)*
        def parse_term(self):
            value = self.parse_factor()
            while self.peek() in ("*", "/"):
                op = self.advance()
                rhs = self.parse_factor()
                if op == "*":
                    value = value * rhs
                else:
                    if rhs == 0:
                        raise ZeroDivisionError("除数不能为 0")
                    value = value / rhs
            return value

        # factor := ('+' | '-') factor | '(' expr ')' | number
        def parse_factor(self):
            ch = self.peek()
            if ch == "+":
                self.advance()
                return self.parse_factor()
            if ch == "-":
                self.advance()
                return -self.parse_factor()
            if ch == "(":
                self.advance()
                value = self.parse_expr()
                if self.peek() != ")":
                    raise ValueError("缺少右括号 ')'")
                self.advance()
                return value
            return self.parse_number()

        def parse_number(self):
            start = self.i
            dot_seen = False
            while self.i < len(self.s) and (self.s[self.i].isdigit() or self.s[self.i] == "."):
                if self.s[self.i] == ".":
                    if dot_seen:
                        raise ValueError(f"数字格式错误: '{self.s[start:self.i + 1]}'")
                    dot_seen = True
                self.i += 1
            if start == self.i:
                raise ValueError(f"位置 {self.i} 处期望一个数字，实际得到 '{self.peek()}'")
            return float(self.s[start:self.i])

        def parse(self):
            value = self.parse_expr()
            if self.i != len(self.s):
                raise ValueError(f"多余的字符: '{self.s[self.i:]}'")
            return value

    try:
        result = Parser(expression).parse()
        # 整数就按整数显示，避免出现 3.0
        if result == int(result):
            return str(int(result))
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def calculate(expression: str) -> str:
    """只支持 + - * / 和括号的四则运算计算器。"""

    class Parser:
        def __init__(self, s: str):
            # 去掉所有空白
            self.s = "".join(s.split())
            self.i = 0

        def peek(self):
            return self.s[self.i] if self.i < len(self.s) else ""

        def advance(self):
            ch = self.peek()
            self.i += 1
            return ch

        # expr   := term (('+' | '-') term)*
        def parse_expr(self):
            value = self.parse_term()
            while self.peek() in ("+", "-"):
                op = self.advance()
                rhs = self.parse_term()
                value = value + rhs if op == "+" else value - rhs
            return value

        # term   := factor (('*' | '/') factor)*
        def parse_term(self):
            value = self.parse_factor()
            while self.peek() in ("*", "/"):
                op = self.advance()
                rhs = self.parse_factor()
                if op == "*":
                    value = value * rhs
                else:
                    if rhs == 0:
                        raise ZeroDivisionError("除数不能为 0")
                    value = value / rhs
            return value

        # factor := ('+' | '-') factor | '(' expr ')' | number
        def parse_factor(self):
            ch = self.peek()
            if ch == "+":
                self.advance()
                return self.parse_factor()
            if ch == "-":
                self.advance()
                return -self.parse_factor()
            if ch == "(":
                self.advance()
                value = self.parse_expr()
                if self.peek() != ")":
                    raise ValueError("缺少右括号 ')'")
                self.advance()
                return value
            return self.parse_number()

        def parse_number(self):
            start = self.i
            dot_seen = False
            while self.i < len(self.s) and (self.s[self.i].isdigit() or self.s[self.i] == "."):
                if self.s[self.i] == ".":
                    if dot_seen:
                        raise ValueError(f"数字格式错误: '{self.s[start:self.i + 1]}'")
                    dot_seen = True
                self.i += 1
            if start == self.i:
                raise ValueError(f"位置 {self.i} 处期望一个数字，实际得到 '{self.peek()}'")
            return float(self.s[start:self.i])

        def parse(self):
            value = self.parse_expr()
            if self.i != len(self.s):
                raise ValueError(f"多余的字符: '{self.s[self.i:]}'")
            return value

    try:
        result = Parser(expression).parse()
        # 整数就按整数显示，避免出现 3.0
        if result == int(result):
            return str(int(result))
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def _calculate(expression: str) -> str:
    """只支持 + - * / 和括号的四则运算计算器。"""

    class Parser:
        def __init__(self, s: str):
            # 去掉所有空白
            self.s = "".join(s.split())
            self.i = 0

        def peek(self):
            return self.s[self.i] if self.i < len(self.s) else ""

        def advance(self):
            ch = self.peek()
            self.i += 1
            return ch

        # expr   := term (('+' | '-') term)*
        def parse_expr(self):
            value = self.parse_term()
            while self.peek() in ("+", "-"):
                op = self.advance()
                rhs = self.parse_term()
                value = value + rhs if op == "+" else value - rhs
            return value

        # term   := factor (('*' | '/') factor)*
        def parse_term(self):
            value = self.parse_factor()
            while self.peek() in ("*", "/"):
                op = self.advance()
                rhs = self.parse_factor()
                if op == "*":
                    value = value * rhs
                else:
                    if rhs == 0:
                        raise ZeroDivisionError("除数不能为 0")
                    value = value / rhs
            return value

        # factor := ('+' | '-') factor | '(' expr ')' | number
        def parse_factor(self):
            ch = self.peek()
            if ch == "+":
                self.advance()
                return self.parse_factor()
            if ch == "-":
                self.advance()
                return -self.parse_factor()
            if ch == "(":
                self.advance()
                value = self.parse_expr()
                if self.peek() != ")":
                    raise ValueError("缺少右括号 ')'")
                self.advance()
                return value
            return self.parse_number()

        def parse_number(self):
            start = self.i
            dot_seen = False
            while self.i < len(self.s) and (self.s[self.i].isdigit() or self.s[self.i] == "."):
                if self.s[self.i] == ".":
                    if dot_seen:
                        raise ValueError(f"数字格式错误: '{self.s[start:self.i + 1]}'")
                    dot_seen = True
                self.i += 1
            if start == self.i:
                raise ValueError(f"位置 {self.i} 处期望一个数字，实际得到 '{self.peek()}'")
            return float(self.s[start:self.i])

        def parse(self):
            value = self.parse_expr()
            if self.i != len(self.s):
                raise ValueError(f"多余的字符: '{self.s[self.i:]}'")
            return value

    try:
        result = Parser(expression).parse()
        # 整数就按整数显示，避免出现 3.0
        if result == int(result):
            return str(int(result))
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def get_default_tools() -> List[Tool]:
    """默认工具集：计算器、tavily"""
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
            name="tavily",
            description=(
                "e.g. tavily: Django\n"
                "Returns a summary from searching Tavily."
            ),
            func=_tavily,
        ),

    ]


class ReActAgent:
    def __init__(
            self,
            model: str = "gpt-4o-mini",
            max_steps: int = 5,
            tools: Optional[List[Tool]] = None,
            system_prompt: Optional[str] = None,
    ):
        self.model = model
        self.max_steps = max_steps
        self.tools: Dict[str, Tool] = {t.name: t for t in (tools or [])}
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL")
        )
        self.system_prompt = system_prompt

    def _log(self, *args):
        if self.verbose:
            print(*args, flush=True)

    def _chat(self, messages: List[Dict[str, str]]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            stop=["Observation:"],  # 让模型在生成 Observation 之前停下
        )
        return resp.choices[0].message.content or ""
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


class ReActAgentBuilder:
    """链式构建 ReActAgent"""

    def __init__(self):
        self._model: str = "gpt-4o-mini"
        self._max_steps: int = 5
        self._tools: List[Tool] = []
        self._system_prompt: Optional[str] = None

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

    def build(self) -> ReActAgent:
        return ReActAgent(
            model=self._model,
            max_steps=self._max_steps,
            tools=self._tools,
            system_prompt=self._system_prompt,
        )
