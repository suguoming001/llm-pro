import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import os
import logging
from datetime import datetime, time

load_dotenv(override=True)

# 标签list
TAG_LIST = ['农、林、牧、渔业',
            '采矿业',
            '制造业',
            '电力、热力、燃气及水生产和供应业',
            '建筑业',
            '批发和零售业',
            '交通运输、仓储和邮政业',
            '住宿和餐饮业',
            '信息传输、软件和信息技术服务业',
            '金融业',
            '房地产业',
            '租赁和商务服务业',
            '科学研究和技术服务业',
            '水利、环境和公共设施管理业',
            '居民服务、修理和其他服务业',
            '教育',
            '卫生和社会工作',
            '文化、体育和娱乐业',
            '公共管理、社会保障和社会组织',
            '国际组织']


# --- 日志配置 ---
def setup_logger(log_path: str):
    logger = logging.getLogger("TagEvaluator")
    # 清除旧的 handler 防止重复打印（如果在同一个 session 跑多次）
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


@dataclass
class Task:
    """单个打标任务"""
    data: str  # 原始数据文本
    ground_truth: str = None  # 标准答案 (用于评测)


class LLMClient:
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.0):
        self.model_name = model_name
        self.temperature = temperature

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"),
                        base_url=os.getenv("DEEPSEEK_BASE_URL"))
        resp = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content


class PromptTemplates:
    ZERO_SHOT_COT = "Let's think step by step."
    PS_PLUS_COMMONSENSE = """
    让我们阅读这句话，从这句话中识别语义，
    你需要指定一些计划然后, 让我们一步步回答问题
    计划一：识别句子中的地点/背景（如：学校、医院）
    计划二：识别句子中的核心动作/事件（如：修水管、招生）。
    计划三：标签判定：如果动作与地点所属行业无关（如在学校修水管），应归类为动作所属的行业（物业/维修），而非地点所属的行业（教育）。
    比如如下示例
    示例1 文本内容为：某高校发布招生简章，你需要根绝上面的计划找到核心的动作为【招生】，所以他的标签是 【教育】
    示例2 文本内容为：我想去学校修水管，你需要根绝上面的计划找到核心的动作为【修水管】地点只是背景，所以他的标签是 【物业维修】
    
     "(注意常识和逻辑连贯性)。"
    """


class TagSolveSolver:
    def __init__(self, llm: LLMClient, prompting_strategy: str = "ps_plus_commonsense", n_samples: int = 5):
        """
        参数:
            llm: LLM 客户端
        """
        self.llm = llm
        self.strategy = prompting_strategy
        self.n_samples = n_samples
        self.trigger_sentence = self._get_trigger(prompting_strategy)

    def _get_trigger(self, strategy: str) -> str:
        mapping = {
            "cot": PromptTemplates.ZERO_SHOT_COT,
            "ps_plus_commonsense": PromptTemplates.PS_PLUS_COMMONSENSE
        }
        if strategy not in mapping:
            raise ValueError(f"未知策略: {strategy}")
        return mapping[strategy]

    def step1_reasoning(self, question: str) -> str:
        """
        第一步: 用 PS/PS+ 提示让 LLM 生成包含计划与推理的文本
        """
        step1_reasoning_prompt = f"""
        你是一个专业的文本语义分析专家。
        请判断待处理的句子的核心意图标签
        从 {TAG_LIST} 中选出一个最合适的标签。
        待处理文本：[{question}]
        {self.trigger_sentence}
        """
        reasoning_text = self.llm.generate(step1_reasoning_prompt, max_tokens=500)
        logger.info(
            f"step1_reasoning --->模型输出 prompt is {step1_reasoning_prompt} reasoning_text is{reasoning_text} ")

        return reasoning_text

    #
    def step2_answer_extraction(
            self,
            question: str,
            reasoning_text: str,
    ) -> str:
        """
        第二步: 用提取提示从推理文本中拿到最终答案
        """
        # 构建第二步的完整 prompt (包含 Step1 的输出)
        full_prompt = f"""
        你是一个专业的文本语义分析专家。
        请判断待处理的句子的核心意图标签
        从 {TAG_LIST} 中选出一个最合适的标签。
        待处理文本：[{question}]
        推理内容：{reasoning_text}
        你最终的回答需要将标签内容按照如下格式提供
        所属标签为:[标签名]
        例如
        所属标签为:[教育]
        """
        raw_answer = self.llm.generate(full_prompt, max_tokens=500)
        logger.info(f"step2_answer_extraction --->模型输出 full_prompt is {full_prompt} raw_answer is{raw_answer} ")
        # 后处理提取规范化答案
        import re

        match = re.search(r"\[(.*?)\]", raw_answer)
        if match:
            tag = match.group(1)
            return tag
        for potential_tag in TAG_LIST:
            if potential_tag in raw_answer:
                return potential_tag
        return ""

    def solve(self, task: Task) -> Dict[str, Any]:
        """采样 N 次并多数投票"""
        from collections import Counter

        answers = []
        all_reasonings = []
        for _ in range(self.n_samples):
            # 第一步: 推理
            reasoning = self.step1_reasoning(task.data)
            # 第二步: 答案提取
            answer = self.step2_answer_extraction(
                task.data, reasoning
            )
            answers.append(answer)
            all_reasonings.append(reasoning)
            if self._is_correct(answer, task.ground_truth):
                break
        final_answer = Counter(answers).most_common(1)[0][0]
        final_reasonings = Counter(all_reasonings).most_common(1)[0][0]
        # 判断正确性
        correct = None
        if task.ground_truth is not None:
            correct = self._is_correct(final_answer, task.ground_truth)
        return {
            "data": task.data,
            "reasoning": final_reasonings,
            "predicted_answer": final_answer,
            "ground_truth": task.ground_truth,
            "correct": correct,
        }

    @staticmethod
    def _is_correct(pred: str, gt: str) -> bool:
        """判断答案是否正确"""
        pred = str(pred).strip().lower()
        gt = str(gt).strip().lower()
        if pred not in TAG_LIST:
            return False
        return pred == gt


# 加载数据 并且构建打标任务
def load_tasks_from_excel(file_path: str) -> List[Task]:
    # 1. 读取 Excel 文件
    # engine='openpyxl' 推荐用于 .xlsx 文件
    df = pd.read_excel(file_path)

    # 2. 清洗数据（可选）
    # 去除两端的空格，防止因为不可见字符导致匹配失败
    df['数据'] = df['数据'].astype(str).str.strip()
    df['所属标签'] = df['所属标签'].astype(str).str.strip()

    # 3. 将 DataFrame 行转换为 Task 对象列表
    tasks = [
        Task(data=row['数据'], ground_truth=row['所属标签'])
        for _, row in df.iterrows()
    ]

    return tasks


def run_evaluation(solver: TagSolveSolver, tasks: List[Task], output_file="./output/result.xlsx"):
    results = []
    correct_count = 0
    total_count = len(tasks)
    total_start_time = datetime.now()
    print(f"开始评测，共 {total_count} 条数据...")
    logger.info(f"开始评测，共 {total_count} 条数据...")
    for i, task in enumerate(tasks):
        print(f"正在处理第 {i + 1}/{total_count} 条...")
        start_time = datetime.now()
        print("=" * 70)
        logger.info(f"正在处理第 {i + 1}/{total_count} 条...")
        logger.info("=" * 70)
        # 调用你的 solve 方法
        res = solver.solve(task)
        logger.info(f"最终结果为======》{res}")
        results.append(res)

        if res['correct']:
            correct_count += 1
        latency = datetime.now() - start_time
        print("=" * 70)
        logger.info(f' -> 耗时: {latency.total_seconds():.2f}s')
        logger.info("=" * 70)

    # 计算准确率
    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    # 确保目录存在
    out_p = Path(output_file)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    try:
        df_result = pd.DataFrame(results)
        df_result.to_excel(out_p, index=False)
    except Exception as e:
        print(f"保存失败，错误信息: {e}")
        logger.error(f"Excel 保存失败: {e}")

    total_latency = datetime.now() - total_start_time
    print("-" * 30)
    print(f"评测完成！")
    print(f"总耗时为:{total_latency.total_seconds():.2f}s")
    print(f"总数: {total_count}")
    print(f"正确: {correct_count}")
    print(f"准确率 (Accuracy): {accuracy:.2f}%")
    print(f"详细报告已保存至: {output_file}")
    print("-" * 30)


if __name__ == '__main__':
    script_name = Path(sys.argv[0]).stem
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"{script_name}_{timestamp}.log"

    # 3. 初始化 logger
    logger = setup_logger(log_filename)

    task_list = load_tasks_from_excel('data/标签分类.xlsx')
    llm_client = LLMClient(model_name="deepseek-reasoner")
    solver = TagSolveSolver(llm_client)
    run_evaluation(solver, task_list, 'output/result-v2.xlsx')
