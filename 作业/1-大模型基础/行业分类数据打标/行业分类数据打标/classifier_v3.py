import json
import re
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


SYSTEM_PROMPT = f"""你是一个专业的中文文本行业分类专家。你的任务是判断给定文本最匹配的 **一个** 行业类别。

## 行业类别列表（共 20 类，基于国民经济行业分类标准）
{TAG_LIST}

## 行业分类的准则
- 农、林、牧、渔业
    - 农业、林业、畜牧业、渔业、农、林、牧、渔专业及辅助性活动
- 采矿业
    - 煤炭开采和洗选业、石油和天然气开采业、黑色金属矿采选业、有色金属矿采选业、非金属矿采选业、开采专业及辅助性活动、其他采矿业
- 制造业
    -（主要包含）：农副食品加工业、食品制造业、酒/饮料和精制茶制造业、纺织业、医药制造业、汽车制造业、计算机/通信和其他电子设备制造业、通用设备制造业、专用设备制造业等（共31个大类）
- 电力、热力、燃气及水生产和供应业
    - 电力、热力生产和供应业、燃气生产和供应业、水的生产和供应业
- 建筑业
    -房屋建筑业、土木工程建筑业、建筑安装业、建筑装饰/装修和其他建筑业
- 批发和零售业
    - 批发业、零售业
- 交通运输、仓储和邮政业
    - 铁路运输业、道路运输业、水上运输业、航空运输业、管道运输业、多式联运和运输代理业、装卸搬运和仓储业、邮政业
- 住宿和餐饮业
    - 住宿业、餐饮业
- 信息传输、软件和信息技术服务业
    - 电信/广播电视和卫星传输服务、互联网和相关服务、软件和信息技术服务业
- 金融业
    - 货币金融服务、资本市场服务、保险业、其他金融业
- 房地产业
    - 房地产业
- 租赁和商务服务业
    - 租赁业、商务服务业
- 科学研究和技术服务业
    - 研究和试验发展、专业技术服务业、科技推广和应用服务业
- 水利、环境和公共设施管理业
    - 水利管理业、生态保护和环境治理业、公共设施管理业、土地管理业
- 居民服务、修理和其他服务业
    - 居民服务业、机动车/电子产品和日用产品修理业、其他服务业
- 教育
    - 教育
- 卫生和社会工作
    - 卫生、社会工作
- 文化、体育和娱乐业
    - 新闻和出版业、广播/电视/电影和录音制作业、文化艺术业、体育、娱乐业
- 公共管理、社会保障和社会组织
    - 中国共产党机关、国家机构、人民政协/民主党派、社会保障、群众团体/社会团体和其他成员组织、基层群众自治组织及其他组织
- 国际组织
    - 国际组织

## 分类要求
1. **只选一个**：从上述 20 个类别中选出最匹配的一个，不可自创类别。
2. **就高不就低**：当文本涉及多个行业时，选与文本核心主题最相关的那个。
3. **区分易混淆项**：
   - 软件开发公司的新闻 → "信息传输、软件和信息技术服务业"（而非"制造业"）
   - 房屋装修施工 → "建筑业"（而非"房地产业"）
   - 医疗器械生产 → "制造业"（而非"卫生和社会工作"）
   - 电商平台卖货 → "批发和零售业"（而非"信息传输、软件和信息技术服务业"）
   - 农产品加工厂 → "制造业"（而非"农、林、牧、渔业"）
   - 银行、保险、证券 → "金融业"
   - 大学、中小学、培训机构 → "教育"
   - 医院、诊所、疾控 → "卫生和社会工作"
   - 影视、游戏、体育赛事 → "文化、体育和娱乐业"
   - 物流快递 → "交通运输、仓储和邮政业"
4. **关注动词和业务实质**：不要只看名词，要理解文本描述的核心业务活动。

## 输出要求
### 思维链(cot)
先在 `reasoning` 中简要记录推理过程：
- 提取文本中的 **关键词和核心业务活动**
- 列出 **2-3 个候选行业** 并逐一分析匹配度
- 说明 **为什么最终选择这个** 而排除其他

### 输出格式
严格输出以下 JSON（不要输出任何其他内容）：

```json
{{
  "reasoning": "推理过程...",
  "tag": "选中的行业类别（必须是列表中的原文）",
  "confidence": 0.0,
  "keywords": ["关键词1", "关键词2"],
  "runner_up": "第二可能的行业类别",
  "runner_up_reason": "为什么排除这个候选"
}}
```"""

FEW_SHOT_EXAMPLES: List[Dict[str, str]] = [
    # ── 示例 1：明确场景 ──
    {
        "role": "user",
        "content": "请判断以下文本属于哪个行业类别：\n\n华为近日发布了新一代5G基站设备，搭载自研芯片，支持大规模MIMO天线技术，将在全球30多个国家部署。"
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "reasoning": (
                "关键词提取：5G基站、自研芯片、MIMO天线、设备发布、全球部署。\n"
                "候选分析：\n"
                "1.「制造业」- 涉及芯片和设备制造，但文本重点是通信设备和部署。\n"
                "2.「信息传输、软件和信息技术服务业」- 5G基站属于信息传输基础设施，核心业务是通信网络建设。\n"
                "3.「科学研究和技术服务业」- 涉及自研技术，但这是产品发布而非纯研究。\n"
                "最终判断：文本核心是 5G 通信设备的发布和部署，属于信息传输领域。"
            ),
            "tag": "信息传输、软件和信息技术服务业",
            "confidence": 0.88,
            "keywords": ["5G基站", "自研芯片", "MIMO天线", "全球部署"],
            "runner_up": "制造业",
            "runner_up_reason": "虽涉及设备制造，但文本重点在通信网络部署而非生产制造过程"
        }, ensure_ascii=False, indent=2)
    },

    # ── 示例 2：模糊/易混淆场景 ──
    {
        "role": "user",
        "content": "请判断以下文本属于哪个行业类别：\n\n京东超市年货节期间，线上下单量同比增长45%，生鲜品类尤其火爆，冷链物流配送能力成为关键竞争力。"
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "reasoning": (
                "关键词提取：京东超市、年货节、线上下单、生鲜品类、冷链物流。\n"
                "候选分析：\n"
                "1.「批发和零售业」- 京东超市本质是零售平台，年货节是促销活动，下单量是零售指标。\n"
                "2.「信息传输、软件和信息技术服务业」- 京东是互联网公司，但此文本描述的是零售业务。\n"
                "3.「交通运输、仓储和邮政业」- 提到冷链物流，但这只是零售的配套环节，非文本主题。\n"
                "最终判断：文本核心是电商零售促销活动和销售数据，归入批发和零售业。"
            ),
            "tag": "批发和零售业",
            "confidence": 0.90,
            "keywords": ["京东超市", "年货节", "线上下单", "生鲜品类", "销量增长"],
            "runner_up": "交通运输、仓储和邮政业",
            "runner_up_reason": "冷链物流只是配套环节，文本主题是零售促销而非物流本身"
        }, ensure_ascii=False, indent=2)
    },

    # ── 示例 3：跨行业/需要深入理解 ──
    {
        "role": "user",
        "content": "请判断以下文本属于哪个行业类别：\n\n中国农业大学联合多家企业，启动了转基因抗虫棉花新品种的田间试验，预计三年内完成审批并推广种植。"
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "reasoning": (
                "关键词提取：中国农业大学、转基因、抗虫棉花、田间试验、品种审批、推广种植。\n"
                "候选分析：\n"
                "1.「农、林、牧、渔业」- 涉及棉花种植和品种推广，属于农业范畴。\n"
                "2.「科学研究和技术服务业」- 大学牵头、转基因研发、田间试验，带有明显科研性质。\n"
                "3.「教育」- 中国农业大学是高校，但文本描述的是科研活动而非教学。\n"
                "最终判断：文本核心是转基因品种的科学研究和田间试验，由高校和企业联合开展，"
                "属于科学研究和技术服务范畴，虽然最终应用于农业，但当前阶段是研发试验。"
            ),
            "tag": "科学研究和技术服务业",
            "confidence": 0.78,
            "keywords": ["转基因", "田间试验", "品种审批", "农业大学", "联合研发"],
            "runner_up": "农、林、牧、渔业",
            "runner_up_reason": "最终应用确实在农业，但文本描述的当前活动是科研试验而非农业生产"
        }, ensure_ascii=False, indent=2)
    },
]


@dataclass
class Task:
    """单个打标任务"""
    data: str  # 原始数据文本
    ground_truth: str = None  # 标准答案 (用于评测)


class LLMClient:
    def __init__(self, model_name: str = "deepseek-v4-pro", api_key: str = os.getenv("DEEPSEEK_API_KEY"),
                 base_url: str = os.getenv("DEEPSEEK_BASE_URL"), temperature: float = 0.0):
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = api_key
        self.base_url = base_url

    def generate(self, messages: List[Dict[str, str]], max_tokens: int = 512) -> str:
        client = OpenAI(api_key=self.api_key,
                        base_url=self.base_url)
        full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

        resp = client.chat.completions.create(
            model=self.model_name,
            messages=full_messages,
            temperature=self.temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content


class TagSolveSolver:
    def __init__(self, llm: LLMClient, n_samples: int = 5):
        """
        参数:
            llm: LLM 客户端
        """
        self.llm = llm
        self.n_samples = n_samples

    def solve(self, task: Task) -> Dict[str, Any]:
        from collections import Counter

        messages = list(FEW_SHOT_EXAMPLES)
        text = task.data
        messages.append({
            "role": "user",
            "content": f"请判断以下文本属于哪个行业类别：\n\n{text}"
        })
        answers = []
        answers_reasoing_dicct = dict()
        for _ in range(self.n_samples):
            raw = self.llm.generate(messages, max_tokens=20000)
            logger.info(
                f"模型输出 prompt is {messages} reponse is{raw} ")

            result = self._parse(raw)
            result["_input"] = text
            answer = result.get('tag', '')
            answers.append(answer)
            answers_reasoing_dicct[answer] = result
        final_answer = Counter(answers).most_common(1)[0][0]
        final_reasonings = answers_reasoing_dicct[final_answer]
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

    def _parse(self, response: str) -> Dict[str, Any]:
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        match = re.search(r'```(?:json)?\s*\n(.*?)\n```', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        return {"_parse_error": True, "_raw": response}

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
    # llm_client = LLMClient(model_name="mimo-v2-pro",api_key=os.getenv("MIMO_API_KEY"),base_url=os.getenv("MIMO_BASE_URL"))
    llm_client = LLMClient(model_name="deepseek-reasoner")
    solver = TagSolveSolver(llm_client)
    run_evaluation(solver, task_list, './output/result-v3.xlsx')
