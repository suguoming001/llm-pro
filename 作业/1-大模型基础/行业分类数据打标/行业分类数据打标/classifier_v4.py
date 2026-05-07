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
SYSTEM_PROMPT = f"""
你是一名中文行业分类助手。请根据输入文本，将其归入最匹配的一个行业类别。

你必须且只能从以下 20 个类别中选择 1 个：
{TAG_LIST}

请按以下顺序判断：
1. 主题 topic 属于哪个行业领域
2. 核心动作是什么（用于辅助判断 topic）
3. 主体是谁（仅在 topic 模糊时用于修正）

总原则：
- 最终 tag 以 topic 所属行业领域为准
- 不要优先按发布主体、文本体裁、表达方式来决定 tag
- 主体和核心动作只用于解决 topic 模糊时的边界冲突
- 即使文本是“介绍、讲解、总结、教程、科普、通知、规划、报道、评论”，也不要先按体裁归类，先看 topic 属于哪个行业
- 不要因为文本出现某行业关键词，就机械归入对应行业
- 不要因为文本出现“介绍、讲解、科普、总结、教程”等词，就机械归入 `教育`
- 若文本是知识讲解/教程类，先判断它是在讲“某行业知识”，还是 topic 本身就是“教育服务”：
  - 主题是学校教育、培训、备考、课程、招生、教研 -> `教育`
  - 主题是计算机、电力、建筑、农业科技、医学、金融等专业知识 -> 按该知识所属行业归类

# 一、topic优先总分流规则

最终 tag 以 topic 所属行业领域为准，可按以下方式理解：

1. 主题是软件、编程、算法、系统、网络安全、数据库、数据处理、云平台、IT架构、互联网产品、数字技术服务
   -> `信息传输、软件和信息技术服务业`

2. 主题是材料性能、工业制品、制造工艺、设备部件、工业原料、材料用途、产品说明、设备修复、工业加工
   -> `制造业`

3. 主题是矿山、原矿、采选、储量、矿权、矿山建设、资源开采、原矿供应
   -> `采矿业`

4. 主题是农业生产、种植养殖、田间管理、农产品供给、稳产增收、林牧渔生产活动
   -> `农、林、牧、渔业`

5. 主题是农业机理、微生物肥料、菌剂研发、农业试验、农业技术原理、农业课题研究、品种试验
   -> `科学研究和技术服务业`

6. 主题是电力系统、发电、供热、燃气、供水、电网运行、并网、能源供给、电力调度、供能原理
   -> `电力、热力、燃气及水生产和供应业`

7. 主题是工程建设、施工改造、房建、园林、景观、水景、绿道、室内装潢、弱电、停车设施、铁路/道路/桥梁建设
   -> `建筑业`

8. 主题是污水、污泥、废气、垃圾、脱硫脱硝、排放治理、生态修复、环境治理、公共设施运维
   -> `水利、环境和公共设施管理业`

9. 主题是疾病诊疗、护理、医疗、卫生服务、公共卫生、康复、健康干预
   -> `卫生和社会工作`

10. 主题是历史文化、遗产传播、艺术风格、菜系文化、文化评论、娱乐传播、体育文化、人物文化传播
   -> `文化、体育和娱乐业`

11. 主题是房屋开发经营、房地产交易、租售、物业经营、不动产经营
   -> `房地产业`

12. 主题是融资、投资、并购、证券、基金、保险、资本运作、银行、信贷、理财、资产管理
   -> `金融业`

13. 主题是物流运输、仓储、邮政快递、交通运输运营、线路运营、货运客运组织
   -> `交通运输、仓储和邮政业`

14. 主题是咨询、审计、法律、认证、人力服务、招商、会展、企业服务、市场服务、商务代理
   -> `租赁和商务服务业`

15. 主题是学校教育、培训、备考、课程、教辅、教研、招生、授课、考试服务
   -> `教育`

16. 主题是政府治理、公共政策、行政认定、审批、监管、考核、督导、社会保障制度、公共组织运行
   -> `公共管理、社会保障和社会组织`

17. 主题是国际组织自身的规约、项目、宣言、数据库、指数、战略、纪念活动、全球治理事务
   -> `国际组织`

18. 主题是批发、零售、商品销售、门店经营、电商零售、商品流通
   -> `批发和零售业`

19. 主题是酒店、住宿、餐饮经营、餐馆服务、菜品制作与餐饮流程
   -> `住宿和餐饮业`

20. 主题是维修、家政、生活服务、便民服务、个人服务、修理服务
   -> `居民服务、修理和其他服务业`

# 二、主体修正规则（仅在 topic 不明确时使用）

1. 如果主体是联合国、WHO、UNICEF、FAO、世界银行、国际法院等国际组织自身，
且文本是其规约、宣言、报告、项目、战略、指数、数据库、纪念活动或官网栏目介绍，
优先归入 `国际组织`。

2. 如果主体是德勤、普华永道、安永、毕马威等专业服务机构自身，
且文本是在介绍其咨询、鉴证、审计、并购顾问、估值、重组、客户业务、组织结构、社会责任、数字化解决方案，
优先归入 `租赁和商务服务业`。

3. 如果文本同时有官方发布、媒体解读、专家评论，
不要先按主体定类，仍先看 topic 属于哪个行业。
只有当 topic 同时可能落入多个行业时，再用主体进行修正判断。

4. 即使主体是政府、法院、部委、事业单位，
也不要直接归入 `公共管理、社会保障和社会组织`；
只有当 topic 本身就是治理、监管、认定、制度、政策执行、公共组织运行时，才归入该类。

# 三、核心动作修正规则（用于辅助 topic 判断）

1. 金融业
若核心动作是并购、重组、投融资、股权运作、股权多元化、战略投资者、投资风险管理，
且 topic 指向资本与金融活动，
归 `金融业`。

2. 公共管理、社会保障和社会组织
若核心动作是行政认定、审批、监管、考核、政策执行、宏观治理、官方统计发布，
且 topic 指向治理、制度、监管体系、公共管理事务，
归 `公共管理、社会保障和社会组织`。

特别注意：
- 不要仅因为主体是国家统计局、国务院新闻办公室、部委发布会、调查总队等，就直接归入 `公共管理、社会保障和社会组织`
- 如果宏观数据发布文本的 topic 是“官方宏观经济运行与治理信息发布”，可归 `公共管理、社会保障和社会组织`
- 但如果 topic 更偏行业研究分析、市场预测、学术研究，则仍按 topic 所属领域判断

3. 科学研究和技术服务业
若核心动作是实验、论文、研发、模型、研究报告、测算、监测、机理分析、技术综述，
且 topic 指向某类研究或技术原理，
归 `科学研究和技术服务业`。

4. 教育
若核心动作是授课、培训、招生、备考辅导、课程讲解、教材教辅服务，
且 topic 本身是教育服务，
归 `教育`。
但如果只是用教学语气介绍某专业知识，不归 `教育`。

# 四、重点边界细化规则

## 1. 教育 vs 专业知识领域
- 如果 topic 是学校教育、考试培训、备考、课程、招生、教研、教辅服务
  -> `教育`
- 如果文本虽然采用“介绍、讲解、总结、教程、科普”的表达方式，
  但 topic 属于计算机、电力、建筑、农业科技、医学、金融等专业知识，
  则按 topic 所属行业归类，不归 `教育`

例如：
- 工厂模式、K-means、网络安全概念讲解 -> `信息传输、软件和信息技术服务业`
- 电网频率、电动机原理、电力系统知识 -> `电力、热力、燃气及水生产和供应业`
- 微生物肥料机理、农业菌剂研发 -> `科学研究和技术服务业`

## 2. 建筑业 vs 水利、环境和公共设施管理业 vs 文化、体育和娱乐业
- 开工、承建、施工、改造、装修、装潢应用 -> `建筑业`
- 城市公共设施配套、环境改善、生态建设、污染治理、公共设施运维 -> `水利、环境和公共设施管理业`
- 纯审美评论、艺术风格史、文化介绍 -> `文化、体育和娱乐业`

特别地：
- 室内装潢风格介绍，只要重点在住宅、公寓、家具、配色、装饰的家装应用，优先归 `建筑业`
- 园林绿化、景观工程、水景、绿道、城市景观节点、室内装潢、弱电系统、停车场等文本，
  只要核心在工程场景中的设计、构造、用途、功能、改造、施工、实施或应用说明，
  优先归为 `建筑业`
- 即使文本使用“介绍、讲解、用途、特点、作用”等表达，
  只要这些内容是在说明工程对象本身（如水景、景观、装潢、弱电、停车设施）的工程应用，
  也不要直接归入 `教育`、`文化、体育和娱乐业` 或 `水利、环境和公共设施管理业`
- 只有当文本主线转为污水处理、污染治理、生态修复、排放治理、公共设施运维时，
  才优先考虑 `水利、环境和公共设施管理业`

## 3. 制造业 vs 采矿业
- 材料性能、工业制品、产品用途、设备修复、加工工艺、零部件、生产制造
  -> `制造业`
- 如果文本核心在工业材料的性能参数、工艺指标、用途说明、设备工艺处理，优先 `制造业`
- 若对象是矿物或矿石，但文本核心在材料加工适用性、工业填料/原料用途、产品性能说明，也优先 `制造业`

- 资源开采、原矿、采选、矿山采掘
  -> `采矿业`
- 如果文本核心在勘探、开采、采选、储量、矿山建设、矿权、矿山经营、原矿供应，优先 `采矿业`
- 如果只是讲矿物材料性能、加工用途、工业用途，不一定是采矿业，优先看是否更符合 `制造业`

专门补充：
- 矿物、矿石、非金属矿材料相关文本中：
  若核心在勘探、开采、采选、储量、矿山建设、矿权、原矿生产、矿山经营、资源供应，
  归为 `采矿业`
- 若文本虽然对象是矿物或矿石，但核心在材料性能、工艺指标、加工适用性、检测参数、工业原料用途、填料用途、产品说明、下游应用，
  归为 `制造业`
- 不要仅因为文本出现“高岭土、铁矿石、石膏、矿物、矿石”等对象词，就直接归入 `采矿业`
- 判断关键不在“是不是矿”，而在“文本落在哪个产业链环节”：
  - 上游资源获取 -> `采矿业`
  - 中下游材料加工与工业应用 -> `制造业`

## 4. 农、林、牧、渔业 vs 科学研究和技术服务业
- 农业相关文本中，如果核心在种植、养殖、施肥、田间管理、稳产增收、作物生产实践、渔牧林业生产活动，
  优先归为 `农、林、牧、渔业`
- 如果文本虽然与农业有关，但核心在肥料机理、微生物技术、菌剂研发、品种试验、作用原理、实验研究、技术瓶颈、课题综述，
  优先归为 `科学研究和技术服务业`
- 特别是出现以下信号时，优先考虑 `科学研究和技术服务业`：
  `机理 / 原理 / 试验站 / 研发 / 基因转移 / 技术难点 / 实验 / 综述 / 菌剂 / 微生物制剂 / 生物肥料`
- 不要仅因为文本出现“农田、作物、稻田、农业、增产”等农业场景词，就直接归入 `农、林、牧、渔业`；
  需要区分它是在讲农业生产活动本身，还是在讲农业相关科学技术

## 5. 电力、热力、燃气及水生产和供应业 vs 水利、环境和公共设施管理业
- 如果 topic 是电力系统、电网、发电、供热、燃气、供水、能源供给、并网、调度、保供、发电结构调整，
  优先归 `电力、热力、燃气及水生产和供应业`
- 即使文本中有白皮书、政策、改革、减排、碳中和、节目解读，也不要轻易改判
- 若主线是污染治理、污水处理、垃圾处理、生态修复、水体环境治理，才归 `水利、环境和公共设施管理业`

## 6. 住宿和餐饮业 vs 卫生和社会工作
- 有大量具体做法、步骤、配料、火候、贴士 -> `住宿和餐饮业`
- 主要是健康指导、营养建议、疾病预防、食疗建议，没有明显菜谱步骤 -> `卫生和社会工作`

## 7. 房地产业 vs 科学研究和技术服务业 vs 公共管理
- 如果文本核心在房屋开发经营、交易、租售、物业经营 -> `房地产业`
- 如果核心在房地产市场研究、空置率调查、走势分析、库存分析、发展历程研究 -> `科学研究和技术服务业`
- 如果核心在房地产政策、保障房制度、政府监管、认定审批 -> `公共管理、社会保障和社会组织`

## 8. 交通运输、仓储和邮政业 vs 建筑业 vs 公共管理
- 如果核心在交通运输服务、仓储物流、邮政快递、线路运营 -> `交通运输、仓储和邮政业`
- 如果核心在铁路、公路、轨道交通、桥梁等工程建设、开工、施工、竣工 -> `建筑业`
- 如果核心在政府交通规划、交通治理方案、交通监管政策 -> `公共管理、社会保障和社会组织`

# 五、官方文件处理规则（topic优先版）

- 不要因为是政府、党委、法院、部委发布，就默认归入 `公共管理、社会保障和社会组织`
- 先看这个文件的 topic 属于哪个行业
- 只有当文件的 topic 本身就是治理、制度、监管、认定、审批、督导、公共组织运行时，才归 `公共管理、社会保障和社会组织`

例如：
- 关于建筑业改革的意见 -> 若 topic 本质是建筑行业发展与建造体系，也可能归 `建筑业`
- 关于森林防火规划 -> 若 topic 本质是林业防火体系，可归 `农、林、牧、渔业` 或 `水利、环境和公共设施管理业`
- 关于教育督导体制改革 -> topic 本身是教育治理与督导制度，可归 `公共管理、社会保障和社会组织`

# 六、输出要求

输出一个 JSON 对象，必须包含：
- subject
- subject_type
- topic
- core_action
- reasoning
- tag
- confidence
- keywords
- runner_up
- runner_up_reason

只能输出 JSON。

要求：
- `tag` 必须是 20 个标签之一
- `confidence` 取 0 到 1 之间的小数
- `reasoning` 必须简洁说明“为什么归这个行业，而不是容易混淆的另一个行业”
- 保持口径一致：最终 tag 以 topic 所属行业领域为准
"""



@dataclass
class Task:
    """单个打标任务"""
    data: str  # 原始数据文本
    ground_truth: str = None  # 标准答案 (用于评测)


class LLMClient:
    def __init__(self, model_name: str = "mimo-v2.5-pro", api_key: str = os.getenv("MIMO_API_KEY"),
                 base_url: str = os.getenv("MIMO_BASE_URL"), temperature: float = 0):
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
    def __init__(self, llm: LLMClient, n_samples: int = 3):
        """
        参数:
            llm: LLM 客户端
        """
        self.llm = llm
        self.n_samples = n_samples

    def solve(self, task: Task) -> Dict[str, Any]:
        from collections import Counter

        # messages = list(FEW_SHOT_EXAMPLES)
        messages = list()
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
        # raw = self.llm.generate(messages, max_tokens=20000)
        # logger.info(
        #     f"模型输出 prompt is {messages} reponse is{raw} ")
        #
        # result = self._parse(raw)
        # result["_input"] = text
        # answer = result.get('tag', '')
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
def load_tasks_from_jsonL(input_path) -> List[Task]:
    input_path = Path(input_path)
    payload = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                payload.append(json.loads(line))
    result = []
    for item in payload:
        if not isinstance(item, dict):
            continue

        text = item.get("text")
        tag = item.get("tag")
        if text is None or tag is None:
            continue
        result.append(Task(data=text, ground_truth=tag))
    return result


def load_tasks_from_excel(file_path: str) -> List[Task]:
    # 1. 读取 Excel 文件
    # engine='openpyxl' 推荐用于 .xlsx 文件
    df = pd.read_excel(file_path)

    # 2. 清洗数据（可选）
    # 去除两端的空格，防止因为不可见字符导致匹配失败
    df['text'] = df['text'].astype(str).str.strip()
    df['tag'] = df['tag'].astype(str).str.strip()

    # 3. 将 DataFrame 行转换为 Task 对象列表
    tasks = [
        Task(data=row['text'], ground_truth=row['tag'])
        for _, row in df.iterrows()
    ]

    return tasks


def run_evaluation(solver: TagSolveSolver, tasks: List[Task], output_file="./output/result.xlsx"):
    results = []
    correct_count = 0
    total_count = len(tasks)
    total_start_time = datetime.now()
    logger.info(f"开始评测，共 {total_count} 条数据...")
    for i, task in enumerate(tasks):
        start_time = datetime.now()
        logger.info(f"正在处理第 {i + 1}/{total_count} 条...")
        logger.info("=" * 70)
        # 调用你的 solve 方法
        res = solver.solve(task)
        # logger.info(f"最终结果为 currect {res['correct']}= tag{res['ground_truth']}====》{res}")
        results.append(res)

        if res['correct']:
            correct_count += 1
        latency = datetime.now() - start_time
        current_accuracy = correct_count / (i + 1) * 100

        logger.info(
            f""
            f"->正确记录 {correct_count} 总数为 {i + 1} 错误记录数 {i + 1 - correct_count} 当前准确率{current_accuracy:.2f}")
        logger.info(f"正确的tag应该是{res['ground_truth']}")
        logger.info(f"模型预测tag是{res['predicted_answer']}")
        logger.info(f"具体reason是{res['reasoning']}")
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

    task_list = load_tasks_from_excel('./data/标签数据集.xlsx')
    # task_list = load_tasks_from_jsonL('./data/industry_top100.jsonl')

    llm_client = LLMClient(model_name="openai/gpt-5.4-nano", api_key=os.getenv("OPENROUTER_API_KEY"),
                           base_url=os.getenv("OPENROUTER_BASE_URL"))
    solver = TagSolveSolver(llm_client)
    run_evaluation(solver, task_list, './output/result-v4.xlsx')
