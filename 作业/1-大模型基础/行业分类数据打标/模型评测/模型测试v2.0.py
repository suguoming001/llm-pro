import ast
import os
import re
import time
from collections import defaultdict

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

MODEL_CONFIGS = {
    "mimo-v2-pro": {"api_key": os.getenv("MIMO_API_KEY"), "base_url": os.getenv("MIMO_BASE_URL")},
    "deepseek-chat": {"api_key": os.getenv("DEEPSEEK_API_KEY"), "base_url": os.getenv("DEEPSEEK_BASE_URL")},
    "openai/gpt-5-nano": {"api_key": os.getenv("OPENROUTER_API_KEY"), "base_url": os.getenv("OPENROUTER_BASE_URL")}
}
methods = ["zero-shot", "zero-shot-cot", "few-shot", "few-shot-cot"]
answer_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G'}

FEW_SHOT_NUM = 5


def call_llm(model_name, system_prompt, prompt):
    config = MODEL_CONFIGS[model_name]
    client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return str(e)


def build_question(row):
    question = dict()
    question['question'] = row['question']
    choices = row['choices']
    if isinstance(choices, str):
        choices = ast.literal_eval(choices)
    question['options'] = [f"{answer_map[i]}.{choice}" for i, choice in enumerate(choices)]
    question['answer'] = answer_map[int(row['answer'])]
    question['subject'] = row['subject']
    return question


def format_question(questions):
    results = []
    for question in questions:
        full_question = question["question"] + "\n" + "\n".join(question["options"])
        right_answer = question["answer"]
        subject = question["subject"]
        results.append({
            "full_question": full_question,
            "correct_answer": right_answer,
            "subject": subject
        })
    return results


def extract_answer(text):
    print("模型返回结果是", text)
    """
    针对 CoT 模式的增强版答案提取逻辑。
    优先匹配末尾的 'The answer is X'，或者全文最后一个出现的 A/B/C/D。
    """
    if len(text) <= 5:  # 如果返回很短（如直接返回 'A'），直接清理返回
        match = re.search(r'[A-D]', text.upper())
        return match.group(0) if match else text

    # 查找常见结论短语
    patterns = [
        r"answer is ([A-D])",
        r"答案是\s*([A-D])",
        r"选项\s*([A-D])",
        r"conclusion is ([A-D])"
    ]
    for p in patterns:
        match = re.search(p, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # 如果都没匹配到，取全文最后一个出现的字母
    letters = re.findall(r'[A-D]', text.upper())
    return letters[-1] if letters else text


def group_questions_by_subject(result):
    subject_map = defaultdict(list)
    for item in result:
        subject_map[item["subject"]].append(item)
    return subject_map


def get_simple_per_subject_list(results, n):
    subject_map = defaultdict(list)

    for q in results:
        subj = q["subject"]
        if len(subject_map[subj]) >= n:
            continue
        subject_map[subj].append(q)
    return dict(subject_map)


def build_few_shot(results, subject):
    subject_map = get_simple_per_subject_list(results, FEW_SHOT_NUM)
    question_list = subject_map[subject]

    shot_list = [f"Question:{question["full_question"]}\n\n  Answer:{question["correct_answer"]}" for question in
                 question_list]
    return "\n\n".join(shot_list)


def build_prompt(method, results, result, model=None):
    few_shot = ""
    cot_prompt = ""
    if "few-shot" in method:
        subject = result['subject']
        few_shot = build_few_shot(results, subject)

    system_prompt = f"""请直接给出正确选项的字母，不要任何其他的文字说明。\n"""

    question_prompt = result["full_question"]
    prompt = fr"""
    请直接给出正确选项的字母，不要任何其他的文字说明。
    {few_shot} 
    {cot_prompt} 
    {question_prompt}
    """
    if 'cot' in method:
        prompt = fr"""
        {few_shot} 
        {question_prompt} 
        Let's think step by step.
        """
        system_prompt = ''
        reasoning_path = call_llm(model, system_prompt, prompt)
        prompt = fr"""
               {few_shot} 
               {question_prompt} 
               Let's think step by step.
               {reasoning_path}\
               Therefore, the answer is\
               请直接给出正确选项的字母，不要任何其他的文字说明。
               """
    return system_prompt, prompt


dataset = pd.read_excel('mmlu_evaluation_dataset.xlsx')
scores = {model: {method: 0 for method in methods} for model in MODEL_CONFIGS.keys()}
total_questions = len(dataset)
questions = [build_question(row) for index, row in dataset.iterrows()]

results = format_question(questions)

for index, result in enumerate(results):
    print('-' * 20)
    print(f"问题{index + 1}, 分类：{result['subject']}")
    print(result["full_question"])
    correct_answer = result["correct_answer"]
    for method in methods:
        system_prompt=''
        prompt=''
        if 'cot' not in method:
            system_prompt, prompt = build_prompt(method, results, result)
            print("system_prompt is ⭐️⭐️⭐️", system_prompt)
            print("prompt is ⭐️⭐️⭐️", prompt)
        for model in MODEL_CONFIGS.keys():
            print(f"[{model}] | 模式: {method} | 第 {index + 1}/{total_questions} 题 | 调用中...")
            if 'cot' in method:
                system_prompt, prompt = build_prompt(method, results, result, model)
                print("system_prompt is ⭐️⭐️⭐️", system_prompt)
                print("prompt is ⭐️⭐️⭐️", prompt)
            start_time = time.time()
            raw_output = call_llm(model, system_prompt, prompt)
            prediction = extract_answer(raw_output)
            latency = time.time() - start_time
            is_correct = (prediction.strip() == correct_answer.strip())
            if is_correct:
                scores[model][method] += 1
            print(
                f"  -> 耗时: {latency:.2f}s | 格式化模型输出: {prediction} | 正确答案: {correct_answer} | {'✅' if is_correct else '❌'}")
    print('-' * 20)
# --- 打印最终报表 ---
print("\n" + "=" * 60)
print(" 最终评测得分汇总 (准确率 %)")
print("=" * 60)

scores_list = []
print(scores)
for model in MODEL_CONFIGS.keys():
    row_data = {"Model": model}
    for method in methods:
        accuracy = (scores[model][method] / total_questions) * 100
        row_data[method] = f"{accuracy:.2f}%"
    scores_list.append(row_data)
print(scores_list)
score_df = pd.DataFrame(scores_list)
score_df.set_index("Model", inplace=True)
print(score_df.to_markdown())
