import ast
import os
import re
import time

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

COLLEGE_BIOLOGY_FEW_SHOT = r"""
问题1:Approximately what fraction of the human genome encodes proteins?
A.2%
B.25%
C.50%
D.90%
答案:A
分析: While about 75% of the human genome is transcribed into RNA, only about 1.5% to 2% consists of protein-coding exons. The remainder consists of non-coding DNA, including introns, regulatory sequences, and repetitive elements (transposons).
--
问题2:Which of the following adaptations would limit pollination by bees and promote hummingbird pollination?
A.Patterns of ultraviolet color on the petals
B.Modified petals to provide a landing space
C.Pendant (hanging) red-colored flowers
D.Nectar with high sugar concentration produced in limited amounts
答案:C
分析:Bees are attracted to UV patterns (A) and require landing platforms (B). Hummingbirds, however, have excellent red vision and can hover, making pendant (hanging) red flowers ideal for them but difficult for bees. Additionally, hummingbirds require large volumes of nectar, unlike the limited amounts mentioned in option D.
--
问题3:A species of goose nests on both cliffs and beaches near the ocean. Soon after hatching, all chicks must make their way to the ocean. Chicks from cliff nests must tumble down the cliff to get to the ocean, and many are killed by the fall. Which of the following is most consistent with the hypothesis that cliff nesting is adaptive in this goose species?
A.Many more geese nest on the beaches than on the cliffs.
B.Cliff-side nesting confers a higher fitness than does beach nesting.
C.Chicks from cliff nests instinctively step off the cliffs at the appropriate time.
D.More chicks survive the fall from the cliffs than are killed.
答案:B
分析:In evolutionary terms, a trait is "adaptive" if it increases an individual's fitness (survival and reproductive success) relative to alternative traits. Even if the fall is dangerous, cliff nesting is adaptive if the total fitness (e.g., due to lower predation) is higher than that of beach nesting.
--
问题4:Cell motility, as viewed in a cultured fibroblast, encompasses all of the following EXCEPT
A.adhesion plaques
B.vinculin
C.clathrin
D.lamellipodia
答案：C
分析:Cell motility involves the actin cytoskeleton: lamellipodia are the leading edge protrusions, and adhesion plaques (containing proteins like vinculin) anchor the cell to the substrate. Clathrin, however, is involved in receptor-mediated endocytosis (vesicle formation), not direct cell movement.
--
问题5:During the mammalian cardiac cycle, a volume of blood equivalent to ventricular stroke volume is transferred from the more compliant venous side to the less compliant arterial side of the circulation. In terms of pressures within the venous and arterial compartments, this transfer results in
A.no change in pressure in either compartment
B.no effect on venous pressure and a small increase in arterial pressure
C.an increase in venous pressure and an equal but opposite decrease in arterial pressure
D.little effect on venous pressure and a large increase in arterial pressure
答案:D
分析:Compliance ($C$) is defined as $\Delta V / \Delta P$. The venous system is highly compliant ("capacitance vessels"), meaning large volume changes cause minimal pressure changes. The arterial system has low compliance; therefore, transferring the stroke volume into the arteries causes a large increase in arterial pressure (systolic pressure) while having little effect on venous pressure.
--
"""

ASTRONOMY_FEW_SHOT = """
问题1:The nebular theory of the formation of the solar system successfully predicts all but one of the following. Which one does the theory not predict?
A.Planets orbit around the Sun in nearly circular orbits in a flattened disk.
B.the equal number of terrestrial and jovian planets
C.the craters on the Moon
D.asteroids Kuiper-belt comets and the Oort cloud
答案:B
分析: The nebular theory explains the common direction of orbits and the flattened disk shape of the solar system due to conservation of angular momentum. However, it does not mandate an equal number of terrestrial and jovian planets; the number of planets is determined by the specific distribution of mass and temperature within the original solar nebula.
--
问题2:Find the best approximation for the surface temperature of the Sun:
A.6000 K
B.7000 K
C.9000 K
D.13000 K
答案:A
分析:The Sun’s effective surface temperature (the photosphere) is approximately 5,800 K (commonly rounded to 6,000 K). This temperature determines the Sun's spectral class (G2V) and its peak emission in the visible light spectrum.
--
问题3:The two moons of Mars are called ...
A.Tritos and Desmos
B.Tritos and Deimos
C.Phobos and Tritos
D.Phobos and Deimos
答案:D
分析:Mars has two small, irregularly shaped moons named Phobos (fear) and Deimos (panic). "Tritos" is not a moon of Mars (Triton is a moon of Neptune).
--
问题4:Earth has been gradually warming over the past few decades. Based on a great deal of evidence scientists conclude that this warming is caused by ________.
A.human activities that are increasing the concentration of greenhouse gases in Earth's atmosphere
B.the fact that our politicians spout a lot of hot air
C.the human release of chemicals called CFCs into the stratosphere
D.the increase in forest fires during recent years
答案：A
分析:Global warming is primarily driven by the enhanced greenhouse effect. Human activities, particularly the burning of fossil fuels and deforestation, increase the atmospheric concentration of $CO_2$ and $CH_4$, which trap more thermal infrared radiation.
--
问题5:From shortest to longest wavelength which of the following correctly orders the different categories of electromagnetic radiation?
A.infrared visible light ultraviolet X rays gamma rays radio
B.radio infrared visible light ultraviolet X rays gamma rays
C.gamma rays X rays visible light ultraviolet infrared radio
D.gamma rays X rays ultraviolet visible light infrared radio
答案:D
分析:In the electromagnetic spectrum, energy/frequency decreases as wavelength increases. The correct order from shortest to longest wavelength is: Gamma rays < X-rays < Ultraviolet < Visible light < Infrared < Radio.
--
"""

ANATOMY_FEW_SHOT = """
问题1:Macrostomia results from failure of fusion of
A.maxillary and mandibular processes.
B.left and right mandibular processes.
C.maxillary and frontonasal processes.
D.mandibular and hyoid arches.
答案:A
分析: Macrostomia (large mouth) occurs when there is incomplete lateral fusion between the maxillary and mandibular processes, which normally determines the width of the mouth.
--
问题2:In the brain stem, pathways for:
A.nociception decussate in the medial lemniscus
B.skilled movements decussate in the medial lemniscus
C.skilled motor movements decussate in the pyramids.
D.discriminative touch decussate in the pyramids.
答案:C
分析:Skilled motor movements are carried by the corticospinal (pyramidal) tract, which decussates (crosses over) in the medullary pyramids. Discriminative touch decussates as internal arcuate fibers to form the medial lemniscus, not in the pyramids.
--
问题3:The head of the sixth rib articulates with
A.The inferior articular facet of T5 and superior articular facet of T6.
B.The inferior articular demifacet of T5 and superior articular facet of T6.
C.The inferior articular demifacet of T5 and the superior articular demifacet of T6.
D.The superior and inferior demifacets of T6.
答案:C
分析:A typical rib head articulates with the superior demifacet of its own vertebrae (T6) and the inferior demifacet of the vertebra above it (T5).
--
问题4:Which of the following allows air to pass into the lungs?
A.Aorta
B.Esophagus
C.Trachea
D.Pancreas
答案：C
分析:The trachea (windpipe) is the primary cartilaginous tube that connects the larynx to the bronchi, allowing the passage of air into the lungs. The aorta carries blood, the esophagus carries food, and the pancreas is a digestive/endocrine organ.
--
问题5:Which of the following is the large bone found superior to the patella and inferior to the ischium?
A.Calcaneus
B.Femur
C.Symphysis pubis
D.Tibia
答案:B
分析:The femur (thigh bone) is the longest bone in the body. It is positioned inferior to the ischium (part of the hip bone/pelvis) and superior to the patella (kneecap). The tibia is inferior to the patella, and the calcaneus is the heel bone.
--
"""

few_shot_dict = {
    "anatomy": ANATOMY_FEW_SHOT,
    "college_biology": COLLEGE_BIOLOGY_FEW_SHOT,
    "astronomy": ASTRONOMY_FEW_SHOT
}


def call_llm(model_name, system_prompt, prompt):
    config = MODEL_CONFIGS[model_name]
    client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
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


dataset = pd.read_excel('mmlu_evaluation_dataset.xlsx')
scores = {model: {method: 0 for method in methods} for model in MODEL_CONFIGS.keys()}
total_questions = len(dataset)
questions = [build_question(row) for index, row in dataset.iterrows()]

results = format_question(questions)


def build_prompt(method, result):
    few_shot = ""
    if "few-shot" in method:
        subject = result['subject']
        few_shot = few_shot_dict[subject]

    system_prompt = f"""请直接给出正确选项的字母，不要任何其他的文字说明。\n{few_shot}"""
    prompt = "请直接给出正确选项的字母，不要任何其他的文字说明。\n"+result["full_question"]

    if 'cot' in method:
        prompt += """ \n Let's think step by step. """
    return system_prompt, prompt


for index, result in enumerate(results):
    print('-'*20)
    print(f"问题{index + 1}, 分类：{result['subject']}")
    print(result["full_question"])
    correct_answer = result["correct_answer"]
    for method in methods:
        system_prompt, prompt = build_prompt(method, result)
        print("system_prompt is ⭐️⭐️⭐️", system_prompt)
        print("prompt is ⭐️⭐️⭐️", prompt)
        for model in MODEL_CONFIGS.keys():
            print(f"[{model}] | 模式: {method} | 第 {index + 1}/{total_questions} 题 | 调用中...")
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