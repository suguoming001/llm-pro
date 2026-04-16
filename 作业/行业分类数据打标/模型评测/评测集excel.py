import logging

import pandas as pd
from modelscope.msdatasets import MsDataset

TARGET_CATEGORIES = ['anatomy', 'astronomy', 'college_biology']
SAMPLES_PER_CATEGORY = 20


def load_and_sample_mmlu():
    dataset = MsDataset.load('cais/mmlu', subset_name='all', split='test')
    df = dataset.to_pandas()
    sampled_dfs = []
    for cat in TARGET_CATEGORIES:
        cat_df = df[df['subject'] == cat]
        if len(cat_df) >= SAMPLES_PER_CATEGORY:
            sampled_dfs.append(cat_df.sample(SAMPLES_PER_CATEGORY, random_state=42))
        else:
            logging.warning(f"分类 {cat} 数据不足 {SAMPLES_PER_CATEGORY} 条！")
            sampled_dfs.append(cat_df)
    eval_df = pd.concat(sampled_dfs).reset_index(drop=True)
    eval_df['choices'] = eval_df['choices'].apply(lambda x: str(list(x)) if hasattr(x, '__iter__') else x)
    eval_df.to_excel('mmlu_evaluation_dataset.xlsx', index=False, engine='openpyxl')


if __name__ == '__main__':
    load_and_sample_mmlu()
