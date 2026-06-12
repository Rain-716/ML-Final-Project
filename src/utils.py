import re
import pandas as pd
from pathlib import Path

CRISIS_KEYWORDS = [
    'suicide','kill myself','hurt myself','end my life','not worth living',
    '自杀','轻生','伤害自己','不想活','结束生命'
]

def clean_text(text: str) -> str:
    text = '' if pd.isna(text) else str(text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'[^A-Za-z0-9\u4e00-\u9fa5\s\'\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

def is_crisis(text: str) -> bool:
    t = str(text).lower()
    return any(k in t for k in CRISIS_KEYWORDS)

def load_dataset(data_source='sample', max_samples=8000):
    if data_source == 'sample':
        path = Path(__file__).resolve().parents[1] / 'data' / 'sample_dialogue.csv'
        df = pd.read_csv(path)
    elif data_source == 'empathetic':
        from datasets import load_dataset
        ds = load_dataset('facebook/empathetic_dialogues')
        raw = ds['train'].to_pandas()
        # EmpatheticDialogues字段：utterance, context(情绪), conv_id等。这里用utterance预测context。
        df = raw[['utterance','context']].rename(columns={'utterance':'text','context':'label'})
        df['response'] = '我理解你的感受。可以先慢下来描述发生了什么，再选择一个小的可执行行动；如情况严重，请联系专业心理咨询。'
        df = df.sample(min(max_samples, len(df)), random_state=42)
    else:
        raise ValueError('data_source must be sample or empathetic')
    df = df.dropna(subset=['text','label']).drop_duplicates(subset=['text'])
    df['text_clean'] = df['text'].apply(clean_text)
    df = df[df['text_clean'].str.len() > 2]
    return df.reset_index(drop=True)

def safe_response(user_text, pred_label, retrieved_response=None):
    if is_crisis(user_text):
        return ('crisis', '我很担心你的安全。请立刻联系当地紧急服务、校园心理危机热线或身边可信任的人，并尽量不要独处。本系统不能替代专业危机干预。')
    return pred_label, retrieved_response or '谢谢你愿意分享。建议先把感受具体说出来，并做一个最小可执行行动。'
