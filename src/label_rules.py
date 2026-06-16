"""Rule-based weak labeling for Chinese psychological dialogue data.

SoulChatCorpus is a large dialogue corpus, but the public release is primarily an
instruction/dialogue corpus rather than a ready-made topic-classification table.
For a course machine-learning project, we create transparent weak labels from
psychological-help-seeking keywords. These labels are only used for educational
classification experiments and UI routing; they are not clinical diagnoses.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, Iterable, List, Tuple

LABELS_ZH: Dict[str, str] = {
    "high_risk": "高风险求助/危机干预",
    "relationship": "亲密关系与失恋",
    "study_work": "学习/工作压力",
    "family": "家庭关系",
    "interpersonal": "人际与社交",
    "emotion": "情绪困扰/焦虑抑郁",
    "sleep_body": "睡眠与身心症状",
    "self_growth": "自我成长/自我评价",
    "other": "其他心理支持",
}

KEYWORDS: Dict[str, List[str]] = {
    "high_risk": [
        "自杀", "想死", "不想活", "活不下去", "结束生命", "轻生", "割腕", "跳楼",
        "消失", "死了算了", "没有活着的意义", "伤害自己", "自残", "遗书",
    ],
    "relationship": [
        "失恋", "分手", "男朋友", "女朋友", "恋爱", "爱情", "暧昧", "前任", "婚姻", "离婚",
        "出轨", "喜欢的人", "伴侣", "对象",
    ],
    "study_work": [
        "考试", "考研", "高考", "作业", "论文", "绩点", "老师", "同学", "上课", "学习",
        "工作", "老板", "职场", "同事", "加班", "面试", "实习", "就业", "科研", "毕业",
    ],
    "family": [
        "父母", "爸爸", "妈妈", "家庭", "家里", "亲戚", "孩子", "儿子", "女儿", "婆婆", "公婆",
        "原生家庭", "爸妈",
    ],
    "interpersonal": [
        "朋友", "室友", "同学", "同事", "社交", "人际", "孤独", "被孤立", "冷暴力", "沟通",
        "关系", "吵架", "误会",
    ],
    "emotion": [
        "焦虑", "抑郁", "难过", "崩溃", "压抑", "恐惧", "害怕", "内耗", "烦躁", "痛苦",
        "情绪", "哭", "绝望", "担心", "紧张", "空虚", "无助", "委屈",
    ],
    "sleep_body": [
        "睡不着", "失眠", "早醒", "做噩梦", "心慌", "胸闷", "头痛", "胃痛", "食欲", "暴食",
        "厌食", "疲惫", "身体", "生病", "睡眠",
    ],
    "self_growth": [
        "自卑", "没用", "价值", "意义", "未来", "人生", "成长", "性格", "自信", "自我",
        "拖延", "目标", "选择", "迷茫", "完美主义",
    ],
}

PRIORITY = [
    "high_risk",
    "relationship",
    "study_work",
    "family",
    "interpersonal",
    "emotion",
    "sleep_body",
    "self_growth",
]

@dataclass(frozen=True)
class LabelResult:
    label: str
    risk_level: str
    risk_keyword_count: int
    matched_keywords: str


def normalize_text(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def count_keywords(text: str, words: Iterable[str]) -> Tuple[int, List[str]]:
    matched = []
    count = 0
    for w in words:
        c = text.count(w)
        if c > 0:
            count += c
            matched.append(w)
    return count, matched


def weak_label(text: str) -> LabelResult:
    text = normalize_text(text)
    all_matches: List[str] = []
    counts: Dict[str, int] = {}
    for label, words in KEYWORDS.items():
        c, m = count_keywords(text, words)
        counts[label] = c
        all_matches.extend(m)

    if counts.get("high_risk", 0) > 0:
        return LabelResult("high_risk", "high", counts["high_risk"], ",".join(sorted(set(all_matches))))

    # Pick the label with most keyword hits; use a priority order to break ties.
    best_label = "other"
    best_score = 0
    for label in PRIORITY[1:]:
        score = counts.get(label, 0)
        if score > best_score:
            best_label = label
            best_score = score

    risk_level = "medium" if any(x in text for x in ["绝望", "崩溃", "活着", "撑不住", "快受不了"]) else "low"
    return LabelResult(best_label if best_score > 0 else "other", risk_level, counts.get("high_risk", 0), ",".join(sorted(set(all_matches))))


def label_display(label: str) -> str:
    return LABELS_ZH.get(label, label)
