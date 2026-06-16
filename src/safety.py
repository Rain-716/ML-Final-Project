from __future__ import annotations

from .label_rules import KEYWORDS, count_keywords

CRISIS_MESSAGE = (
    "我很担心你现在的安全。这个系统不能替代心理咨询师、医生或紧急救援。"
    "请先不要独自待着，马上联系身边可信任的人；如果你已经有伤害自己的计划、工具或冲动，"
    "请立刻拨打当地紧急电话，或前往最近医院急诊/精神心理科寻求帮助。"
)

DISCLAIMER = (
    "提示：本项目用于课程展示与机器学习实验，只提供情绪支持与信息整理，不提供诊断，"
    "不能替代专业心理咨询、医疗诊疗或紧急救援。"
)


def detect_crisis(text: str) -> bool:
    count, _ = count_keywords(str(text or ""), KEYWORDS["high_risk"])
    return count > 0


def safety_prefix(text: str, predicted_label: str | None = None) -> str:
    if predicted_label == "high_risk" or detect_crisis(text):
        return CRISIS_MESSAGE
    return DISCLAIMER
