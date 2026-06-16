from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gradio as gr
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_DIR, OUTPUT_DIR, LABEL_COLUMN, RESPONSE_COLUMN, TEXT_COLUMN  # noqa: E402
from src.predict import Predictor  # noqa: E402
from src.retrieval import ResponseRetriever  # noqa: E402
from src.safety import CRISIS_MESSAGE, DISCLAIMER, detect_crisis, safety_prefix  # noqa: E402
from src.label_rules import label_display  # noqa: E402

# ------------------------------------------------------------
# Global loading
# ------------------------------------------------------------

predictor = Predictor(MODEL_DIR)
retriever = None
index_path = MODEL_DIR / "response_retrieval.joblib"
if index_path.exists():
    retriever = ResponseRetriever(index_path)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_TIPS: Dict[str, str] = {
    "emotion": "情绪压力、焦虑、低落、委屈等一般情绪困扰。重点是先识别情绪，再把问题拆小。",
    "family": "家庭关系、亲子沟通、家庭压力相关。重点是边界、沟通和支持资源。",
    "high_risk": "高风险表达，可能涉及自伤、轻生或严重危机。系统会优先给出安全求助提醒。",
    "interpersonal": "同学、朋友、室友、同事等人际关系困扰。重点是事实、感受、需求分开表达。",
    "other": "暂未明显落入单一主题，可能是综合性困扰。适合继续追问背景和具体感受。",
    "relationship": "恋爱、分手、亲密关系相关。重点是接纳情绪、恢复生活节奏和降低反复内耗。",
    "self_growth": "自我成长、自我价值、迷茫、拖延、自信相关。重点是小目标和可执行行动。",
    "sleep_body": "睡眠、躯体不适、疲惫、食欲等身心状态相关。重点是生活节律和必要时求医。",
    "study_work": "学习、考试、工作、职业压力相关。重点是任务拆解、计划和压力管理。",
}

TEMPLATE_REPLIES: Dict[str, str] = {
    "emotion": "听起来你现在承受了不少情绪压力。可以先不用急着把所有问题一次解决，我们先把它拆成三步：第一，给现在的情绪命名；第二，找出最让你难受的一个触发点；第三，选一个今天能做的小动作，比如散步十分钟、写下担心清单，或找一个信任的人说几句。",
    "family": "家庭里的压力常常会让人既在意又无力。你可以先区分两件事：哪些是你能表达和调整的，哪些暂时不是你能控制的。下一步可以尝试用“我感到……因为……我希望……”的方式表达，而不是直接争对错。",
    "interpersonal": "人际关系让你不舒服时，先别急着否定自己。可以把事件拆成：对方具体做了什么、你当时有什么感受、你真正希望关系怎样变化。这样会更容易找到沟通的切入口。",
    "relationship": "亲密关系里的失落会很消耗人。你现在的难受不是矫情，而是在经历关系变化后的正常反应。可以先给自己一点恢复空间，减少反复追问“是不是我不够好”，把注意力慢慢拉回睡眠、饮食、学习和朋友支持。",
    "self_growth": "迷茫和自我怀疑通常不是说明你不行，而是说明你正在面对变化。先不要把目标定得太大，可以从一个很小的行动开始，比如今天只完成一件能让你恢复掌控感的事。",
    "sleep_body": "睡眠和身体状态会强烈影响情绪。你可以先观察最近的作息、饮食、运动和压力源。如果持续失眠、胸闷、心慌或明显影响生活，建议及时咨询医生或学校心理中心。",
    "study_work": "学习或工作压力大时，大脑很容易进入“越焦虑越做不动”的循环。可以把任务拆成 25 分钟一组，只盯住下一步，而不是同时想着所有后果。",
    "other": "我听到了你的困扰。你可以先把事情分成三部分说：发生了什么、你现在最强烈的感受是什么、你最希望别人怎样支持你。我们可以慢一点梳理。",
}

CSS = """
.gradio-container {max-width: 1240px !important; margin: auto;}
#title h1 {font-size: 2.15rem; margin-bottom: 0.25rem;}
#title p {font-size: 1rem; color: #475569;}
.badge {display:inline-block; padding:4px 10px; border-radius:999px; background:#eef2ff; margin:4px 6px 4px 0; font-size:0.92rem;}
.badge-risk {background:#fee2e2; color:#991b1b; font-weight:600;}
.safe-card {padding:12px 14px; border-radius:14px; background:#f8fafc; border:1px solid #e2e8f0; margin-bottom:10px;}
.tip-card {padding:12px 14px; border-radius:14px; background:#fff7ed; border:1px solid #fed7aa;}
.small-note {color:#64748b; font-size:0.92rem;}
.footer-note {font-size:0.88rem; color:#64748b; line-height:1.5;}
"""

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def clean_text(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_history(history: Any) -> List[Dict[str, str]]:
    """Convert legacy tuple history or new messages history to Gradio messages format.

    This avoids the Gradio error:
    "Data incompatible with messages format... role and content keys".
    """
    if not history:
        return []

    normalized: List[Dict[str, str]] = []
    for item in history:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")
            if role in {"user", "assistant", "system"} and content is not None:
                normalized.append({"role": str(role), "content": str(content)})
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            user_msg, assistant_msg = item
            if user_msg:
                normalized.append({"role": "user", "content": str(user_msg)})
            if assistant_msg:
                normalized.append({"role": "assistant", "content": str(assistant_msg)})
    return normalized


def model_status_markdown() -> str:
    model_badge = f"<span class='badge'>当前分类模型：{predictor.kind}</span>"
    retrieval_badge = (
        "<span class='badge'>检索索引：已加载</span>"
        if retriever is not None
        else "<span class='badge badge-risk'>检索索引：未加载</span>"
    )
    safety_badge = "<span class='badge'>危机保护：开启</span>"
    return f"<div class='safe-card'>{model_badge}{retrieval_badge}{safety_badge}</div>"


def prediction_to_json(pred: Dict[str, Any]) -> Dict[str, Any]:
    label = pred.get("label", "other")
    return {
        "label": label,
        "label_zh": pred.get("label_zh", label_display(label)),
        "confidence": round(float(pred.get("confidence", 0.0)), 4),
        "model_kind": pred.get("kind", predictor.kind),
        "explanation": LABEL_TIPS.get(label, LABEL_TIPS["other"]),
    }


def prediction_markdown(pred: Dict[str, Any], user_text: str) -> str:
    label = pred.get("label", "other")
    label_zh = pred.get("label_zh", label_display(label))
    conf = float(pred.get("confidence", 0.0))
    kind = pred.get("kind", predictor.kind)
    risk_class = " badge-risk" if label == "high_risk" or detect_crisis(user_text) else ""
    return (
        "<div class='safe-card'>"
        f"<span class='badge{risk_class}'>预测类别：{label_zh}</span>"
        f"<span class='badge'>置信度：{conf:.2f}</span>"
        f"<span class='badge'>模型：{kind}</span>"
        f"<p class='small-note'>{LABEL_TIPS.get(label, LABEL_TIPS['other'])}</p>"
        "</div>"
    )


def retrieve_topk(query: str, label: str | None, top_k: int = 3) -> Tuple[pd.DataFrame, str, float, str]:
    """Return a dataframe for UI plus the best response.

    Returns: table, best_response, best_score, source_note
    """
    columns = ["rank", "score", "label", "source_text", "retrieved_response"]
    if retriever is None:
        return pd.DataFrame(columns=columns), TEMPLATE_REPLIES.get(label or "other", TEMPLATE_REPLIES["other"]), 0.0, "未加载检索索引，使用模板回复。"

    query = clean_text(query)
    if not query:
        return pd.DataFrame(columns=columns), TEMPLATE_REPLIES["other"], 0.0, "输入为空，使用默认模板。"

    try:
        qv = retriever.vectorizer.transform([query])
        candidate_idx = np.arange(len(retriever.df))
        if label and LABEL_COLUMN in retriever.df.columns:
            same = np.where(retriever.df[LABEL_COLUMN].astype(str).values == str(label))[0]
            if len(same) >= max(10, int(top_k)):
                candidate_idx = same

        scores = cosine_similarity(qv, retriever.matrix[candidate_idx]).ravel()
        if scores.size == 0:
            return pd.DataFrame(columns=columns), TEMPLATE_REPLIES.get(label or "other", TEMPLATE_REPLIES["other"]), 0.0, "没有检索到候选回复，使用模板回复。"

        order = np.argsort(scores)[::-1][: max(1, int(top_k))]
        rows = []
        for rank, local_i in enumerate(order, start=1):
            idx = int(candidate_idx[int(local_i)])
            row = retriever.df.iloc[idx]
            rows.append(
                {
                    "rank": rank,
                    "score": round(float(scores[int(local_i)]), 4),
                    "label": str(row.get(LABEL_COLUMN, "")),
                    "source_text": str(row.get(TEXT_COLUMN, ""))[:120],
                    "retrieved_response": str(row.get(RESPONSE_COLUMN, ""))[:180],
                }
            )

        table = pd.DataFrame(rows, columns=columns)
        best = rows[0]
        best_response = str(best["retrieved_response"])
        best_score = float(best["score"])
        note = f"已从训练集检索相似样本，Top-1 相似度 {best_score:.3f}。"
        return table, best_response, best_score, note
    except Exception as exc:  # Keep UI from crashing during live demo.
        return pd.DataFrame(columns=columns), TEMPLATE_REPLIES.get(label or "other", TEMPLATE_REPLIES["other"]), 0.0, f"检索失败，使用模板回复。错误：{exc}"


def compose_reply(user_text: str, pred: Dict[str, Any], use_retrieval: bool, top_k: int, style: str) -> Tuple[str, pd.DataFrame, str]:
    label = str(pred.get("label", "other"))
    label_zh = str(pred.get("label_zh", label_display(label)))
    prefix = safety_prefix(user_text, label)

    if label == "high_risk" or detect_crisis(user_text):
        body = (
            "我会优先关注你的安全。请尽量不要一个人承受这件事，先联系身边可信任的人、学校心理中心、医院急诊或当地紧急电话。"
            "如果你已经有具体计划或冲动，请立刻离开危险物品并寻求线下帮助。"
        )
        table = pd.DataFrame(columns=["rank", "score", "label", "source_text", "retrieved_response"])
        source_note = "检测到高风险表达，已跳过历史回复检索，优先显示安全提示。"
    elif use_retrieval:
        table, retrieved_body, score, source_note = retrieve_topk(user_text, label, top_k=top_k)
        # If the retrieved score is too low, blend with template to avoid irrelevant answers.
        if score < 0.08:
            body = TEMPLATE_REPLIES.get(label, TEMPLATE_REPLIES["other"])
            source_note += " 相似度偏低，最终采用更稳妥的模板回复。"
        else:
            body = retrieved_body
    else:
        body = TEMPLATE_REPLIES.get(label, TEMPLATE_REPLIES["other"])
        table = pd.DataFrame(columns=["rank", "score", "label", "source_text", "retrieved_response"])
        source_note = "当前关闭检索，使用分类模板回复。"

    if style == "更温柔":
        body = "先抱抱你。" + body
    elif style == "更行动导向":
        body = body + "\n\n你现在可以先做一个很小的动作：写下此刻最困扰你的一个点，并给它标一个 0-10 的压力分数。"
    elif style == "更适合答辩展示":
        body = body + f"\n\n【系统演示说明】模型先将输入识别为“{label_zh}”，再结合安全规则和检索模块生成回复。"

    reply = (
        f"{prediction_markdown(pred, user_text)}\n\n"
        f"**安全提示**：{prefix}\n\n"
        f"**AI 支持回复**：{body}\n\n"
        f"> {source_note}\n\n"
        "<p class='footer-note'>注意：本系统是机器学习课程 Demo，只用于文本分类、相似回复检索和情绪支持展示，不能替代心理咨询师、医生或紧急救援。</p>"
    )
    return reply, table, source_note


# ------------------------------------------------------------
# Gradio callbacks
# ------------------------------------------------------------

def build_reply(message: str, history: Any, use_retrieval: bool, top_k: int, style: str):
    history = normalize_history(history)
    message = clean_text(message)
    if not message:
        status = "<div class='tip-card'>请输入一段文字后再发送。</div>"
        return "", history, status, {}, pd.DataFrame(columns=["rank", "score", "label", "source_text", "retrieved_response"])

    pred = predictor.predict(message)
    reply, table, _ = compose_reply(message, pred, bool(use_retrieval), int(top_k), str(style))
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})
    return "", history, prediction_markdown(pred, message), prediction_to_json(pred), table


def analyze_only(message: str, use_retrieval: bool, top_k: int):
    message = clean_text(message)
    if not message:
        return "<div class='tip-card'>请输入需要分析的文本。</div>", {}, pd.DataFrame(columns=["rank", "score", "label", "source_text", "retrieved_response"])
    pred = predictor.predict(message)
    label = str(pred.get("label", "other"))
    table, _, _, _ = retrieve_topk(message, label, top_k=top_k) if use_retrieval else (pd.DataFrame(columns=["rank", "score", "label", "source_text", "retrieved_response"]), "", 0.0, "")
    return prediction_markdown(pred, message), prediction_to_json(pred), table


def reset_chat():
    return [], "<div class='safe-card'>对话已清空，可以开始新的演示。</div>", {}, pd.DataFrame(columns=["rank", "score", "label", "source_text", "retrieved_response"]), None


def export_chat(history: Any):
    history = normalize_history(history)
    if not history:
        return None, "暂无可导出的对话。"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"demo_chat_export_{ts}.md"
    lines = ["# 中文 AI 心理问诊对话系统 Demo 记录", "", f"导出时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ""]
    for item in history:
        role = "用户" if item["role"] == "user" else "系统"
        content = re.sub(r"<[^>]+>", "", str(item["content"]))
        lines.append(f"## {role}")
        lines.append(content)
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return str(out_path), f"已导出：{out_path.name}"


def quick_fill(example: str):
    return example


EMPTY_TABLE = pd.DataFrame(columns=["rank", "score", "label", "source_text", "retrieved_response"])


# ------------------------------------------------------------
# Gradio compatibility helpers
# ------------------------------------------------------------

def make_chatbot(**kwargs):
    """Create Chatbot across Gradio versions.

    Some Gradio versions use `type="messages"`, while newer/packaged builds
    may already default to messages and reject the `type` parameter.  This
    helper tries the richest configuration first and removes unsupported
    parameters one by one, so the app runs on both old and new Gradio.
    """
    candidates = [dict(kwargs)]
    if "type" in kwargs:
        k2 = dict(kwargs)
        k2.pop("type", None)
        candidates.append(k2)
    if "show_copy_button" in kwargs:
        k3 = dict(kwargs)
        k3.pop("show_copy_button", None)
        candidates.append(k3)
    if "type" in kwargs or "show_copy_button" in kwargs:
        k4 = dict(kwargs)
        k4.pop("type", None)
        k4.pop("show_copy_button", None)
        candidates.append(k4)

    last_err = None
    for cand in candidates:
        try:
            return gr.Chatbot(**cand)
        except TypeError as e:
            last_err = e
            continue
    raise last_err


def launch_demo(app):
    """Launch across Gradio 5/6 style APIs."""
    launch_kwargs = {
        "server_name": "0.0.0.0",
        "server_port": 7860,
    }
    # In Gradio 6, theme/css moved to launch(); older versions may not accept them.
    try:
        return app.launch(**launch_kwargs, theme=gr.themes.Soft(), css=CSS)
    except TypeError:
        return app.launch(**launch_kwargs)

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------

with gr.Blocks(title="中文 AI 心理问诊对话系统") as demo:
    gr.HTML(
        """
        <div id='title'>
          <h1>中文 AI 心理问诊对话系统</h1>
          <p>机器学习课程项目 Demo：输入中文心理求助文本，系统完成主题分类、风险提示、相似回复检索与可视化解释。</p>
        </div>
        """
    )
    gr.Markdown(model_status_markdown())

    with gr.Tabs():
        with gr.Tab("智能对话 Demo"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = make_chatbot(
                        label="对话演示",
                        height=560,
                        type="messages",  # old Gradio needs this; new Gradio may ignore it via helper
                        show_copy_button=True,
                    )
                    msg = gr.Textbox(
                        label="输入你的困扰",
                        placeholder="例如：最近考试压力很大，晚上睡不着，总觉得自己很没用……",
                        lines=4,
                    )
                    with gr.Row():
                        send = gr.Button("发送", variant="primary")
                        clear = gr.Button("清空对话")
                        export_btn = gr.Button("导出对话")
                    export_file = gr.File(label="导出的 Markdown 记录", interactive=False)
                    export_status = gr.Markdown("")

                with gr.Column(scale=2):
                    gr.Markdown("### 实时分析面板")
                    status_md = gr.Markdown("<div class='safe-card'>等待输入。</div>")
                    pred_json = gr.JSON(label="预测结果 JSON")
                    retrieval_table = gr.Dataframe(
                        label="Top-K 相似历史样本",
                        headers=["rank", "score", "label", "source_text", "retrieved_response"],
                        value=EMPTY_TABLE,
                        wrap=True,
                        interactive=False,
                    )
                    with gr.Accordion("演示控制", open=True):
                        use_retrieval = gr.Checkbox(label="启用相似回复检索", value=True)
                        top_k = gr.Slider(label="检索 Top-K", minimum=1, maximum=5, value=3, step=1)
                        style = gr.Radio(
                            label="回复风格",
                            choices=["默认", "更温柔", "更行动导向", "更适合答辩展示"],
                            value="默认",
                        )

            gr.Markdown("### 快速演示样例")
            with gr.Row():
                ex1 = gr.Button("考试压力 / 失眠")
                ex2 = gr.Button("分手难过")
                ex3 = gr.Button("室友冲突")
                ex4 = gr.Button("高风险测试")
            ex1.click(lambda: "最近考试压力很大，晚上总睡不着，感觉自己很没用。", outputs=msg)
            ex2.click(lambda: "我和女朋友分手了，心里特别难受，什么都不想做。", outputs=msg)
            ex3.click(lambda: "我和室友总是吵架，回宿舍就很压抑，不知道怎么相处。", outputs=msg)
            ex4.click(lambda: "我觉得活着没有意义，真的想消失。", outputs=msg)

        with gr.Tab("单句分析 / 检索调试"):
            with gr.Row():
                with gr.Column(scale=2):
                    analyze_text = gr.Textbox(
                        label="待分析文本",
                        placeholder="粘贴一句中文心理求助文本，查看分类与检索结果。",
                        lines=6,
                    )
                    analyze_btn = gr.Button("分析", variant="primary")
                with gr.Column(scale=2):
                    analyze_status = gr.Markdown("<div class='safe-card'>等待分析。</div>")
                    analyze_json = gr.JSON(label="分类结果")
            analyze_table = gr.Dataframe(
                label="相似样本检索结果",
                headers=["rank", "score", "label", "source_text", "retrieved_response"],
                value=EMPTY_TABLE,
                wrap=True,
                interactive=False,
            )

        with gr.Tab("项目说明"):
            gr.Markdown(
                f"""
### 功能模块
1. **文本分类**：优先加载 `models/bert_best`，如果不存在则使用 `models/best_baseline.joblib`，再否则使用规则兜底。
2. **相似回复检索**：读取 `models/response_retrieval.joblib`，从训练集回复中检索相似样本。
3. **风险保护**：检测到自伤、轻生等高风险表达时，跳过历史回复检索，优先展示求助提醒。
4. **可解释展示**：展示预测类别、置信度、模型来源、Top-K 相似样本、检索相似度。
5. **对话导出**：可把现场演示对话保存到 `outputs/` 目录，方便答辩留档。

### 当前状态
- 分类模型：`{predictor.kind}`
- 检索索引：`{'已加载' if retriever is not None else '未加载，请先运行 scripts/05_build_retrieval_index.py'}`
- 运行地址：`http://127.0.0.1:7860`

### 重要说明
本系统仅用于机器学习课程项目展示，不能替代专业心理咨询、医疗诊断或紧急救援。
                """
            )

    send.click(
        build_reply,
        inputs=[msg, chatbot, use_retrieval, top_k, style],
        outputs=[msg, chatbot, status_md, pred_json, retrieval_table],
    )
    msg.submit(
        build_reply,
        inputs=[msg, chatbot, use_retrieval, top_k, style],
        outputs=[msg, chatbot, status_md, pred_json, retrieval_table],
    )
    clear.click(reset_chat, outputs=[chatbot, status_md, pred_json, retrieval_table, export_file])
    export_btn.click(export_chat, inputs=[chatbot], outputs=[export_file, export_status])
    analyze_btn.click(
        analyze_only,
        inputs=[analyze_text, use_retrieval, top_k],
        outputs=[analyze_status, analyze_json, analyze_table],
    )

if __name__ == "__main__":
    launch_demo(demo)
