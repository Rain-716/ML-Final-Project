# 中文 AI 心理问诊对话系统（机器学习自选主题项目）

本项目面向“中文 AI 心理问诊对话”课程展示，核心任务不是替代心理咨询师，而是完成一个可解释、可训练、可演示的机器学习系统：

1. 将用户输入的中文心理求助文本分类为不同主题/风险类别；
2. 用成熟中文心理对话数据集构建训练集与回复库；
3. 对比传统机器学习基准模型与 BERT/MacBERT GPU 微调模型；
4. 用 Gradio 搭建可交互 Web Demo，实时展示预测类别、置信度、安全提示与相似回复。

> 安全声明：本项目仅用于课程展示和机器学习实验，不提供医学诊断、心理诊断或治疗建议。遇到自伤/轻生风险，应立即联系身边可信任的人、当地紧急电话或专业机构。

---

## 1. 参考开源项目与数据集选择

### 1.1 主要参考项目

- **SoulChat**：中文心理健康对话大模型，开源了心理咨询领域中文长文本指令与多轮共情对话数据相关说明。
- **PsyQA**：ACL Findings 2021 中文心理健康支持问答数据集，可参考其心理支持问答任务设计；完整数据需要向作者申请授权。
- **CPsyCoun**：ACL 2024 中文心理咨询多轮对话重构与评估框架，可作为报告中“后续改进方向”的参考。

### 1.2 本项目采用的数据集

采用 **SoulChatCorpus 开源版本**，原因：

- 中文心理健康对话领域相关；
- 规模较大，适合训练与检索；
- 可通过 ModelScope 下载到本地；
- 语料是“用户：...\n心理咨询师：...”形式，适合预处理成用户文本分类样本和心理支持回复库。

**下载地址/命令：**

```bash
pip install -U modelscope
modelscope download --dataset YIRONGCHEN/SoulChatCorpus --local_dir data/raw/SoulChatCorpus
```

也可以运行项目内脚本：

```bash
python scripts/00_download_soulchat_dataset.py --local_dir data/raw/SoulChatCorpus
```

如果下载速度慢，可到 ModelScope 页面搜索 `YIRONGCHEN/SoulChatCorpus` 手动下载后解压到：

```text
data/raw/SoulChatCorpus/
```

---

## 2. 问题定义

- **问题类型**：中文心理求助文本多分类任务。
- **输入**：用户的一段中文求助文本或多轮对话中的最近一轮用户发言。
- **输出类别**：
  - 高风险求助/危机干预 `high_risk`
  - 亲密关系与失恋 `relationship`
  - 学习/工作压力 `study_work`
  - 家庭关系 `family`
  - 人际与社交 `interpersonal`
  - 情绪困扰/焦虑抑郁 `emotion`
  - 睡眠与身心症状 `sleep_body`
  - 自我成长/自我评价 `self_growth`
  - 其他心理支持 `other`

SoulChatCorpus 原始公开数据主要是对话/指令语料，不是直接的分类表。因此本项目在 `scripts/01_preprocess_soulchat.py` 中使用透明的关键词规则生成弱标签，用于课程机器学习训练。该标签不是医学诊断，只服务于分类实验和界面路由。

---

## 3. 环境安装

建议使用 Python 3.9 或 3.10。

### 3.1 Conda GPU 环境（推荐）

```bash
conda env create -f environment.yml
conda activate cn-psy-dialog-ml
```

### 3.2 pip 环境

先根据你的显卡 CUDA 版本安装 GPU 版 PyTorch，例如 CUDA 11.8：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

检查 GPU：

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 4. 全流程运行

### Step 1：下载数据集

```bash
python scripts/00_download_soulchat_dataset.py --local_dir data/raw/SoulChatCorpus
```

### Step 2：独立预处理

```bash
python scripts/01_preprocess_soulchat.py --raw_dir data/raw/SoulChatCorpus
```

如果只是先测试流程，可以用小样本：

```bash
python scripts/01_preprocess_soulchat.py --raw_dir examples --max_samples 1000
```

输出：

```text
data/processed/all.csv
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
data/processed/preprocess_stats.json
```

预处理包含：

- 解析多轮对话，抽取用户发言与咨询师回复；
- 清理 HTML、URL、空白字符；
- 去重、过滤过短/过长样本；
- 基于关键词生成弱标签与风险等级；
- 构造数值特征：文本长度、轮次数、问号数、感叹号数、高风险关键词数；
- 分层划分训练集/验证集/测试集。

### Step 3：训练传统机器学习基准模型

```bash
python scripts/02_train_baselines.py --n_iter 12 --cv 3
```

包含：

- DummyClassifier：最低基准；
- Logistic Regression：TF-IDF + 数值特征；
- Linear SVM：TF-IDF + 数值特征；
- SGD Logistic：适合较大文本数据；
- RandomizedSearchCV + 交叉验证调参；
- 输出混淆矩阵、学习曲线、模型对比图。

### Step 4：GPU 训练 BERT/MacBERT 模型

默认使用 `hfl/chinese-macbert-base`。如果你已将模型下载到本地，可以把 `--pretrained_model` 改成本地目录。

```bash
python scripts/03_train_bert_gpu.py \
  --pretrained_model hfl/chinese-macbert-base \
  --epochs 3 \
  --batch_size 16 \
  --lr 2e-5 \
  --max_length 256 \
  --fp16 \
  --device cuda
```

显存不足时：

```bash
python scripts/03_train_bert_gpu.py --batch_size 8 --grad_accum 2 --fp16
```

### Step 5：汇总评估

```bash
python scripts/04_evaluate.py
```

评估指标：Accuracy、Macro-F1、Weighted-F1、每类 Precision/Recall/F1、混淆矩阵。Macro-F1 用于避免类别不均衡时只看准确率导致的误判。

### Step 6：构建回复检索索引

```bash
python scripts/05_build_retrieval_index.py
```

### Step 7：启动交互界面

```bash
python app.py
```

浏览器打开：

```text
http://127.0.0.1:7860
```

---

## 5. 一键运行

```bash
python scripts/06_run_all.py --raw_dir data/raw/SoulChatCorpus
```

快速测试，不训练 BERT：

```bash
python scripts/06_run_all.py --raw_dir data/raw/SoulChatCorpus --max_samples 5000 --skip_bert
```

---

## 6. 输出文件说明

```text
models/
  best_baseline.joblib              # 最优传统机器学习模型
  bert_best/                        # BERT/MacBERT GPU 微调模型
  response_retrieval.joblib         # 回复检索索引

outputs/
  baseline/
    baseline_summary.json
    *_metrics.json
    *_confusion_matrix.png
    best_learning_curve.png
    model_macro_f1_comparison.png
  bert/
    bert_summary.json
    test_metrics.json
    bert_confusion_matrix.png
    bert_training_loss.png
    bert_val_f1.png
  final/
    final_evaluation_summary.json
    final_macro_f1_comparison.png
```

---

## 7. 项目结构

```text
chinese_ai_psych_consultation/
  app.py
  README.md
  requirements.txt
  environment.yml
  scripts/
    00_download_soulchat_dataset.py
    01_preprocess_soulchat.py
    02_train_baselines.py
    03_train_bert_gpu.py
    04_evaluate.py
    05_build_retrieval_index.py
    06_run_all.py
  src/
    config.py
    data_utils.py
    label_rules.py
    metrics_utils.py
    predict.py
    retrieval.py
    safety.py
  docs/
    project_report.md
    scoring_mapping.md
    defense_questions.md
  examples/
    sample_soulchat_like.jsonl
```

---

## 8. 常见问题

### Q1：`modelscope` 下载失败怎么办？

先升级：

```bash
pip install -U modelscope
```

也可以登录 ModelScope，在网页搜索 `YIRONGCHEN/SoulChatCorpus` 后手动下载。

### Q2：BERT 模型下载太慢怎么办？

可先把预训练模型下载到本地，例如：

```bash
huggingface-cli download hfl/chinese-macbert-base --local-dir models/chinese-macbert-base
```

然后训练时：

```bash
python scripts/03_train_bert_gpu.py --pretrained_model models/chinese-macbert-base --fp16
```

### Q3：没有 GPU 能不能跑？

可以跑传统机器学习基准和 Gradio 界面。BERT 微调建议使用 GPU，否则会非常慢。

### Q4：这个系统能做心理诊断吗？

不能。它只是课程项目 Demo，用于文本分类、模型评估和交互展示。任何危机、自伤、轻生或严重心理困扰都应寻求专业帮助。
