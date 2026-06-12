# AI心理问诊对话：基于机器学习的情绪识别与回应检索系统

本项目面向“AI心理问诊对话”场景，将用户输入映射为心理支持场景标签，并检索/生成安全的共情回应。项目使用传统机器学习完成可解释分类，再结合模板与最近邻检索实现可交互 Demo。

> 安全声明：本系统仅用于课程项目与心理健康科普，不提供诊断、治疗或危机干预；检测到自伤/轻生风险时会优先提示联系专业人员与紧急服务。

## 参考开源项目/成熟数据集
- EmpatheticDialogues：Meta/Facebook AI 发布的 25k 共情对话数据集，可通过 Hugging Face `facebook/empathetic_dialogues` 下载。
- Mental Health Counseling Conversations：Kaggle 上的心理咨询问答数据，可作为真实问答风格补充。
- 参考 GitHub 主题：mental-health-chatbot，常见实现包括 Streamlit/Gradio 交互与检索式回应。

## 目录
```text
ai_psych_dialogue_project/
├─ data/sample_dialogue.csv          # 离线演示小样本；正式训练可自动下载成熟数据集
├─ src/train.py                      # 数据处理、基准模型、多模型比较、GridSearchCV
├─ src/app.py                        # Gradio实时对话Demo
├─ src/utils.py                      # 数据加载、安全规则、回应检索
├─ outputs/                          # 训练输出、图表、模型
├─ docs/项目报告.docx
├─ docs/答辩PPT.pptx
└─ requirements.txt
```

## 快速运行
```bash
pip install -r requirements.txt
python src/train.py --data_source sample
python src/app.py
```

使用成熟数据集训练（需联网）：
```bash
python src/train.py --data_source empathetic --max_samples 8000
```

## 评分点对应
- 问题定义：多分类任务（情绪/心理支持场景识别）+ 检索式回应。
- 数据处理：缺失值处理、去重、文本清洗、TF-IDF编码、类别编码。
- 建模训练：DummyClassifier基准；Logistic Regression、LinearSVM、RandomForest；GridSearchCV。
- 评估可视化：Accuracy、Precision、Recall、F1、混淆矩阵、学习曲线、特征重要性。
- 展示：Gradio Web Demo。
