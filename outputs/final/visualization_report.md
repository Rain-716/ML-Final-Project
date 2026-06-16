# 最终评估与可视化索引

本文件由 `scripts/04_evaluate.py` 自动生成，用于课程报告/答辩时快速定位图表。

## 1. 核心结论摘要

| model_key | display_name | family | accuracy | macro_f1 | weighted_f1 |
| --- | --- | --- | --- | --- | --- |
| bert_gpu | BERT / MacBERT GPU | bert_gpu | 0.9981 | 0.9971 | 0.9981 |
| best_traditional_ml | Best traditional ML
(sgd_hinge_alpha_1e-5_ngram_2_3) | traditional_ml_best | 0.9683 | 0.9483 | 0.9684 |
| baseline::sgd_hinge_alpha_1e-5_ngram_2_3 | sgd_hinge_alpha_1e-5_ngram_2_3 | traditional_ml_candidate | 0.9657 | 0.9437 | 0.9659 |
| baseline::sgd_modified_huber_alpha_1e-5_ngram_2_3 | sgd_modified_huber_alpha_1e-5_ngram_2_3 | traditional_ml_candidate | 0.9652 | 0.9381 | 0.9650 |
| baseline::sgd_log_loss_alpha_1e-5_ngram_2_3 | sgd_log_loss_alpha_1e-5_ngram_2_3 | traditional_ml_candidate | 0.9333 | 0.9032 | 0.9331 |
| baseline::complement_nb_alpha_0.2_ngram_2_3 | complement_nb_alpha_0.2_ngram_2_3 | traditional_ml_candidate | 0.7563 | 0.7249 | 0.7567 |
| baseline::dummy_most_frequent | dummy_most_frequent | traditional_ml_candidate | 0.4529 | 0.0693 | 0.2823 |

## 2. 每类表现最低的类别

| model_key | label | precision | recall | f1 | support |
| --- | --- | --- | --- | --- | --- |
| bert_gpu | high_risk | 1.0000 | 0.9869 | 0.9934 | 837.0000 |
| bert_gpu | family | 0.9941 | 0.9954 | 0.9948 | 1535.0000 |
| bert_gpu | relationship | 0.9931 | 0.9974 | 0.9952 | 1152.0000 |
| best_traditional_ml | sleep_body | 0.8657 | 0.9268 | 0.8952 | 1830.0000 |
| best_traditional_ml | relationship | 0.8910 | 0.9226 | 0.9066 | 5247.0000 |
| best_traditional_ml | study_work | 0.9202 | 0.9386 | 0.9293 | 10294.0000 |

## 3. 已生成图表

| 文件 | 用途 |
|---|---|
| `final_core_metrics_grouped_bar.png` | 传统 ML 与 BERT 的 Accuracy/Macro-F1/Weighted-F1 对比 |
| `all_models_macro_f1_ranking.png` | 所有候选模型 Macro-F1 排名 |
| `all_models_accuracy_ranking.png` | 模型评估可视化 |
| `all_models_metric_heatmap.png` | 多模型多指标热力图 |
| `baseline_speed_vs_macro_f1.png` | 传统 ML 候选模型拟合耗时与 Macro-F1 的关系，已使用短标签防重叠 |
| `baseline_fit_seconds_bar.png` | 传统 ML 候选模型拟合耗时条形图，比散点图更适合报告展示 |
| `macro_weighted_f1_gap.png` | Weighted-F1 与 Macro-F1 差距，说明类别不均衡影响 |
| `best_traditional_ml_per_class_f1.png` | 模型评估可视化 |
| `best_traditional_ml_per_class_prf_grouped.png` | 模型评估可视化 |
| `best_traditional_ml_class_support.png` | 模型评估可视化 |
| `best_traditional_ml_support_vs_f1.png` | 模型评估可视化 |
| `bert_gpu_per_class_f1.png` | 模型评估可视化 |
| `bert_gpu_per_class_prf_grouped.png` | 模型评估可视化 |
| `bert_gpu_class_support.png` | 模型评估可视化 |
| `bert_gpu_support_vs_f1.png` | 模型评估可视化 |
| `dataset_label_distribution.png` | 训练/验证/测试类别分布，说明样本不均衡 |
| `baseline_learning_curve_enhanced.png` | 学习曲线，展示样本量增加后的训练/验证表现 |
| `baseline_generalization_gap.png` | 训练-验证差距，辅助说明过拟合风险 |
| `bert_training_loss_curve.png` | 模型评估可视化 |
| `bert_val_macro_f1_curve.png` | 模型评估可视化 |
| `bert_epoch_time_bar.png` | 模型评估可视化 |

## 4. 指标异常与展示处理说明

- 本脚本不会修改 Accuracy、Macro-F1、Weighted-F1 等真实评估数值，只会增加 `validity_flag` 和 `report_note` 字段帮助解释。
- 如果 BERT/MacBERT 指标接近 1.0，而本项目使用的是关键词弱标签，则应表述为：模型很好地拟合了弱标签分类任务，不能表述为真实心理诊断准确率接近 100%。
- `baseline_speed_vs_macro_f1.png` 中的 fit_seconds 仅代表传统 ML 候选模型在脚本中的拟合耗时，不包含数据下载、预处理、BERT GPU 训练和最终评估时间。
- 若散点图仍显拥挤，报告中优先使用 `baseline_fit_seconds_bar.png` 和 `all_models_macro_f1_ranking.png`。

## 5. 报告写作提醒

- 分类任务不要只汇报 Accuracy，应同时汇报 Macro-F1、Weighted-F1、Precision、Recall。
- 如果 Weighted-F1 明显高于 Macro-F1，说明大类样本对总体指标影响较大，需要说明类别不均衡。
- 如果训练 Macro-F1 远高于验证 Macro-F1，说明可能存在过拟合，需要在局限性中说明。
- 高风险类别的召回率比普通类别更重要；报告中应单独解释 high_risk 的 Recall。