# CS-Math Assignment: PageRank & Naive Bayes Classification

\> [Jump to English](#english-version) | [跳转到中文](#chinese-version)

---
# English Version

## Overview
This assignment contains two independent tasks:

1. **PageRank Graph Analysis**  
2. **Naive Bayes Classification on Census Income Dataset**

All program outputs are stored in the **results/** directory.

## Project Structure
```
├── pagerank.py  
├── bayesian.py  
├── dataset/  
│   ├── PageRank_Dataset.csv  
│   ├── Bayesian_Dataset_train.csv  
│   └── Bayesian_Dataset_test.csv  
└── results/  
    ├── pagerank/  
    └── bayesian/  
```

## Task 1: PageRank
- Load directed graph dataset  
- Compute PageRank values using NetworkX  
- Analyze effect of different damping factors (α)  
- Save visualization results including PageRank distribution chart

Output files saved to: `results/pagerank/`

## Task 2: Bayesian Classification
- Encode categorical features  
- Train Categorical Naive Bayes model  
- Evaluate with precision / recall / F1-score  
- Plot best confusion matrix  
- Compare performance under different α

Output files saved to: `results/bayesian/`

## How to Run
```bash
pip install -r requirements.txt
python3 pagerank.py
python3 bayesian.py
```

## Report and Submission
- PDF report will include task descriptions, visualizations, analysis, and conclusion  
- Code will be submitted as attachments  
- All result files compressed as required

Submission structure:
```
📦 Submission
 ├─ report.pdf
 ├─ code.zip
 │   ├─ pagerank.py
 │   ├─ bayesian.py
 │   ├─ dataset/
 │   └─ results/
```

---

# Chinese Version
## 概述
本作业包含两个相互独立的任务：

1. PageRank 图模型分析  
2. 朴素贝叶斯分类（预测收入是否超过 50K）

程序输出均保存在 `results/` 目录。

## 项目结构
```
├── pagerank.py  
├── bayesian.py  
├── dataset/  
│   ├── PageRank_Dataset.csv  
│   ├── Bayesian_Dataset_train.csv  
│   └── Bayesian_Dataset_test.csv  
└── results/  
    ├── pagerank/  
    └── bayesian/  
```

## 任务 1：PageRank
- 读取有向图数据集  
- 使用 NetworkX 计算 PageRank  
- 不同阻尼因子 α 的影响分析  
- 保存 PageRank 分布可视化结果

输出文件保存至：`results/pagerank/`

## 任务 2：朴素贝叶斯分类
- 类别特征编码  
- 训练 Categorical Naive Bayes 模型  
- 输出精确率 / 召回率 / F1-score  
- 绘制最优模型混淆矩阵  
- 不同 α 对模型性能的影响分析

输出文件保存至：`results/bayesian/`

## 使用方式
```bash
pip install -r requirements.txt
python3 pagerank.py
python3 bayesian.py
```
---
