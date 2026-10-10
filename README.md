# 实验一：基于 TF-IDF 的英文文本分类

本项目完成一个 10 类英文新闻组文本分类任务，包含数据清洗、文本特征工程比较、SVM 调参、MLP 调参和无标签测试集预测。

最终采用的最佳方案为：

- 数据清洗：删除邮件结构噪声、引用、邮箱、签名和新闻组路径，保留否定词，标准化带符号技术词，并将标题重复三次；
- 文本特征：50000 维单词 unigram+bigram TF-IDF 与 30000 维字符 3～5 gram TF-IDF 拼接；
- 分类器：`LinearSVC(C=1.0, loss="squared_hinge")`；
- 主要选择指标：验证集 Macro-F1；
- 最终输出：`predictions.csv`。

## 1. 项目目录

运行前，建议保持以下文件位于同一个目录中：

```text
data-Project1/
└─ data-Project1/
   ├─ train_data.csv
   ├─ test_data_unlabeled.csv
   ├─ 多粒度文本特征融合.py
   ├─ SVC.py
   ├─ MLP.py
   ├─ predict_test.py
   ├─ README.md
   ├─ results/                 # SVM 运行后自动创建
   └─ mlp_results/             # MLP 运行后自动创建
```

数据文件格式：

- `train_data.csv`：必须包含 `text` 和 `target` 两列；
- `test_data_unlabeled.csv`：必须包含 `text` 列，不包含标签；
- 当前训练集有 7368 条文本，测试集有 2457 条文本，标签为 0～9。

## 2. 环境准备

建议使用 Python 3.10 或更高版本。在 PowerShell 中进入项目目录：

```powershell
cd "C:\Users\Lenovo\Desktop\当代人工智能\实验一\data-Project1\data-Project1"
```

安装依赖：

```powershell
python -m pip install pandas numpy scikit-learn matplotlib joblib
```

检查当前 Python 是否可以导入依赖：

```powershell
python -c "import pandas, numpy, sklearn, matplotlib, joblib; print('环境正常')"
```

所有命令都应在包含训练数据和脚本的项目目录中运行。这样可以避免相对路径找不到数据文件的问题。

## 3. 推荐运行顺序

完整实验建议按照以下顺序运行：

```text
多粒度文本特征融合.py
          ↓
        SVC.py
          ↓
        MLP.py
          ↓
   predict_test.py
```

前三个脚本用于实验比较和选择模型。确定最佳方案后，最后一个脚本才读取无标签测试集并生成提交结果。

如果只需要直接生成最终预测文件，可以直接运行 `predict_test.py`，不需要重新运行前三个调参脚本。

## 4. 各代码文件说明

### 4.1 `多粒度文本特征融合.py`

用途：在固定数据划分、清洗策略和分类器参数的条件下，对比五种文本特征工程方法。

比较的方法包括：

1. 5000 维单词 unigram；
2. 20000 维单词 unigram；
3. 50000 维单词 unigram+bigram；
4. 30000 维字符 `char_wb` 3～5 gram；
5. 50000 维单词特征与 30000 维字符特征组合。

运行命令：

```powershell
python ".\多粒度文本特征融合.py"
```

主要输出：

- `svc_feature_comparison_results.csv`：五种特征方法的 Accuracy、Macro-F1、Hinge Loss 和训练时间；
- `best_svc_classification_report.csv`：最佳方法的分类报告；
- `best_svc_confusion_matrix.csv`：最佳方法的混淆矩阵；
- `best_svc_feature_extractor.joblib`：验证实验阶段保存的最佳特征提取器；
- `best_svc_model.joblib`：验证实验阶段保存的最佳 SVC；
- `best_svc_information.csv`：最佳方法及指标摘要。

注意：这些模型文件来自训练集/验证集实验，主要用于复现实验结果。最终测试集预测仍应使用 `predict_test.py`，以便用全部带标签数据重新训练模型。

### 4.2 `SVC.py`

用途：完成 SVM 系列模型调参，并生成损失函数和性能曲线。

该脚本包含项目的完整文本清洗流程：

- 提取 `Subject` 和正文；
- 删除邮件头、邮箱、引用内容和签名；
- 删除可能直接暴露类别的新闻组路径；
- 展开否定缩写，例如 `don't → do not`；
- 保留 `no`、`not`、`never`、`without`；
- 通用标准化带符号技术词，例如 `C++ → c_plusplus`、`OS/2 → os_2`；
- 转换为小写并删除低价值标点；
- 删除英文停用词和单字符；
- 将清洗后的标题重复三次后与正文拼接。

特征提取采用方法 5：

```text
50000 维单词 unigram+bigram
              +
30000 维字符 char_wb 3～5 gram
              =
80000 维组合稀疏特征
```

模型比较包括：

- `LinearSVC`：调节 `C=[0.01, 0.1, 1, 10, 100]`；
- 线性核 `SVC`：调节 `C=[0.1, 1, 10, 100]`；
- RBF 核 `SVC`：同时调节 `C` 和 `gamma=[scale, auto, 0.1, 1]`。

运行命令：

```powershell
python .\SVC.py
```

结果自动保存到 `results/`：

- `svm_tuning_results.csv`：全部 SVM 参数结果；
- `best_svm_classification_report.csv`：最佳 SVM 分类报告；
- `best_svm_validation_results.csv`：最佳 SVM 的验证集逐样本结果；
- `linearsvc_loss_curve.png`：LinearSVC 的 Hinge Loss 曲线；
- `linearsvc_score_curve.png`：LinearSVC 的验证指标曲线；
- `svc_linear_loss_curve.png`：线性核 SVC 的损失曲线；
- `svc_rbf_loss_curve.png`：RBF-SVC 的损失曲线；
- `svc_rbf_score_curve.png`：RBF-SVC 的 Macro-F1 曲线。

该脚本需要训练多个 `SVC`，运行时间可能较长。不要在运行过程中关闭终端。

### 4.3 `MLP.py`

用途：比较多层感知机的隐藏层结构和 L2 正则化参数，并绘制训练损失曲线。

固定设置：

- 激活函数：`relu`；
- 优化器：`adam`；
- 使用 Early Stopping；
- 文本特征仍采用方法 5；
- 主要评价指标为 Macro-F1，损失指标为 Log Loss。

主要比较：

- 隐藏层结构：`(50,)`、`(100,)`、`(200,)`、`(100, 50)`；
- 正则化参数：`alpha=[0.0001, 0.001, 0.01, 0.1]`；
- 使用多个随机种子检查最佳配置的稳定性。

运行命令：

```powershell
python .\MLP.py
```

结果自动保存到 `mlp_results/`：

- `mlp_structure_results.csv`：不同隐藏层结构的结果；
- `mlp_alpha_results.csv`：不同 `alpha` 的结果；
- `mlp_repeat_results.csv`：不同随机种子的重复实验结果；
- `mlp_summary.csv`：MLP 实验摘要；
- `mlp_classification_report.csv`：最佳 MLP 分类报告；
- `mlp_validation_results.csv`：最佳 MLP 的验证集逐样本结果；
- `mlp_structure_loss_curves.png`：隐藏层结构训练损失曲线；
- `mlp_alpha_loss_curves.png`：不同 `alpha` 的训练损失曲线；
- `mlp_best_loss_curve.png`：最佳 MLP 训练损失曲线；
- `mlp_internal_validation_curve.png`：Early Stopping 内部验证曲线。

MLP 对随机初始化敏感，因此不同随机种子的结果可能存在小幅变化。

### 4.4 `predict_test.py`

用途：使用最终确定的最佳方案训练全部带标签数据，预测无标签测试集并生成提交文件。

运行过程：

1. 读取全部 7368 条带标签训练数据；
2. 读取 2457 条无标签测试数据；
3. 从 `SVC.py` 导入 `clean_text`，对训练集和测试集执行完全相同的清洗流程；
4. 在全部训练文本上学习 80000 维组合 TF-IDF；
5. 使用全部训练数据训练 `LinearSVC(C=1.0)`；
6. 预测测试集标签；
7. 保存 `predictions.csv`。

运行命令：

```powershell
python .\predict_test.py
```

该脚本中：

```python
from SVC import clean_text
```

表示直接复用 `SVC.py` 中的完整清洗流程。因此 `predict_test.py` 和 `SVC.py` 必须保存在同一个目录中，并且不要只修改其中一个文件的清洗规则。

最终输出格式：

```csv
4
7
1
9
0
```

`predictions.csv` 应满足：

- 一共 2457 行；
- 每行一个预测标签；
- 标签范围为 0～9；
- 没有表头；
- 没有 DataFrame 索引。

检查预测文件行数：

```powershell
(Get-Content .\predictions.csv).Count
```

正常结果应为：

```text
2457
```

## 5. 评价指标说明

### Accuracy

预测正确的样本数占全部样本数的比例，越高越好。

### Macro-F1

先分别计算每个类别的 F1，再对 10 个类别取平均。它能够平等对待每个类别，是本实验选择最佳模型的主要指标，越高越好。

### Hinge Loss

用于评价 SVM 决策间隔。损失越低，说明样本分类间隔通常越合理。Hinge Loss 用于同类 SVM 配置之间的比较，不应与 MLP 的 Log Loss 直接比较。

### Log Loss

用于评价 MLP 的概率输出。模型对错误类别越自信，惩罚越大。Log Loss 越低越好。

## 6. 实验规范

- 调参阶段只使用训练集和验证集，不使用无标签测试集选择模型；
- TF-IDF 只能在训练部分执行 `fit_transform()`，验证集只能执行 `transform()`；
- 最终模型和参数确定后，才使用全部带标签数据重新拟合 TF-IDF 和模型；
- 训练集和测试集必须使用完全相同的清洗函数；
- 不要利用测试文本手工设计类别关键词；
- 固定 `random_state=42`，保证数据划分和主要结果可复现；
- 不要把 SVM 的 Hinge Loss 与 MLP 的 Log Loss 直接比较大小。

## 7. 常见问题

### 7.1 `ModuleNotFoundError: No module named 'sklearn'`

说明当前运行脚本的 Python 环境没有安装 scikit-learn。执行：

```powershell
python -m pip install scikit-learn pandas numpy matplotlib joblib
```

如果电脑中安装了多个 Python，应确保安装依赖和运行脚本使用的是同一个解释器：

```powershell
python -c "import sys; print(sys.executable)"
python -m pip --version
```

### 7.2 `PermissionError: predictions.csv`

这通常表示 `predictions.csv` 正被 Excel、WPS 或其他程序占用，Windows 不允许 Python 覆盖它。

处理方法：

1. 关闭 Excel 或 WPS 中打开的 `predictions.csv`；
2. 确认文件没有被其他程序预览；
3. 重新运行 `python .\predict_test.py`。

当前 `predict_test.py` 已增加备用保存机制。如果 `predictions.csv` 被占用，程序会自动保存为：

```text
predictions_new.csv
```

若提交系统要求文件名必须为 `predictions.csv`，关闭占用程序后再将备用文件改名。

### 7.3 无法保存到 `results/` 或 `mlp_results/`

当前 `SVC.py` 和 `MLP.py` 会自动创建输出目录。如果仍然报错，请确认项目目录具有写入权限，并检查同名路径是否被错误创建成普通文件。

### 7.4 运行时间较长

这是正常现象。80000 维组合特征会增加内存和训练时间，RBF-SVC 和 MLP 通常明显慢于 LinearSVC。实验运行时可通过终端输出观察当前模型和耗时，不要反复同时启动多个训练进程。

### 7.5 修改清洗策略后应该做什么

清洗策略改变会导致特征空间变化，因此必须重新运行特征比较和模型调参，不能直接沿用旧模型的最佳参数或旧的模型文件。最终预测时，训练集和测试集也必须同时使用新的清洗策略。

## 8. 最简运行方式

如果模型选择实验已经完成，只需要生成提交文件，执行：

```powershell
cd "C:\Users\Lenovo\Desktop\当代人工智能\实验一\data-Project1\data-Project1"
python .\predict_test.py
```

运行完成后检查：

```powershell
(Get-Content .\predictions.csv).Count
Get-Content .\predictions.csv -TotalCount 5
```

应得到 2457 行预测标签，随后提交 `predictions.csv`。
