# svc_feature_comparison.py
# 固定数据划分、清洗策略和SVC参数
# 只比较不同文本特征工程方法
# 不读取、不处理测试集

import re
import time
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    ENGLISH_STOP_WORDS
)
from sklearn.pipeline import FeatureUnion
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hinge_loss,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. 固定实验参数
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.2

# 所有实验使用完全相同的SVC参数
SVC_C = 1.0


# ============================================================
# 2. 读取训练数据
# ============================================================

def load_data():
    train_df = pd.read_csv("train_data.csv")

    X = train_df["text"].fillna("").astype(str).tolist()
    y = train_df["target"].values

    return X, y


X, y = load_data()

print("=" * 65)
print("SVC文本特征工程对比实验")
print("=" * 65)
print(f"样本总数：{len(X)}")
print(f"类别数量：{len(np.unique(y))}")


# ============================================================
# 3. 固定训练集和验证集
# ============================================================

X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\n--- 数据划分完成 ---")
print(f"训练集数量：{len(X_train_raw)}")
print(f"验证集数量：{len(X_val_raw)}")


# ============================================================
# 4. 文本清洗配置
# ============================================================

STOP_WORDS = set(
    ENGLISH_STOP_WORDS
)

STOP_WORDS -= {
    "no",
    "not",
    "never",
    "without"
}


# ============================================================
# 5. 数据清洗函数
# ============================================================

def normalize_symbolic_tokens(text):
    # C++ → C_plusplus
    text = re.sub(
        r"(?<=\w)\+\+(?!\w)",
        "_plusplus",
        text
    )

    # C# → C_sharp
    text = re.sub(
        r"(?<=\w)#(?!\w)",
        "_sharp",
        text
    )

    # OS/2 → OS_2
    # PC/GEOS → PC_GEOS
    # e-mail → e_mail
    text = re.sub(
        r"(?<=\w)[/-](?=\w)",
        "_",
        text
    )

    return text


def normalize_contractions(text):
    replacements = [
        (r"(?i)\bwon't\b", "will not"),
        (r"(?i)\bcan't\b", "can not"),
        (r"(?i)\bdon't\b", "do not"),
        (r"(?i)\bdoesn't\b", "does not"),
        (r"(?i)\bdidn't\b", "did not"),
        (r"(?i)\bisn't\b", "is not"),
        (r"(?i)\baren't\b", "are not"),
        (r"(?i)\bwasn't\b", "was not"),
        (r"(?i)\bweren't\b", "were not"),
        (r"(?i)\bhasn't\b", "has not"),
        (r"(?i)\bhaven't\b", "have not"),
        (r"(?i)\bhadn't\b", "had not"),
        (r"(?i)\bshouldn't\b", "should not"),
        (r"(?i)\bwouldn't\b", "would not"),
        (r"(?i)\bcouldn't\b", "could not"),
        (r"(?i)\bmustn't\b", "must not")
    ]

    for pattern, replacement in replacements:
        text = re.sub(
            pattern,
            replacement,
            text
        )

    return text


def remove_newsgroup_paths(text):
    text = re.sub(
        (
            r"(?i)\b"
            r"(?:comp|sci|rec|talk|soc|misc|alt)"
            r"(?:\.[a-z0-9_+-]+)+"
            r"\b"
        ),
        " ",
        text
    )

    return text


def process_text_part(text):
    text = str(text)

    text = normalize_symbolic_tokens(
        text
    )

    text = normalize_contractions(
        text
    )

    text = remove_newsgroup_paths(
        text
    )

    text = re.sub(
        r"(?i)^In article .*?writes:\s*$",
        " ",
        text,
        flags=re.MULTILINE
    )

    text = re.sub(
        r"^>.*$",
        " ",
        text,
        flags=re.MULTILINE
    )

    text = re.sub(
        r"\b[\w.-]+@[\w.-]+\.\w+\b",
        " ",
        text
    )

    text = re.split(
        r"\n--+\s*\n",
        text,
        maxsplit=1
    )[0]

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9_\s]",
        " ",
        text
    )

    words = text.split()

    words = [
        word
        for word in words
        if (
            word not in STOP_WORDS
            and len(word) > 1
        )
    ]

    return " ".join(words)


def clean_text(text):
    text = str(text)

    subject_match = re.search(
        r"^Subject:\s*(.*)$",
        text,
        flags=(
            re.MULTILINE
            | re.IGNORECASE
        )
    )

    if subject_match:
        subject = subject_match.group(1)
    else:
        subject = ""

    parts = re.split(
        r"\n\s*\n",
        text,
        maxsplit=1
    )

    if len(parts) == 2:
        body = parts[1]
    else:
        body = text

    body = re.sub(
        (
            r"^(From|Organization|Lines|Reply-To|"
            r"NNTP-Posting-Host|Distribution|"
            r"Newsgroups|Xref|Path):.*$"
        ),
        " ",
        body,
        flags=(
            re.MULTILINE
            | re.IGNORECASE
        )
    )

    body = re.sub(
        r"^Subject:.*$",
        " ",
        body,
        flags=(
            re.MULTILINE
            | re.IGNORECASE
        )
    )

    subject_clean = process_text_part(
        subject
    )

    body_clean = process_text_part(
        body
    )

    # 三重标题：标题出现三次
    return (
        f"{subject_clean} "
        f"{subject_clean} "
        f"{subject_clean} "
        f"{body_clean}"
    ).strip()


# ============================================================
# 6. 清洗文本
# ============================================================

print("\n--- 开始清洗文本 ---")
clean_start = time.time()

X_train_clean = [
    clean_text(text)
    for text in X_train_raw
]

X_val_clean = [
    clean_text(text)
    for text in X_val_raw
]

clean_time = time.time() - clean_start

print(f"文本清洗完成，用时：{clean_time:.2f} 秒")

print("\n--- 清洗结果示例 ---")
print(X_train_clean[0][:500])


# ============================================================
# 7. 定义不同文本特征工程方法
# ============================================================

def create_feature_methods():
    """
    每个方法只改变文本特征工程。
    SVC参数、数据清洗和数据划分保持不变。
    """

    return {
        # 方法1：基础词袋
        "方法1_单词unigram_5000": TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 1),
            max_features=5000,
            sublinear_tf=True,
            norm="l2",
            dtype=np.float32
        ),

        # 方法2：扩大单词特征数量
        "方法2_单词unigram_20000": TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 1),
            min_df=2,
            max_df=0.95,
            max_features=20000,
            sublinear_tf=True,
            norm="l2",
            dtype=np.float32
        ),

        # 方法3：加入二元词组
        "方法3_单词unigram_bigram_50000": TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            max_features=50000,
            sublinear_tf=True,
            norm="l2",
            dtype=np.float32
        ),

        # 方法4：只使用字符特征
        "方法4_字符char_wb_30000": TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=30000,
            sublinear_tf=True,
            norm="l2",
            dtype=np.float32
        ),

        # 方法5：单词特征和字符特征组合
        "方法5_单词加字符组合": FeatureUnion([
            (
                "word_tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    max_features=50000,
                    sublinear_tf=True,
                    norm="l2",
                    dtype=np.float32
                )
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=30000,
                    sublinear_tf=True,
                    norm="l2",
                    dtype=np.float32
                )
            )
        ])
    }


# ============================================================
# 8. 运行单个特征工程实验
# ============================================================

def run_experiment(
    experiment_name,
    feature_extractor,
    X_train_text,
    X_val_text,
    y_train_data,
    y_val_data
):
    print("\n" + "=" * 65)
    print(f"正在运行：{experiment_name}")
    print("=" * 65)

    # --------------------------------------------------------
    # 提取特征
    # --------------------------------------------------------

    feature_start = time.time()

    X_train_features = feature_extractor.fit_transform(
        X_train_text
    )

    X_val_features = feature_extractor.transform(
        X_val_text
    )

    feature_time = time.time() - feature_start

    print(f"训练集特征形状：{X_train_features.shape}")
    print(f"验证集特征形状：{X_val_features.shape}")
    print(f"特征提取时间：{feature_time:.2f} 秒")

    # --------------------------------------------------------
    # 所有实验使用相同的SVC
    # --------------------------------------------------------

    model = SVC(
        kernel="linear",
        C=SVC_C,
        decision_function_shape="ovr",
        probability=False,
        cache_size=2000,
        random_state=RANDOM_STATE
    )

    train_start = time.time()

    model.fit(
        X_train_features,
        y_train_data
    )

    train_time = time.time() - train_start

    # --------------------------------------------------------
    # 模型预测
    # --------------------------------------------------------

    y_train_pred = model.predict(
        X_train_features
    )

    y_val_pred = model.predict(
        X_val_features
    )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    train_accuracy = accuracy_score(
        y_train_data,
        y_train_pred
    )

    val_accuracy = accuracy_score(
        y_val_data,
        y_val_pred
    )

    # --------------------------------------------------------
    # Macro-F1
    # --------------------------------------------------------

    train_macro_f1 = f1_score(
        y_train_data,
        y_train_pred,
        average="macro"
    )

    val_macro_f1 = f1_score(
        y_val_data,
        y_val_pred,
        average="macro"
    )

    # --------------------------------------------------------
    # Hinge Loss
    # --------------------------------------------------------

    train_scores = model.decision_function(
        X_train_features
    )

    val_scores = model.decision_function(
        X_val_features
    )

    train_hinge = hinge_loss(
        y_train_data,
        train_scores,
        labels=model.classes_
    )

    val_hinge = hinge_loss(
        y_val_data,
        val_scores,
        labels=model.classes_
    )

    hinge_gap = val_hinge - train_hinge

    print(f"模型训练时间：{train_time:.2f} 秒")
    print(f"训练集 Accuracy：{train_accuracy:.4f}")
    print(f"验证集 Accuracy：{val_accuracy:.4f}")
    print(f"训练集 Macro-F1：{train_macro_f1:.4f}")
    print(f"验证集 Macro-F1：{val_macro_f1:.4f}")
    print(f"训练集 Hinge Loss：{train_hinge:.4f}")
    print(f"验证集 Hinge Loss：{val_hinge:.4f}")
    print(f"Hinge Loss差距：{hinge_gap:.4f}")

    result = {
        "实验名称": experiment_name,
        "特征数量": X_train_features.shape[1],
        "SVC_C": SVC_C,
        "训练集Accuracy": train_accuracy,
        "验证集Accuracy": val_accuracy,
        "训练集Macro-F1": train_macro_f1,
        "验证集Macro-F1": val_macro_f1,
        "训练集HingeLoss": train_hinge,
        "验证集HingeLoss": val_hinge,
        "HingeLoss差距": hinge_gap,
        "特征提取时间（秒）": feature_time,
        "模型训练时间（秒）": train_time
    }

    return {
        "result": result,
        "model": model,
        "feature_extractor": feature_extractor,
        "y_val_pred": y_val_pred
    }


# ============================================================
# 9. 依次比较所有特征工程方法
# ============================================================

feature_methods = create_feature_methods()

all_results = []
all_experiments = {}

total_start = time.time()

for experiment_name, feature_extractor in feature_methods.items():
    experiment_output = run_experiment(
        experiment_name=experiment_name,
        feature_extractor=feature_extractor,
        X_train_text=X_train_clean,
        X_val_text=X_val_clean,
        y_train_data=y_train,
        y_val_data=y_val
    )

    all_results.append(
        experiment_output["result"]
    )

    all_experiments[experiment_name] = experiment_output

total_time = time.time() - total_start


# ============================================================
# 10. 对实验结果排序
# ============================================================

results_df = pd.DataFrame(all_results)

# 第一标准：验证集Macro-F1越高越好
# 第二标准：验证集Hinge Loss越低越好
# 第三标准：验证集Accuracy越高越好
results_df = results_df.sort_values(
    by=[
        "验证集Macro-F1",
        "验证集HingeLoss",
        "验证集Accuracy"
    ],
    ascending=[
        False,
        True,
        False
    ]
).reset_index(drop=True)

results_df.insert(
    0,
    "排名",
    range(1, len(results_df) + 1)
)

print("\n" + "=" * 90)
print("不同文本特征工程方法对比结果")
print("=" * 90)

display_columns = [
    "排名",
    "实验名称",
    "特征数量",
    "训练集Accuracy",
    "验证集Accuracy",
    "验证集Macro-F1",
    "训练集HingeLoss",
    "验证集HingeLoss",
    "HingeLoss差距",
    "模型训练时间（秒）"
]

print(
    results_df[display_columns].to_string(
        index=False
    )
)


# ============================================================
# 11. 选出最佳特征工程方法
# ============================================================

best_name = results_df.iloc[0]["实验名称"]
best_result = results_df.iloc[0]

best_experiment = all_experiments[best_name]

best_model = best_experiment["model"]
best_feature_extractor = best_experiment["feature_extractor"]
best_y_val_pred = best_experiment["y_val_pred"]

print("\n" + "=" * 65)
print("最佳文本特征工程方法")
print("=" * 65)
print(f"最佳方法：{best_name}")
print(f"特征数量：{int(best_result['特征数量'])}")
print(f"验证集 Accuracy：{best_result['验证集Accuracy']:.4f}")
print(f"验证集 Macro-F1：{best_result['验证集Macro-F1']:.4f}")
print(f"验证集 Hinge Loss：{best_result['验证集HingeLoss']:.4f}")
print(f"Hinge Loss差距：{best_result['HingeLoss差距']:.4f}")

print("\n选择规则：")
print("1. 优先选择验证集Macro-F1最高的方法。")
print("2. Macro-F1接近时，优先选择验证集Hinge Loss更低的方法。")
print("3. Hinge Loss差距过大时，需要注意过拟合。")


# ============================================================
# 12. 输出最佳模型分类报告
# ============================================================

print("\n--- 最佳模型分类报告 ---")

best_report_text = classification_report(
    y_val,
    best_y_val_pred,
    digits=4,
    zero_division=0
)

print(best_report_text)


# ============================================================
# 13. 输出最佳模型混淆矩阵
# ============================================================

best_confusion_matrix = confusion_matrix(
    y_val,
    best_y_val_pred
)

print("\n--- 最佳模型混淆矩阵 ---")
print(best_confusion_matrix)


# ============================================================
# 14. 保存所有对比结果
# ============================================================

results_df.to_csv(
    "svc_feature_comparison_results.csv",
    index=False,
    encoding="utf-8-sig"
)

best_report_df = pd.DataFrame(
    classification_report(
        y_val,
        best_y_val_pred,
        output_dict=True,
        zero_division=0
    )
).transpose()

best_report_df.to_csv(
    "best_svc_classification_report.csv",
    encoding="utf-8-sig"
)

class_labels = best_model.classes_

best_confusion_df = pd.DataFrame(
    best_confusion_matrix,
    index=[
        f"真实类别_{label}"
        for label in class_labels
    ],
    columns=[
        f"预测类别_{label}"
        for label in class_labels
    ]
)

best_confusion_df.to_csv(
    "best_svc_confusion_matrix.csv",
    encoding="utf-8-sig"
)


# ============================================================
# 15. 保存最佳特征提取器和最佳SVC
# ============================================================

joblib.dump(
    best_feature_extractor,
    "best_svc_feature_extractor.joblib"
)

joblib.dump(
    best_model,
    "best_svc_model.joblib"
)


# ============================================================
# 16. 保存最佳方法信息
# ============================================================

best_information = pd.DataFrame([
    {
        "最佳方法": best_name,
        "特征数量": int(best_result["特征数量"]),
        "SVC核函数": "linear",
        "SVC_C": SVC_C,
        "验证集Accuracy": best_result["验证集Accuracy"],
        "验证集Macro-F1": best_result["验证集Macro-F1"],
        "验证集HingeLoss": best_result["验证集HingeLoss"],
        "HingeLoss差距": best_result["HingeLoss差距"]
    }
])

best_information.to_csv(
    "best_svc_information.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 65)
print("实验完成")
print("=" * 65)
print(f"全部实验总用时：{total_time:.2f} 秒")
print("对比结果：svc_feature_comparison_results.csv")
print("最佳模型信息：best_svc_information.csv")
print("最佳分类报告：best_svc_classification_report.csv")
print("最佳混淆矩阵：best_svc_confusion_matrix.csv")
print("最佳特征提取器：best_svc_feature_extractor.joblib")
print("最佳SVC模型：best_svc_model.joblib")
print("\n本实验只使用了训练集和验证集，没有读取测试集。")
