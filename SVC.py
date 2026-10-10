# my_experiment.py

import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_extraction.text import (
    ENGLISH_STOP_WORDS,
    TfidfVectorizer
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hinge_loss
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.svm import (
    LinearSVC,
    SVC
)


# ============================================================
# 1. 基本设置
# ============================================================

RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "train_data.csv"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. 读取带标签训练数据
# ============================================================

def load_data():
    train_df = pd.read_csv(
        TRAIN_PATH
    )

    if "text" not in train_df.columns:
        raise ValueError(
            "train_data.csv中缺少text列"
        )

    if "target" not in train_df.columns:
        raise ValueError(
            "train_data.csv中缺少target列"
        )

    X = (
        train_df["text"]
        .fillna("")
        .astype(str)
    )

    y = train_df["target"]

    return X, y


# ============================================================
# 3. 停用词
# ============================================================

custom_stop_words = set(
    ENGLISH_STOP_WORDS
)

# 保留否定词
custom_stop_words -= {
    "no",
    "not",
    "never",
    "without"
}

# ============================================================
# 4. 通用处理带特殊符号的词
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


# ============================================================
# 5. 展开否定缩写
# ============================================================

def normalize_contractions(text):
    replacements = [
        (
            r"(?i)\bwon't\b",
            "will not"
        ),
        (
            r"(?i)\bcan't\b",
            "can not"
        ),
        (
            r"(?i)\bdon't\b",
            "do not"
        ),
        (
            r"(?i)\bdoesn't\b",
            "does not"
        ),
        (
            r"(?i)\bdidn't\b",
            "did not"
        ),
        (
            r"(?i)\bisn't\b",
            "is not"
        ),
        (
            r"(?i)\baren't\b",
            "are not"
        ),
        (
            r"(?i)\bwasn't\b",
            "was not"
        ),
        (
            r"(?i)\bweren't\b",
            "were not"
        ),
        (
            r"(?i)\bhasn't\b",
            "has not"
        ),
        (
            r"(?i)\bhaven't\b",
            "have not"
        ),
        (
            r"(?i)\bhadn't\b",
            "had not"
        ),
        (
            r"(?i)\bshouldn't\b",
            "should not"
        ),
        (
            r"(?i)\bwouldn't\b",
            "would not"
        ),
        (
            r"(?i)\bcouldn't\b",
            "could not"
        ),
        (
            r"(?i)\bmustn't\b",
            "must not"
        )
    ]

    for pattern, replacement in replacements:
        text = re.sub(
            pattern,
            replacement,
            text
        )

    return text


# ============================================================
# 6. 删除新闻组路径
# ============================================================

def remove_newsgroup_paths(text):
    """
    删除可能直接暴露类别的信息，例如：

    comp.os.linux
    comp.os.386bsd.announce
    rec.sport.baseball
    sci.space
    """

    return re.sub(
        (
            r"(?i)\b"
            r"(?:comp|sci|rec|talk|soc|misc|alt)"
            r"(?:\.[a-z0-9_+-]+)+"
            r"\b"
        ),
        " ",
        text
    )


# ============================================================
# 7. 清洗标题或正文
# ============================================================

def process_text_part(
    text
):
    text = str(text)

    # 通用处理带特殊符号的词
    text = normalize_symbolic_tokens(
        text
    )

    # 展开否定缩写
    text = normalize_contractions(
        text
    )

    # 删除新闻组路径
    text = remove_newsgroup_paths(
        text
    )

    # 删除“In article ... writes:”格式
    text = re.sub(
        r"(?i)^In article .*?writes:\s*$",
        " ",
        text,
        flags=re.MULTILINE
    )

    # 删除以 > 开头的引用行
    text = re.sub(
        r"^>.*$",
        " ",
        text,
        flags=re.MULTILINE
    )

    # 删除邮箱地址
    text = re.sub(
        r"\b[\w.-]+@[\w.-]+\.\w+\b",
        " ",
        text
    )

    # 删除签名分隔符之后的内容
    text = re.split(
        r"\n--+\s*\n",
        text,
        maxsplit=1
    )[0]

    # 转换为小写
    text = text.lower()

    # 删除标点，保留字母、数字、下划线和空格
    text = re.sub(
        r"[^a-z0-9_\s]",
        " ",
        text
    )

    words = text.split()

    # 删除停用词和单字符
    words = [
        word
        for word in words
        if (
            word not in custom_stop_words
            and len(word) > 1
        )
    ]

    return " ".join(words)


# ============================================================
# 8. 提取Subject并清洗正文
# ============================================================

def clean_text(text):
    text = str(text)

    # 提取Subject
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

    # 第一处空行之前一般是邮件头
    parts = re.split(
        r"\n\s*\n",
        text,
        maxsplit=1
    )

    if len(parts) == 2:
        body = parts[1]
    else:
        body = text

    # 删除可能残留的邮件头
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

    # 删除正文中残留的Subject行
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

    # 标题重复两次，提高标题权重
    return (
        f"{subject_clean} "
        f"{subject_clean} "
        f"{subject_clean} "
        f"{body_clean}"
    ).strip()


# ============================================================
# 9. 创建全部调参配置
# ============================================================

def create_experiments():
    experiments = []

    # --------------------------------------------------------
    # 9.1 基础实验：LinearSVC调节C
    # --------------------------------------------------------

    linear_svc_c_values = [
        0.01,
        0.1,
        1.0,
        10.0,
        100.0
    ]

    for c_value in linear_svc_c_values:
        experiments.append({
            "name": (
                f"LinearSVC C={c_value}"
            ),
            "family": "LinearSVC",
            "kernel": "linear",
            "C": c_value,
            "gamma": "not_used",
            "model": LinearSVC(
                C=c_value,
                loss="squared_hinge",
                random_state=RANDOM_STATE,
                max_iter=10000
            )
        })

    # --------------------------------------------------------
    # 9.2 扩展实验：SVC线性核
    # --------------------------------------------------------

    svc_linear_c_values = [
        0.1,
        1.0,
        10.0,
        100.0
    ]

    for c_value in svc_linear_c_values:
        experiments.append({
            "name": (
                f"SVC linear C={c_value}"
            ),
            "family": "SVC-linear",
            "kernel": "linear",
            "C": c_value,
            "gamma": "not_used",
            "model": SVC(
                kernel="linear",
                C=c_value,
                decision_function_shape="ovr",
                cache_size=1024
            )
        })

    # --------------------------------------------------------
    # 9.3 扩展实验：SVC RBF核
    # --------------------------------------------------------

    svc_rbf_c_values = [
        0.1,
        1.0,
        10.0,
        100.0
    ]

    svc_rbf_gamma_values = [
        "scale",
        "auto",
        0.1,
        1.0
    ]

    for c_value in svc_rbf_c_values:
        for gamma_value in (
            svc_rbf_gamma_values
        ):
            experiments.append({
                "name": (
                    f"SVC rbf C={c_value} "
                    f"gamma={gamma_value}"
                ),
                "family": "SVC-rbf",
                "kernel": "rbf",
                "C": c_value,
                "gamma": gamma_value,
                "model": SVC(
                    kernel="rbf",
                    C=c_value,
                    gamma=gamma_value,
                    decision_function_shape="ovr",
                    cache_size=1024
                )
            })

    return experiments


# ============================================================
# 10. 训练和评价单个模型
# ============================================================

def evaluate_model(
    experiment,
    X_train_tfidf,
    y_train,
    X_val_tfidf,
    y_val
):
    model = experiment["model"]

    print("\n" + "=" * 75)
    print(
        "正在训练：",
        experiment["name"]
    )

    start_time = time.perf_counter()

    model.fit(
        X_train_tfidf,
        y_train
    )

    training_time = (
        time.perf_counter()
        - start_time
    )

    # SVC在8万维特征上计算训练集决策分数非常慢。
    # 训练指标使用固定子集估计；LinearSVC仍使用完整训练集。
    if isinstance(model, SVC):
        train_evaluation_size = min(
            1000,
            X_train_tfidf.shape[0]
        )

        X_train_evaluation = X_train_tfidf[
            :train_evaluation_size
        ]

        y_train_evaluation = y_train.iloc[
            :train_evaluation_size
        ]

        print(
            "SVC训练指标评估样本数：",
            train_evaluation_size
        )
    else:
        X_train_evaluation = X_train_tfidf
        y_train_evaluation = y_train

    # 获取决策分数。后面直接由决策分数生成预测类别，
    # 避免predict和decision_function重复执行昂贵的核计算。
    train_decision_scores = (
        model.decision_function(
            X_train_evaluation
        )
    )

    val_decision_scores = (
        model.decision_function(
            X_val_tfidf
        )
    )

    if train_decision_scores.ndim == 1:
        y_train_pred = model.classes_[
            (train_decision_scores > 0).astype(int)
        ]
    else:
        y_train_pred = model.classes_[
            train_decision_scores.argmax(axis=1)
        ]

    if val_decision_scores.ndim == 1:
        y_val_pred = model.classes_[
            (val_decision_scores > 0).astype(int)
        ]
    else:
        y_val_pred = model.classes_[
            val_decision_scores.argmax(axis=1)
        ]

    # Accuracy
    train_accuracy = accuracy_score(
        y_train_evaluation,
        y_train_pred
    )

    val_accuracy = accuracy_score(
        y_val,
        y_val_pred
    )

    # Macro-F1
    val_macro_f1 = f1_score(
        y_val,
        y_val_pred,
        average="macro"
    )

    # Hinge Loss
    train_hinge_loss = hinge_loss(
        y_train_evaluation,
        train_decision_scores,
        labels=model.classes_
    )

    val_hinge_loss = hinge_loss(
        y_val,
        val_decision_scores,
        labels=model.classes_
    )

    hinge_loss_gap = (
        val_hinge_loss
        - train_hinge_loss
    )

    print(
        f"训练集 Accuracy："
        f"{train_accuracy:.4f}"
    )

    print(
        f"验证集 Accuracy："
        f"{val_accuracy:.4f}"
    )

    print(
        f"验证集 Macro-F1："
        f"{val_macro_f1:.4f}"
    )

    print(
        f"训练集 Hinge Loss："
        f"{train_hinge_loss:.6f}"
    )

    print(
        f"验证集 Hinge Loss："
        f"{val_hinge_loss:.6f}"
    )

    print(
        f"Hinge Loss差距："
        f"{hinge_loss_gap:.6f}"
    )

    print(
        f"训练时间："
        f"{training_time:.2f}秒"
    )

    result = {
        "实验名称": experiment["name"],
        "模型系列": experiment["family"],
        "kernel": experiment["kernel"],
        "C": experiment["C"],
        "gamma": str(
            experiment["gamma"]
        ),
        "训练集Accuracy": train_accuracy,
        "验证集Accuracy": val_accuracy,
        "验证集Macro-F1": val_macro_f1,
        "训练集HingeLoss": train_hinge_loss,
        "验证集HingeLoss": val_hinge_loss,
        "HingeLoss差距": hinge_loss_gap,
        "训练时间（秒）": training_time
    }

    return (
        result,
        model,
        y_val_pred
    )


# ============================================================
# 11. 绘制一个模型系列的损失曲线
# ============================================================

def plot_loss_curve(
    family_df,
    title,
    output_path
):
    plt.figure(
        figsize=(8, 5)
    )

    plt.semilogx(
        family_df["C"],
        family_df["训练集HingeLoss"],
        marker="o",
        label="Training Hinge Loss"
    )

    plt.semilogx(
        family_df["C"],
        family_df["验证集HingeLoss"],
        marker="o",
        label="Validation Hinge Loss"
    )

    plt.xlabel("C")
    plt.ylabel("Hinge Loss")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()


# ============================================================
# 12. 主程序
# ============================================================

def main():
    # 确保结果目录在开始实验时存在
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 12.1 读取数据
    # --------------------------------------------------------

    X, y = load_data()

    print("--- 数据加载完成 ---")
    print(f"带标签数据数量：{len(X)}")
    print(f"标签数量：{len(y)}")
    print(f"类别：{sorted(y.unique())}")

    # --------------------------------------------------------
    # 12.2 划分训练集和验证集
    # --------------------------------------------------------

    X_train, X_val, y_train, y_val = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y
        )
    )

    print("\n--- 数据划分完成 ---")
    print(f"训练集数量：{len(X_train)}")
    print(f"验证集数量：{len(X_val)}")

    # --------------------------------------------------------
    # 12.3 清洗文本
    # --------------------------------------------------------

    print("\n--- 开始清洗文本 ---")

    cleaning_start_time = (
        time.perf_counter()
    )

    X_train_clean = [
        clean_text(text)
        for text in X_train
    ]

    X_val_clean = [
        clean_text(text)
        for text in X_val
    ]

    cleaning_time = (
        time.perf_counter()
        - cleaning_start_time
    )

    print(
        f"文本清洗完成，用时："
        f"{cleaning_time:.2f}秒"
    )

    print("\n--- 清洗前示例 ---")
    print(
        X_train.iloc[0][:600]
    )

    print("\n--- 清洗后示例 ---")
    print(
        X_train_clean[0][:600]
    )

    # --------------------------------------------------------
    # 12.4 TF-IDF特征提取
    # --------------------------------------------------------

    # 方法5：单词TF-IDF + 字符TF-IDF

    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        max_features=50000,
        norm="l2"
    )

    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
        max_features=30000,
        norm="l2"
    )

    vectorizer = FeatureUnion([
        ("word_tfidf", word_vectorizer),
        ("char_tfidf", char_vectorizer)
    ])

    feature_start_time = time.perf_counter()

    X_train_tfidf = vectorizer.fit_transform(
        X_train_clean
    )

    X_val_tfidf = vectorizer.transform(
        X_val_clean
    )

    feature_time = (
        time.perf_counter()
        - feature_start_time
    )

    print("\n--- TF-IDF转换完成 ---")
    print(
        f"训练集特征形状："
        f"{X_train_tfidf.shape}"
    )
    print(
        f"验证集特征形状："
        f"{X_val_tfidf.shape}"
    )
    print(
        f"特征提取用时："
        f"{feature_time:.2f}秒"
    )

    # --------------------------------------------------------
    # 12.5 创建全部调参实验
    # --------------------------------------------------------

    experiments = create_experiments()

    total_experiments = len(
        experiments
    )

    print(
        f"\n总实验数量："
        f"{total_experiments}"
    )

    results = []
    trained_records = []

    # --------------------------------------------------------
    # 12.6 依次训练全部模型
    # --------------------------------------------------------

    for index, experiment in enumerate(
        experiments,
        start=1
    ):
        print(
            f"\n当前进度："
            f"{index}/{total_experiments}"
        )

        (
            result,
            trained_model,
            y_val_pred
        ) = evaluate_model(
            experiment=experiment,
            X_train_tfidf=X_train_tfidf,
            y_train=y_train,
            X_val_tfidf=X_val_tfidf,
            y_val=y_val
        )

        results.append(result)

        trained_records.append({
            "name": experiment["name"],
            "model": trained_model,
            "y_val_pred": y_val_pred,
            "result": result
        })

    # --------------------------------------------------------
    # 12.7 整理全部结果
    # --------------------------------------------------------

    result_df = pd.DataFrame(
        results
    )

    result_df = result_df.sort_values(
        by=[
            "验证集Macro-F1",
            "验证集HingeLoss",
            "训练时间（秒）"
        ],
        ascending=[
            False,
            True,
            True
        ]
    ).reset_index(drop=True)

    print("\n" + "=" * 75)
    print("--- 全部调参结果 ---")

    print(
        result_df.to_string(
            index=False
        )
    )

    result_path = (
        RESULTS_DIR
        / "svm_tuning_results.csv"
    )

    # 防止长时间训练期间结果目录被删除
    result_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result_df.to_csv(
        result_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"\n调参结果已保存到："
        f"{result_path}"
    )

    # --------------------------------------------------------
    # 12.8 输出各模型系列最佳结果
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("--- 各模型系列最佳方案 ---")

    for family_name in [
        "LinearSVC",
        "SVC-linear",
        "SVC-rbf"
    ]:
        family_df = result_df[
            result_df["模型系列"]
            == family_name
        ].copy()

        family_df = family_df.sort_values(
            by=[
                "验证集Macro-F1",
                "验证集HingeLoss"
            ],
            ascending=[
                False,
                True
            ]
        )

        best_family_row = (
            family_df.iloc[0]
        )

        print(
            f"\n模型系列："
            f"{family_name}"
        )

        print(
            f"实验名称："
            f"{best_family_row['实验名称']}"
        )

        print(
            f"验证集Accuracy："
            f"{best_family_row['验证集Accuracy']:.4f}"
        )

        print(
            f"验证集Macro-F1："
            f"{best_family_row['验证集Macro-F1']:.4f}"
        )

        print(
            f"验证集Hinge Loss："
            f"{best_family_row['验证集HingeLoss']:.6f}"
        )

        print(
            f"训练时间："
            f"{best_family_row['训练时间（秒）']:.2f}秒"
        )

    # --------------------------------------------------------
    # 12.9 获取整体最佳模型
    # --------------------------------------------------------

    best_experiment_name = result_df.loc[
        0,
        "实验名称"
    ]

    best_record = next(
        record
        for record in trained_records
        if (
            record["name"]
            == best_experiment_name
        )
    )

    best_val_pred = best_record[
        "y_val_pred"
    ]

    best_result = best_record[
        "result"
    ]

    print("\n" + "=" * 75)
    print("--- 整体最佳模型 ---")

    print(
        f"实验名称："
        f"{best_experiment_name}"
    )

    print(
        f"验证集Accuracy："
        f"{best_result['验证集Accuracy']:.4f}"
    )

    print(
        f"验证集Macro-F1："
        f"{best_result['验证集Macro-F1']:.4f}"
    )

    print(
        f"训练集Hinge Loss："
        f"{best_result['训练集HingeLoss']:.6f}"
    )

    print(
        f"验证集Hinge Loss："
        f"{best_result['验证集HingeLoss']:.6f}"
    )

    print(
        f"Hinge Loss差距："
        f"{best_result['HingeLoss差距']:.6f}"
    )

    # --------------------------------------------------------
    # 12.10 保存最佳模型分类报告
    # --------------------------------------------------------

    print("\n--- 最佳模型分类报告 ---")

    report_text = classification_report(
        y_val,
        best_val_pred,
        digits=4
    )

    print(report_text)

    report_dict = classification_report(
        y_val,
        best_val_pred,
        output_dict=True
    )

    report_df = pd.DataFrame(
        report_dict
    ).transpose()

    report_path = (
        RESULTS_DIR
        / "best_svm_classification_report.csv"
    )

    report_df.to_csv(
        report_path,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # 12.11 保存最佳模型验证结果
    # --------------------------------------------------------

    validation_result_df = pd.DataFrame({
        "text": X_val.to_numpy(),
        "true_label": y_val.to_numpy(),
        "predicted_label": best_val_pred
    })

    validation_result_df[
        "correct"
    ] = (
        validation_result_df[
            "true_label"
        ]
        == validation_result_df[
            "predicted_label"
        ]
    )

    validation_path = (
        RESULTS_DIR
        / "best_svm_validation_results.csv"
    )

    validation_result_df.to_csv(
        validation_path,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # 12.12 绘制LinearSVC损失曲线
    # --------------------------------------------------------

    linear_svc_df = pd.DataFrame(
        results
    )

    linear_svc_df = linear_svc_df[
        linear_svc_df["模型系列"]
        == "LinearSVC"
    ].sort_values(
        by="C"
    )

    plot_loss_curve(
        family_df=linear_svc_df,
        title=(
            "LinearSVC Hinge Loss "
            "under Different C"
        ),
        output_path=(
            RESULTS_DIR
            / "linearsvc_loss_curve.png"
        )
    )

    # --------------------------------------------------------
    # 12.13 绘制LinearSVC性能曲线
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.semilogx(
        linear_svc_df["C"],
        linear_svc_df[
            "验证集Accuracy"
        ],
        marker="o",
        label="Validation Accuracy"
    )

    plt.semilogx(
        linear_svc_df["C"],
        linear_svc_df[
            "验证集Macro-F1"
        ],
        marker="o",
        label="Validation Macro-F1"
    )

    plt.xlabel("C")
    plt.ylabel("Score")
    plt.title(
        "LinearSVC Validation Performance"
    )
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR
        / "linearsvc_score_curve.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------------------------
    # 12.14 绘制SVC线性核损失曲线
    # --------------------------------------------------------

    svc_linear_df = pd.DataFrame(
        results
    )

    svc_linear_df = svc_linear_df[
        svc_linear_df["模型系列"]
        == "SVC-linear"
    ].sort_values(
        by="C"
    )

    plot_loss_curve(
        family_df=svc_linear_df,
        title=(
            "Linear-kernel SVC Hinge Loss "
            "under Different C"
        ),
        output_path=(
            RESULTS_DIR
            / "svc_linear_loss_curve.png"
        )
    )

    # --------------------------------------------------------
    # 12.15 绘制RBF核损失曲线
    # --------------------------------------------------------

    svc_rbf_df = pd.DataFrame(
        results
    )

    svc_rbf_df = svc_rbf_df[
        svc_rbf_df["模型系列"]
        == "SVC-rbf"
    ].copy()

    plt.figure(
        figsize=(9, 6)
    )

    for gamma_value in (
        svc_rbf_df["gamma"].unique()
    ):
        gamma_df = svc_rbf_df[
            svc_rbf_df["gamma"]
            == gamma_value
        ].sort_values(
            by="C"
        )

        plt.semilogx(
            gamma_df["C"],
            gamma_df[
                "验证集HingeLoss"
            ],
            marker="o",
            label=(
                f"gamma={gamma_value}"
            )
        )

    plt.xlabel("C")
    plt.ylabel(
        "Validation Hinge Loss"
    )
    plt.title(
        "RBF SVC Validation Loss"
    )
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR
        / "svc_rbf_loss_curve.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------------------------
    # 12.16 绘制RBF核Macro-F1曲线
    # --------------------------------------------------------

    plt.figure(
        figsize=(9, 6)
    )

    for gamma_value in (
        svc_rbf_df["gamma"].unique()
    ):
        gamma_df = svc_rbf_df[
            svc_rbf_df["gamma"]
            == gamma_value
        ].sort_values(
            by="C"
        )

        plt.semilogx(
            gamma_df["C"],
            gamma_df[
                "验证集Macro-F1"
            ],
            marker="o",
            label=(
                f"gamma={gamma_value}"
            )
        )

    plt.xlabel("C")
    plt.ylabel(
        "Validation Macro-F1"
    )
    plt.title(
        "RBF SVC Validation Performance"
    )
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR
        / "svc_rbf_score_curve.png",
        dpi=300
    )

    plt.close()

    print("\n--- 所有结果文件 ---")

    print(result_path)
    print(report_path)
    print(validation_path)

    print(
        RESULTS_DIR
        / "linearsvc_loss_curve.png"
    )

    print(
        RESULTS_DIR
        / "linearsvc_score_curve.png"
    )

    print(
        RESULTS_DIR
        / "svc_linear_loss_curve.png"
    )

    print(
        RESULTS_DIR
        / "svc_rbf_loss_curve.png"
    )

    print(
        RESULTS_DIR
        / "svc_rbf_score_curve.png"
    )


if __name__ == "__main__":
    main()
