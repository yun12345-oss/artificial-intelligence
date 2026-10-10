# mlp_experiment.py

import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import (
    ENGLISH_STOP_WORDS,
    TfidfVectorizer
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    log_loss
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion


# ============================================================
# 1. 基本设置
# ============================================================

RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "train_data.csv"
RESULTS_DIR = BASE_DIR / "mlp_results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. MLP和TF-IDF固定参数
# ============================================================

TFIDF_MAX_FEATURES = 50000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95
TFIDF_SUBLINEAR_TF = True

MLP_ACTIVATION = "relu"
MLP_SOLVER = "adam"
MLP_LEARNING_RATE = 0.001
MLP_MAX_ITER = 300
MLP_VALIDATION_FRACTION = 0.1
MLP_NO_CHANGE = 15


# ============================================================
# 3. 读取带标签训练数据
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
# 4. 停用词
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
# 5. 通用处理带特殊符号的词
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
# 6. 展开否定缩写
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
# 7. 删除新闻组路径
# ============================================================

def remove_newsgroup_paths(text):
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
# 8. 清洗标题或正文
# ============================================================

def process_text_part(
    text
):
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

    # 删除“In article ... writes:”格式
    text = re.sub(
        r"(?i)^In article .*?writes:\s*$",
        " ",
        text,
        flags=re.MULTILINE
    )

    # 删除引用行
    text = re.sub(
        r"^>.*$",
        " ",
        text,
        flags=re.MULTILINE
    )

    # 删除邮箱
    text = re.sub(
        r"\b[\w.-]+@[\w.-]+\.\w+\b",
        " ",
        text
    )

    # 删除签名
    text = re.split(
        r"\n--+\s*\n",
        text,
        maxsplit=1
    )[0]

    # 转为小写
    text = text.lower()

    # 删除标点，保留技术词中的下划线
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
# 9. 提取Subject并清洗正文
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

    # 删除残留邮件头
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

    # 删除正文中残留的Subject
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
# 10. 创建MLP
# ============================================================

def create_mlp(
    hidden_layer_sizes,
    alpha,
    random_state
):
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=MLP_ACTIVATION,
        solver=MLP_SOLVER,
        alpha=alpha,
        learning_rate_init=MLP_LEARNING_RATE,
        max_iter=MLP_MAX_ITER,
        early_stopping=True,
        validation_fraction=(
            MLP_VALIDATION_FRACTION
        ),
        n_iter_no_change=MLP_NO_CHANGE,
        random_state=random_state,
        verbose=False
    )


# ============================================================
# 11. 训练并评价一个MLP
# ============================================================

def evaluate_mlp(
    experiment_name,
    hidden_layer_sizes,
    alpha,
    random_state,
    X_train_tfidf,
    y_train,
    X_val_tfidf,
    y_val
):
    model = create_mlp(
        hidden_layer_sizes=(
            hidden_layer_sizes
        ),
        alpha=alpha,
        random_state=random_state
    )

    print("\n" + "=" * 75)
    print("正在训练：", experiment_name)

    start_time = time.perf_counter()

    model.fit(
        X_train_tfidf,
        y_train
    )

    training_time = (
        time.perf_counter()
        - start_time
    )

    # --------------------------------------------------------
    # 预测类别
    # --------------------------------------------------------

    y_train_pred = model.predict(
        X_train_tfidf
    )

    y_val_pred = model.predict(
        X_val_tfidf
    )

    # --------------------------------------------------------
    # 预测概率
    # --------------------------------------------------------

    train_probabilities = (
        model.predict_proba(
            X_train_tfidf
        )
    )

    val_probabilities = (
        model.predict_proba(
            X_val_tfidf
        )
    )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    train_accuracy = accuracy_score(
        y_train,
        y_train_pred
    )

    val_accuracy = accuracy_score(
        y_val,
        y_val_pred
    )

    # --------------------------------------------------------
    # Macro-F1
    # --------------------------------------------------------

    val_macro_f1 = f1_score(
        y_val,
        y_val_pred,
        average="macro"
    )

    # --------------------------------------------------------
    # Log Loss
    # --------------------------------------------------------

    train_log_loss = log_loss(
        y_train,
        train_probabilities,
        labels=model.classes_
    )

    val_log_loss = log_loss(
        y_val,
        val_probabilities,
        labels=model.classes_
    )

    log_loss_gap = (
        val_log_loss
        - train_log_loss
    )

    internal_best_score = getattr(
        model,
        "best_validation_score_",
        np.nan
    )

    print(
        f"隐藏层："
        f"{hidden_layer_sizes}"
    )

    print(
        f"alpha：{alpha}"
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
        f"训练集 Log Loss："
        f"{train_log_loss:.6f}"
    )

    print(
        f"验证集 Log Loss："
        f"{val_log_loss:.6f}"
    )

    print(
        f"Log Loss差距："
        f"{log_loss_gap:.6f}"
    )

    print(
        f"实际迭代次数："
        f"{model.n_iter_}"
    )

    print(
        f"内部最佳验证Accuracy："
        f"{internal_best_score:.4f}"
    )

    print(
        f"训练时间："
        f"{training_time:.2f}秒"
    )

    result = {
        "实验名称": experiment_name,
        "hidden_layer_sizes": str(
            hidden_layer_sizes
        ),
        "alpha": alpha,
        "activation": MLP_ACTIVATION,
        "random_state": random_state,
        "训练集Accuracy": train_accuracy,
        "验证集Accuracy": val_accuracy,
        "验证集Macro-F1": val_macro_f1,
        "训练集LogLoss": train_log_loss,
        "验证集LogLoss": val_log_loss,
        "LogLoss差距": log_loss_gap,
        "实际迭代次数": model.n_iter_,
        "内部最佳验证Accuracy": (
            internal_best_score
        ),
        "训练时间（秒）": training_time
    }

    return (
        result,
        model,
        y_val_pred
    )


# ============================================================
# 12. 绘制多条训练损失曲线
# ============================================================

def plot_multiple_loss_curves(
    loss_curves,
    title,
    output_path
):
    plt.figure(
        figsize=(10, 6)
    )

    for name, curve in loss_curves.items():
        plt.plot(
            range(1, len(curve) + 1),
            curve,
            label=name
        )

    plt.xlabel("Iteration")
    plt.ylabel("Training Loss")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()


# ============================================================
# 13. 主程序
# ============================================================

def main():
    # 确保结果目录在开始实验时存在
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 13.1 读取数据
    # --------------------------------------------------------

    X, y = load_data()

    print("--- 数据加载完成 ---")
    print(f"带标签数据数量：{len(X)}")
    print(f"标签数量：{len(y)}")
    print(f"类别：{sorted(y.unique())}")

    # --------------------------------------------------------
    # 13.2 划分训练集和验证集
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
    # 13.3 文本清洗
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
    # 13.4 TF-IDF特征提取
    # --------------------------------------------------------

    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        sublinear_tf=(
            TFIDF_SUBLINEAR_TF
        ),
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

    X_train_tfidf = (
        vectorizer.fit_transform(
            X_train_clean
        )
    )

    X_val_tfidf = (
        vectorizer.transform(
            X_val_clean
        )
    )

    print("\n--- TF-IDF转换完成 ---")
    print(
        "训练集特征形状：",
        X_train_tfidf.shape
    )
    print(
        "验证集特征形状：",
        X_val_tfidf.shape
    )

    # ========================================================
    # 14. 第一阶段：调整隐藏层结构
    # ========================================================

    print("\n" + "#" * 75)
    print("--- 第一阶段：调整隐藏层结构 ---")

    hidden_layer_values = [
        (50,),
        (100,),
        (200,),
        (100, 50)
    ]

    fixed_alpha = 0.001

    structure_results = []
    structure_loss_curves = {}

    for hidden_layers in hidden_layer_values:
        experiment_name = (
            f"MLP structure={hidden_layers} "
            f"alpha={fixed_alpha}"
        )

        (
            result,
            model,
            y_val_pred
        ) = evaluate_mlp(
            experiment_name=experiment_name,
            hidden_layer_sizes=hidden_layers,
            alpha=fixed_alpha,
            random_state=RANDOM_STATE,
            X_train_tfidf=X_train_tfidf,
            y_train=y_train,
            X_val_tfidf=X_val_tfidf,
            y_val=y_val
        )

        structure_results.append(
            result
        )

        structure_loss_curves[
            str(hidden_layers)
        ] = model.loss_curve_

    structure_df = pd.DataFrame(
        structure_results
    )

    structure_df = structure_df.sort_values(
        by=[
            "验证集Macro-F1",
            "验证集LogLoss",
            "训练时间（秒）"
        ],
        ascending=[
            False,
            True,
            True
        ]
    ).reset_index(drop=True)

    structure_path = (
        RESULTS_DIR
        / "mlp_structure_results.csv"
    )

    # 防止长时间训练期间结果目录被删除
    structure_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    structure_df.to_csv(
        structure_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n--- 隐藏层结构比较结果 ---")
    print(
        structure_df.to_string(
            index=False
        )
    )

    # 根据结果找到最佳结构
    best_structure_string = (
        structure_df.loc[
            0,
            "hidden_layer_sizes"
        ]
    )

    best_structure = eval(
        best_structure_string,
        {
            "__builtins__": {}
        },
        {}
    )

    print(
        f"\n最佳隐藏层结构："
        f"{best_structure}"
    )

    plot_multiple_loss_curves(
        loss_curves=structure_loss_curves,
        title=(
            "MLP Training Loss - "
            "Different Hidden Layers"
        ),
        output_path=(
            RESULTS_DIR
            / "mlp_structure_loss_curves.png"
        )
    )

    # ========================================================
    # 15. 第二阶段：调整alpha
    # ========================================================

    print("\n" + "#" * 75)
    print("--- 第二阶段：调整alpha ---")

    alpha_values = [
        0.0001,
        0.001,
        0.01,
        0.1
    ]

    alpha_results = []
    alpha_loss_curves = {}

    for alpha_value in alpha_values:
        experiment_name = (
            f"MLP structure={best_structure} "
            f"alpha={alpha_value}"
        )

        (
            result,
            model,
            y_val_pred
        ) = evaluate_mlp(
            experiment_name=experiment_name,
            hidden_layer_sizes=(
                best_structure
            ),
            alpha=alpha_value,
            random_state=RANDOM_STATE,
            X_train_tfidf=X_train_tfidf,
            y_train=y_train,
            X_val_tfidf=X_val_tfidf,
            y_val=y_val
        )

        alpha_results.append(
            result
        )

        alpha_loss_curves[
            f"alpha={alpha_value}"
        ] = model.loss_curve_

    alpha_df = pd.DataFrame(
        alpha_results
    )

    alpha_df = alpha_df.sort_values(
        by=[
            "验证集Macro-F1",
            "验证集LogLoss",
            "训练时间（秒）"
        ],
        ascending=[
            False,
            True,
            True
        ]
    ).reset_index(drop=True)

    alpha_path = (
        RESULTS_DIR
        / "mlp_alpha_results.csv"
    )

    alpha_df.to_csv(
        alpha_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n--- alpha比较结果 ---")
    print(
        alpha_df.to_string(
            index=False
        )
    )

    best_alpha = alpha_df.loc[
        0,
        "alpha"
    ]

    print(
        f"\n最佳alpha："
        f"{best_alpha}"
    )

    plot_multiple_loss_curves(
        loss_curves=alpha_loss_curves,
        title=(
            "MLP Training Loss - "
            "Different Alpha"
        ),
        output_path=(
            RESULTS_DIR
            / "mlp_alpha_loss_curves.png"
        )
    )

    # ========================================================
    # 16. 最佳配置重复运行
    # ========================================================

    print("\n" + "#" * 75)
    print("--- 最佳配置重复运行 ---")

    repeat_random_states = [
        42,
        52,
        62
    ]

    repeat_results = []

    seed_42_model = None
    seed_42_val_pred = None

    for seed in repeat_random_states:
        experiment_name = (
            f"Best MLP seed={seed}"
        )

        (
            result,
            model,
            y_val_pred
        ) = evaluate_mlp(
            experiment_name=experiment_name,
            hidden_layer_sizes=(
                best_structure
            ),
            alpha=best_alpha,
            random_state=seed,
            X_train_tfidf=X_train_tfidf,
            y_train=y_train,
            X_val_tfidf=X_val_tfidf,
            y_val=y_val
        )

        repeat_results.append(
            result
        )

        if seed == 42:
            seed_42_model = model
            seed_42_val_pred = (
                y_val_pred
            )

    repeat_df = pd.DataFrame(
        repeat_results
    )

    repeat_path = (
        RESULTS_DIR
        / "mlp_repeat_results.csv"
    )

    repeat_df.to_csv(
        repeat_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n--- 重复运行结果 ---")
    print(
        repeat_df.to_string(
            index=False
        )
    )

    # ========================================================
    # 17. 计算平均值和标准差
    # ========================================================

    summary = {
        "hidden_layer_sizes": str(
            best_structure
        ),
        "alpha": best_alpha,
        "activation": MLP_ACTIVATION,
        "Accuracy平均值": (
            repeat_df[
                "验证集Accuracy"
            ].mean()
        ),
        "Accuracy标准差": (
            repeat_df[
                "验证集Accuracy"
            ].std(ddof=1)
        ),
        "Macro-F1平均值": (
            repeat_df[
                "验证集Macro-F1"
            ].mean()
        ),
        "Macro-F1标准差": (
            repeat_df[
                "验证集Macro-F1"
            ].std(ddof=1)
        ),
        "LogLoss平均值": (
            repeat_df[
                "验证集LogLoss"
            ].mean()
        ),
        "LogLoss标准差": (
            repeat_df[
                "验证集LogLoss"
            ].std(ddof=1)
        ),
        "训练时间平均值": (
            repeat_df[
                "训练时间（秒）"
            ].mean()
        )
    }

    summary_df = pd.DataFrame([
        summary
    ])

    summary_path = (
        RESULTS_DIR
        / "mlp_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n--- 最佳配置统计结果 ---")

    print(
        f"最佳隐藏层："
        f"{best_structure}"
    )

    print(
        f"最佳alpha："
        f"{best_alpha}"
    )

    print(
        f"激活函数："
        f"{MLP_ACTIVATION}"
    )

    print(
        f"验证Accuracy："
        f"{summary['Accuracy平均值']:.4f} "
        f"± "
        f"{summary['Accuracy标准差']:.4f}"
    )

    print(
        f"验证Macro-F1："
        f"{summary['Macro-F1平均值']:.4f} "
        f"± "
        f"{summary['Macro-F1标准差']:.4f}"
    )

    print(
        f"验证Log Loss："
        f"{summary['LogLoss平均值']:.6f} "
        f"± "
        f"{summary['LogLoss标准差']:.6f}"
    )

    # ========================================================
    # 18. 保存固定seed=42的分类报告
    # ========================================================

    print(
        "\n--- 最佳配置分类报告 "
        "(random_state=42) ---"
    )

    report_text = classification_report(
        y_val,
        seed_42_val_pred,
        digits=4
    )

    print(report_text)

    report_dict = classification_report(
        y_val,
        seed_42_val_pred,
        output_dict=True
    )

    report_df = pd.DataFrame(
        report_dict
    ).transpose()

    report_path = (
        RESULTS_DIR
        / "mlp_classification_report.csv"
    )

    report_df.to_csv(
        report_path,
        encoding="utf-8-sig"
    )

    # ========================================================
    # 19. 保存验证集预测
    # ========================================================

    validation_df = pd.DataFrame({
        "text": X_val.to_numpy(),
        "true_label": y_val.to_numpy(),
        "predicted_label": (
            seed_42_val_pred
        )
    })

    validation_df["correct"] = (
        validation_df["true_label"]
        == validation_df[
            "predicted_label"
        ]
    )

    validation_path = (
        RESULTS_DIR
        / "mlp_validation_results.csv"
    )

    validation_df.to_csv(
        validation_path,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # 20. 绘制最佳模型训练损失曲线
    # ========================================================

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        range(
            1,
            len(
                seed_42_model.loss_curve_
            ) + 1
        ),
        seed_42_model.loss_curve_,
        marker="o",
        markersize=3,
        label="Training Loss"
    )

    plt.xlabel("Iteration")
    plt.ylabel("Training Loss")
    plt.title(
        "Best MLP Training Loss Curve"
    )
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    best_loss_path = (
        RESULTS_DIR
        / "mlp_best_loss_curve.png"
    )

    plt.savefig(
        best_loss_path,
        dpi=300
    )

    plt.close()

    # ========================================================
    # 21. 绘制Early Stopping内部验证准确率
    # ========================================================

    validation_scores = getattr(
        seed_42_model,
        "validation_scores_",
        None
    )

    if validation_scores is not None:
        plt.figure(
            figsize=(8, 5)
        )

        plt.plot(
            range(
                1,
                len(
                    validation_scores
                ) + 1
            ),
            validation_scores,
            marker="o",
            markersize=3,
            label=(
                "Internal Validation Accuracy"
            )
        )

        plt.xlabel("Iteration")
        plt.ylabel("Accuracy")
        plt.title(
            "MLP Early-Stopping "
            "Validation Accuracy"
        )
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()

        validation_curve_path = (
            RESULTS_DIR
            / "mlp_internal_validation_curve.png"
        )

        plt.savefig(
            validation_curve_path,
            dpi=300
        )

        plt.close()

    # ========================================================
    # 22. 输出生成的文件
    # ========================================================

    print("\n--- 实验结果文件 ---")

    print(structure_path)
    print(alpha_path)
    print(repeat_path)
    print(summary_path)
    print(report_path)
    print(validation_path)
    print(best_loss_path)

    print(
        RESULTS_DIR
        / "mlp_structure_loss_curves.png"
    )

    print(
        RESULTS_DIR
        / "mlp_alpha_loss_curves.png"
    )

    if validation_scores is not None:
        print(
            RESULTS_DIR
            / "mlp_internal_validation_curve.png"
        )


if __name__ == "__main__":
    main()
