import time
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

from SVC import clean_text


RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "train_data.csv"
TEST_PATH = BASE_DIR / "test_data_unlabeled.csv"
OUTPUT_PATH = BASE_DIR / "predictions.csv"


def main():
    print("--- 读取训练集和测试集 ---")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    if not {"text", "target"}.issubset(train_df.columns):
        raise ValueError("train_data.csv 必须包含 text 和 target 列")
    if "text" not in test_df.columns:
        raise ValueError("test_data_unlabeled.csv 必须包含 text 列")

    train_texts = train_df["text"].fillna("").astype(str).tolist()
    train_labels = train_df["target"].values
    test_texts = test_df["text"].fillna("").astype(str).tolist()

    print(f"训练样本数量：{len(train_texts)}")
    print(f"测试样本数量：{len(test_texts)}")

    print("\n--- 使用固定策略清洗全部文本 ---")
    start = time.perf_counter()
    train_clean = [clean_text(text) for text in train_texts]
    test_clean = [clean_text(text) for text in test_texts]
    print(f"文本清洗完成，用时：{time.perf_counter() - start:.2f} 秒")

    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        max_features=50000,
        norm="l2",
    )

    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
        max_features=30000,
        norm="l2",
    )

    vectorizer = FeatureUnion([
        ("word_tfidf", word_vectorizer),
        ("char_tfidf", char_vectorizer),
    ])

    print("\n--- 用全部训练集拟合 TF-IDF ---")
    start = time.perf_counter()
    X_train = vectorizer.fit_transform(train_clean)
    X_test = vectorizer.transform(test_clean)
    print(f"训练集特征形状：{X_train.shape}")
    print(f"测试集特征形状：{X_test.shape}")
    print(f"特征提取完成，用时：{time.perf_counter() - start:.2f} 秒")

    model = LinearSVC(
        C=1.0,
        loss="squared_hinge",
        random_state=RANDOM_STATE,
        max_iter=10000,
    )

    print("\n--- 使用全部训练集训练最佳模型 ---")
    start = time.perf_counter()
    model.fit(X_train, train_labels)
    predictions = model.predict(X_test)
    print(f"训练与预测完成，用时：{time.perf_counter() - start:.2f} 秒")

    prediction_df = pd.DataFrame(predictions)

    try:
        prediction_df.to_csv(
            OUTPUT_PATH,
            index=False,
            header=False,
        )
        actual_output_path = OUTPUT_PATH
    except PermissionError:
        # Windows中如果predictions.csv正被Excel或WPS占用，
        # 自动改用备用文件名，避免预测结果丢失。
        actual_output_path = (
            BASE_DIR / "predictions_new.csv"
        )
        prediction_df.to_csv(
            actual_output_path,
            index=False,
            header=False,
        )

        print(
            "警告：predictions.csv 正被其他程序占用，"
            "已改用备用文件名。"
        )

    print("\n--- 测试集预测完成 ---")
    print(f"预测数量：{len(predictions)}")
    print(f"预测标签范围：{sorted(set(predictions.tolist()))}")
    print(f"结果已保存到：{actual_output_path}")


if __name__ == "__main__":
    main()
