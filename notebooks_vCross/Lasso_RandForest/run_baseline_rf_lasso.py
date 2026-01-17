import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, roc_curve, precision_recall_curve, recall_score, precision_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# 指定 Python 环境说明
# 请使用: /opt/micromamba/envs/MiCoGPT_dev/bin/python run_baseline_rf_lasso.py

def load_data(pkl_path):
    print(f"Loading corpus from {pkl_path} ...")
    with open(pkl_path, "rb") as f:
        corpus = pickle.load(f)
    print(f"Loaded corpus with {len(corpus)} samples.")
    return corpus

def extract_features_and_labels(corpus, label_col="Is_Healthy"):
    """
    将 MiCoGPTCorpus 对象转换为传统的特征矩阵 (X) 和标签向量 (y)。
    X: 相对丰度矩阵 (Samples x Taxa)
    y: 0/1 标签
    """
    print("Extracting features (OTU table) and labels...")
    
    # 1. 获取 Metadata
    meta = corpus.metadata
    
    # 2. 构建特征矩阵 X
    # MiCoGPTCorpus 存储的是 Token IDs (序列)，我们需要统计每个样本中各个 Token (Taxa) 的出现次数
    # 也就是还原回 OTU Table。
    # 注意：corpus[i] 返回的是处理后的 input_ids，可能已经截断或padding。
    # 更准确的方法是直接利用原始的 input_ids list (如果存在) 或者重新统计。
    # 这里我们假设 corpus.input_ids 是一个 list of list (Token IDs)
    
    # 获取词表大小
    vocab_size = corpus.tokenizer.vocab_size
    num_samples = len(corpus)
    
    # 初始化稀疏矩阵或稠密矩阵
    # 为了简单起见，这里用 numpy array (如果内存不够可能需要 scipy.sparse)
    X = np.zeros((num_samples, vocab_size), dtype=np.float32)
    
    # 填充矩阵
    # 遍历所有样本
    # 注意：corpus 可能是一个 list 或自定义对象，我们需要访问原始数据
    # 查看 corpus 源码结构，通常 corpus.input_ids 存储了序列
    
    input_ids_list = corpus.input_ids
    
    for i, tokens in enumerate(tqdm(input_ids_list, desc="Building OTU Table")):
        # tokens 是一个 list of int
        # 统计词频 (Count)
        unique, counts = np.unique(tokens, return_counts=True)
        # 过滤掉 special tokens (0: pad, 1: unk, 2: bos, 3: eos)
        # 假设 0-3 是特殊 token，具体看 tokenizer 定义
        # 通常 Taxa ID 从 4 开始
        valid_mask = unique >= 4 
        X[i, unique[valid_mask]] = counts[valid_mask]
        
    # TSS 标准化 (Relative Abundance)
    row_sums = X.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1 # 避免除零
    X_tss = X / row_sums
    
    # 3. 构建标签 y
    # 标签列可能包含 NaN
    y = meta[label_col].values
    
    # 处理标签: False -> 1 (Disease), True -> 0 (Healthy)
    # 根据之前的 R 代码逻辑: Is_Healthy == FALSE -> 1
    y_encoded = np.zeros(len(y), dtype=int)
    # 只有明确为 False 的才设为 1，明确为 True 的设为 0
    # NaN 的后面会过滤掉
    mask_disease = (y == False)
    y_encoded[mask_disease] = 1
    
    # 记录哪些样本有有效标签
    valid_label_mask = ~pd.isna(y)
    
    return X_tss, y_encoded, meta, valid_label_mask

def calculate_metrics_at_thresholds(y_true, y_prob, model_name):
    """
    计算两组指标:
    1. Default (t=0.5)
    2. Best (Max F1)
    """
    # 1. Default (t=0.5)
    y_pred_default = (y_prob >= 0.5).astype(int)
    
    res_default = {
        "Model": model_name,
        "Type": "Default (t=0.5)",
        "Threshold": 0.5,
        "AUC": roc_auc_score(y_true, y_prob),
        "Accuracy": accuracy_score(y_true, y_pred_default),
        "Sensitivity": recall_score(y_true, y_pred_default), # Sensitivity = Recall
        "Precision": precision_score(y_true, y_pred_default),
        "F1": f1_score(y_true, y_pred_default)
    }
    
    # 2. Best (Max Accuracy)
    # 使用 roc_curve 得到的阈值来计算 Accuracy
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    
    # 计算正负样本数
    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)
    
    # 计算每个阈值下的 Accuracy
    # Acc = (TP + TN) / Total = (TPR * P + (1 - FPR) * N) / (P + N)
    accuracies = (tpr * P + (1 - fpr) * N) / (P + N)
    
    best_idx = np.argmax(accuracies)
    best_threshold = thresholds[best_idx]
    best_acc = accuracies[best_idx]
    
    # 注意: roc_curve 的 thresholds[0] 通常是 max(score) + 1，这个点预测全为 0 (TPR=0, FPR=0)
    # 如果 best_threshold > 1，需要特殊处理，但通常 argmax 不会选到它除非模型极差
    if best_threshold > 1:
        best_threshold = 1.0
    
    y_pred_best = (y_prob >= best_threshold).astype(int)
    
    res_best = {
        "Model": model_name,
        "Type": "Best (Max Accuracy)",
        "Threshold": best_threshold,
        "AUC": roc_auc_score(y_true, y_prob), 
        "Accuracy": best_acc,
        "Sensitivity": recall_score(y_true, y_pred_best),
        "Precision": precision_score(y_true, y_pred_best),
        "F1": f1_score(y_true, y_pred_best)
    }
    
    return res_default, res_best

def save_roc_data(y_true, y_prob, model_name, output_dir):
    """
    保存 ROC 曲线数据，格式兼容 plot_roc_curves.R
    需要的列: fpr, tpr, auc
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    
    # 构造 DataFrame
    df = pd.DataFrame({
        "fpr": fpr,
        "tpr": tpr,
        # 阈值长度比 fpr/tpr 少 1 (sklearn 约定)，这里补一个
        "t": np.concatenate([thresholds, [0]])[:len(fpr)], 
        "auc": [auc_score] * len(fpr)
    })
    
    filename = f"{output_dir}/{model_name}_roc.csv"
    df.to_csv(filename, index=False)
    print(f"ROC data saved to {filename}")
    return filename

def train_eval_rf(X_train, y_train, X_test, y_test, output_dir):
    print("\nTraining Random Forest...")
    # 参数参考 R 代码: ntree=92, mtry=sqrt
    rf = RandomForestClassifier(
        n_estimators=92,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    rf.fit(X_train, y_train)
    
    # 预测概率
    y_pred_prob = rf.predict_proba(X_test)[:, 1]
    
    # 计算详细指标
    res_default, res_best = calculate_metrics_at_thresholds(y_test, y_pred_prob, "RandomForest")
    
    # 保存 ROC 数据
    save_roc_data(y_test, y_pred_prob, "RandomForest", output_dir)
    
    return [res_default, res_best], y_pred_prob

def train_eval_lasso(X_train, y_train, X_test, y_test, output_dir):
    print("\nTraining Lasso (Logistic Regression l1)...")
    # 参数参考 R 代码: alpha=1 (Lasso), lambda=0.001
    # sklearn C = 1/lambda = 1/0.001 = 1000
    lasso = LogisticRegression(
        penalty='l1',
        C=1.0/0.001, 
        solver='liblinear', # liblinear 支持 l1
        random_state=42,
        class_weight='balanced',
        max_iter=1000
    )
    lasso.fit(X_train, y_train)
    
    # 预测概率
    y_pred_prob = lasso.predict_proba(X_test)[:, 1]
    
    # 计算详细指标
    res_default, res_best = calculate_metrics_at_thresholds(y_test, y_pred_prob, "Lasso")
    
    # 保存 ROC 数据
    save_roc_data(y_test, y_pred_prob, "Lasso", output_dir)
    
    return [res_default, res_best], y_pred_prob

def main():
    # 1. 路径设置
    pkl_path = "../../data/vCross/ResMicroDB_90338_vCross.pkl"
    output_dir = "Lasso_RandForest"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 2. 加载数据
    corpus = load_data(pkl_path)
    
    # 3. 特征提取
    X, y, meta, valid_mask = extract_features_and_labels(corpus)
    
    # 4. 数据划分 (Train on A, Test on B)
    # 必须同时满足: valid_mask (标签存在) AND split_group 条件
    train_mask = valid_mask & (meta["Split_Group"] == "A")
    test_mask = valid_mask & (meta["Split_Group"] == "B")
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    print(f"\nData Split:")
    print(f"Train Set (Group A): {X_train.shape[0]} samples")
    print(f"Test Set (Group B):  {X_test.shape[0]} samples")
    print(f"Feature Dimension:   {X_train.shape[1]}")
    
    # 5. 训练与评估
    all_results = []
    
    # Random Forest
    rf_results_list, rf_probs = train_eval_rf(X_train, y_train, X_test, y_test, output_dir)
    all_results.extend(rf_results_list)
    print(f"RF Results (Default): {rf_results_list[0]}")
    print(f"RF Results (Best Acc): {rf_results_list[1]}")
    
    # Lasso
    lasso_results_list, lasso_probs = train_eval_lasso(X_train, y_train, X_test, y_test, output_dir)
    all_results.extend(lasso_results_list)
    print(f"Lasso Results (Default): {lasso_results_list[0]}")
    print(f"Lasso Results (Best Acc): {lasso_results_list[1]}")
    
    # 6. 保存结果
    res_df = pd.DataFrame(all_results)
    # 调整列顺序
    cols = ["Model", "Type", "Threshold", "AUC", "Accuracy", "Sensitivity", "Precision", "F1"]
    res_df = res_df[cols]
    
    res_df.to_csv(f"{output_dir}/baseline_detailed_metrics.csv", index=False)
    print(f"\nDetailed metrics saved to {output_dir}/baseline_detailed_metrics.csv")
    
    # 7. 绘制 ROC 曲线对比
    plt.figure(figsize=(8, 6))
    
    # RF ROC
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
    rf_auc = rf_results_list[0]["AUC"]
    plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {rf_auc:.3f})')
    
    # Lasso ROC
    fpr_lasso, tpr_lasso, _ = roc_curve(y_test, lasso_probs)
    lasso_auc = lasso_results_list[0]["AUC"]
    plt.plot(fpr_lasso, tpr_lasso, label=f'Lasso (AUC = {lasso_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve: Baseline Models (Train A -> Test B)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/baseline_roc_curve.png")
    print(f"ROC plot saved to {output_dir}/baseline_roc_curve.png")

if __name__ == "__main__":
    main()

