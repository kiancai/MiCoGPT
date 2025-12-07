import os
import json
import shutil
import glob
import numpy as np

# === 配置需要处理的目录 ===
TARGET_DIRS = [
    "../models/finetuned_vCross_zscore",
    "../models/finetuned_vCross_baseline",
    "../models/finetuned_vCross_onlyValue",
    "../models/finetuned_vCross_value_condition",
    "../models/finetuned_vCross_base"
]

PATIENCE = 5  # 模拟 Early Stopping 的耐心值

def get_sorted_logs(model_dir):
    """
    从 checkpoint 文件夹中提取所有验证日志，并按 step/epoch 排序
    """
    ckpt_dirs = glob.glob(os.path.join(model_dir, "checkpoint-*"))
    if not ckpt_dirs:
        return []

    logs = []
    for ckpt in ckpt_dirs:
        state_file = os.path.join(ckpt, "trainer_state.json")
        if not os.path.exists(state_file):
            continue
        
        try:
            with open(state_file, 'r') as f:
                data = json.load(f)
            
            # 提取 step
            step = int(ckpt.split('-')[-1])
            
            # 寻找该 checkpoint 对应的 eval_loss
            # 注意：必须找到 log_history 中 step 匹配的那条
            for record in data.get("log_history", []):
                if record.get("step") == step and "eval_loss" in record:
                    logs.append({
                        "step": step,
                        "epoch": record.get("epoch"),
                        "val_loss": record["eval_loss"],
                        "ckpt_path": ckpt
                    })
                    break # 找到即止
                    
        except Exception:
            pass
            
    # 按 step 排序，模拟训练的时间轴
    logs.sort(key=lambda x: x["step"])
    return logs

def simulate_early_stopping(logs):
    """
    核心逻辑：模拟 Early Stopping 过程
    """
    if not logs:
        return None, "No logs"

    best_log = logs[0]
    best_loss = logs[0]["val_loss"]
    patience_counter = 0
    stop_reason = "Finished all logs (No early stopping triggered)"
    
    print(f"     [Start] Initial Loss: {best_loss:.6f} at Epoch {best_log['epoch']}")

    # 从第二个点开始遍历
    for i in range(1, len(logs)):
        curr = logs[i]
        curr_loss = curr["val_loss"]
        
        if curr_loss < best_loss:
            # Loss 改善
            print(f"     [Improve] Epoch {curr['epoch']}: Loss {best_loss:.6f} -> {curr_loss:.6f}")
            best_loss = curr_loss
            best_log = curr
            patience_counter = 0
        else:
            # Loss 未改善
            patience_counter += 1
            # print(f"     [Patience] Epoch {curr['epoch']}: Loss {curr_loss:.6f} >= Best {best_loss:.6f} ({patience_counter}/{PATIENCE})")
            
            if patience_counter >= PATIENCE:
                stop_reason = f"Early Stopping Triggered at Epoch {curr['epoch']}"
                print(f"     [STOP] Patience exhausted ({PATIENCE}). Stopping.")
                break
    
    return best_log, stop_reason

def process_model_dir(model_root):
    print(f"\n{'='*40}")
    print(f"Processing: {model_root}")
    
    if not os.path.exists(model_root):
        print(f"  [Skip] Directory not found.")
        return

    # 1. 获取排序后的日志序列
    logs = get_sorted_logs(model_root)
    if not logs:
        print("  [Error] No valid logs found.")
        return
        
    print(f"  Found {len(logs)} checkpoints. Simulating training...")

    # 2. 模拟 Early Stopping
    best_log, reason = simulate_early_stopping(logs)
    
    print(f"\n  => Result: {reason}")
    print(f"  => Selected Best Model:")
    print(f"     Checkpoint: {os.path.basename(best_log['ckpt_path'])}")
    print(f"     Epoch: {best_log['epoch']}")
    print(f"     Val Loss: {best_log['val_loss']:.6f}")

    # 3. 保存
    save_best_model(best_log['ckpt_path'], model_root)

def save_best_model(src_ckpt, original_root_dir):
    original_root_dir = original_root_dir.rstrip('/')
    target_dir = f"{original_root_dir}_valLoss"
    
    if os.path.exists(target_dir):
        print(f"  => Overwriting target: {target_dir}")
        shutil.rmtree(target_dir)
    else:
        print(f"  => Saving to: {target_dir}")
    
    shutil.copytree(src_ckpt, target_dir)
    
    # 复制辅助文件
    for extra_file in ["tokenizer.joblib", "label_encoder.joblib", "training_logs.json"]:
        src_file = os.path.join(original_root_dir, extra_file)
        if os.path.exists(src_file):
            shutil.copy2(src_file, target_dir)

    print(f"  [Success] Saved.")

if __name__ == "__main__":
    for d in TARGET_DIRS:
        process_model_dir(d)