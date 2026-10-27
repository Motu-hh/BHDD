"""Replay each fold's best synthetic checkpoint against saved predictions."""
import sys
import json
import csv
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "synthetic_data" / "lmvd_smoke_seed0"
OUT = ROOT / "outputs" / "synthetic_smoke_seed0"
sys.path.insert(0, str(ROOT / "model" / "BHDD"))
from model import Net
from dataloader import BHDDDataset
from train import build_adjacency_matrix, sort_feature_filenames, collect_labels


def main():
    files = sort_feature_filenames(DATA / "landmarks")
    labels = collect_labels(files, DATA / "labels")
    saved_preds = np.load(OUT / "total_pre.npy", allow_pickle=False)
    saved_labels = np.load(OUT / "total_label.npy", allow_pickle=False)
    device = torch.device("cuda")
    rows, checks, offset = [], [], 0
    splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(splitter.split(files, labels), 1):
        assert set(train_idx).isdisjoint(set(val_idx))
        best, checkpoint_path, checkpoint = -1, None, None
        for path in sorted((OUT / "checkpoints" / f"fold_{fold}").glob("*.pth")):
            item = torch.load(path, map_location="cpu", weights_only=True)
            if item["metrics"]["accuracy"] > best:
                best, checkpoint_path, checkpoint = item["metrics"]["accuracy"], path, item
        if checkpoint is None:
            raise RuntimeError(f"No checkpoint for fold {fold}")
        net = Net(build_adjacency_matrix().to(device), K=186).to(device)
        net.load_state_dict(checkpoint["net"], strict=True)
        net.eval()
        val_files = [files[i] for i in val_idx]
        dataset = BHDDDataset(DATA / "landmarks", DATA / "au_pose_gaze", DATA / "audio", DATA / "labels", val_files, mode="dev")
        observed, expected = [], []
        with torch.no_grad():
            for lm, au, audio, label in DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0):
                logits = net(lm.to(device), au.to(device), audio.to(device))[0]
                assert torch.isfinite(logits).all().item()
                observed.extend(logits.argmax(1).cpu().tolist())
                expected.extend(label.tolist())
        n = len(val_idx)
        assert np.array_equal(observed, saved_preds[offset:offset+n]), f"Predictions differ in fold {fold}"
        assert np.array_equal(expected, saved_labels[offset:offset+n]), f"Labels differ in fold {fold}"
        accuracy = float(np.mean(np.array(observed) == expected) * 100)
        assert abs(accuracy - best) < 1e-8
        for name, truth, pred in zip(val_files, expected, observed):
            rows.append({"synthetic": True, "fold": fold, "sample_id": Path(name).stem, "label": truth, "prediction": pred})
        checks.append({"fold": fold, "checkpoint": str(checkpoint_path.relative_to(OUT)), "epoch": checkpoint["epoch"], "accuracy_percent": accuracy, "saved_predictions_match": True})
        print(f"Fold {fold}: best checkpoint epoch {checkpoint['epoch']}; predictions match", flush=True)
        offset += n
        del net, checkpoint
    assert offset == 20 and len({r["sample_id"] for r in rows}) == 20
    with (OUT / "synthetic_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    report = {"status": "PASS", "synthetic_only": True, "each_sample_validated_once": True, "train_validation_ids_disjoint_per_fold": True, "folds": checks}
    (OUT / "best_checkpoint_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown = """# BHDD 合成数据流程测试记录

本记录只证明代码流程可执行，不代表 LMVD 复现或真实抑郁识别效果。

## 数据与配置

- 20 个独立生成的虚拟样本，人工类别各 10 个；没有真实参与者。
- 人脸关键点 915×136，表情/姿态/视线 915×29，音频 186×128，均为 float32。
- 生成与训练种子 0；原训练入口使用固定随机状态 42 做分层两折划分。
- 每折训练 10 个、验证 10 个，训练与验证样本 ID 不重叠。
- 每折 2 轮，batch size=2，学习率 1e-5，Adam，线性调度，warmup=1 轮。

## 已验证

- 数据生成、格式与有限值检查、原始 DataLoader 加载。
- 原训练入口完成前向、反向、参数更新和验证。
- 两折最佳模型重新加载后，逐样本预测与原保存结果完全一致。
- 20 个样本各在一个验证折出现一次；预测明细见 synthetic_predictions.csv。
- 日志、配置、混淆矩阵、指标汇总及模型文件均已保存。

## 仅供调试的合成结果

| 折 | 最佳轮次 | 合成验证准确率 |
|---|---|---|
| 1 | 1 | 50% |
| 2 | 2 | 80% |

两折平均 65%，加权 F1 两折均值约 0.5625。标签含人为信号，样本量很小，且使用验证准确率选择模型。这些数字不能作为独立测试表现、临床证据或与论文比较的结果。

## 已知限制

- 某些轮次仅预测一个类别，原指标函数产生 UndefinedMetricWarning。
- 先前反向检查发现 token_selector.head_weights、decoder.norm.weight、decoder.norm.bias 没有梯度；尚未修改模型。
- 未验证真实数据预处理、真实数据划分、长期训练、跨种子稳定性或论文结果。

## 下一阶段

获取 LMVD 或 D-Vlog 官方数据，核对特征、标签、被试 ID 和划分协议，再制定真实数据复现方案。当前合成流程测试已完成。
"""
    (OUT / "流程测试记录.md").write_text(markdown, encoding="utf-8")
    print("PASS: best-checkpoint replay and per-sample prediction audit")


if __name__ == "__main__":
    main()
