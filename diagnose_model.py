"""Read-only model diagnostics using synthetic inputs; no optimizer steps."""
import sys
import json
import hashlib
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "model" / "BHDD"))
from model import Net, PositionalEncoding
from train import build_adjacency_matrix, reconstruction_loss


def main():
    torch.manual_seed(0)
    device = torch.device("cuda")
    model_file = ROOT / "model" / "BHDD" / "model.py"
    before = hashlib.sha256(model_file.read_bytes()).hexdigest()
    pe = PositionalEncoding(512).to(device)
    with torch.no_grad():
        encoded = pe(torch.zeros(915, 2, 512, device=device))
    time_delta = (encoded[1, 0] - encoded[0, 0]).abs().max().item()
    batch_delta = (encoded[0, 1] - encoded[0, 0]).abs().max().item()
    assert time_delta == 0 and batch_delta > 0
    print("Position encoding: time delta", time_delta, "batch-slot delta", batch_delta, flush=True)

    net = Net(build_adjacency_matrix().to(device), K=186).to(device)
    net.eval()
    lm = torch.randn(1, 915, 136, device=device)
    au = torch.randn(1, 915, 29, device=device)
    audio = torch.randn(1, 186, 128, device=device)
    with torch.no_grad():
        pair = net(lm.repeat(2,1,1), au.repeat(2,1,1), audio.repeat(2,1,1))[0]
    output_delta = (pair[0] - pair[1]).abs().max().item()
    print("Identical samples in same batch: max logit difference", output_delta, flush=True)

    logits, _, _, visual, reconstructed = net(lm, au, audio)
    ce = torch.nn.functional.cross_entropy(logits, torch.tensor([1], device=device))
    rec = reconstruction_loss(visual, reconstructed)
    raw_mse = torch.nn.functional.mse_loss(visual, reconstructed)
    shared = net.feature_proj.weight
    ce_grad = torch.autograd.grad(ce, shared, retain_graph=True)[0]
    rec_grad = torch.autograd.grad(0.7*rec, shared, retain_graph=True)[0]
    ratio = (rec_grad.norm() / ce_grad.norm()).item()
    loss_value = ce.item()
    scaled_rec_value = (0.7*rec).item()
    combined = ce + 0.7*rec
    combined_value = combined.item()
    combined.backward()
    missing = [name for name, p in net.named_parameters() if p.requires_grad and p.grad is None]
    after = hashlib.sha256(model_file.read_bytes()).hexdigest()
    assert before == after
    report = {
        "synthetic_only": True, "seed": 0, "mode": "eval, gradients enabled for loss diagnosis",
        "source_model_sha256": before, "source_unchanged": before == after,
        "positional_encoding_time_delta": time_delta,
        "positional_encoding_batch_slot_delta": batch_delta,
        "identical_input_pair_max_logit_difference": output_delta,
        "cross_entropy": loss_value, "raw_reconstruction_mse": raw_mse.item(),
        "weighted_reconstruction_loss": scaled_rec_value,
        "combined_loss": combined_value,
        "combined_equals_ce_in_this_float32_check": combined_value == loss_value,
        "shared_feature_projection_rec_to_ce_gradient_norm_ratio": ratio,
        "parameters_without_gradients": missing,
        "limits": "No real-data accuracy or paper-intent conclusion; no model edits or optimizer updates",
    }
    folder = ROOT / "outputs" / "model_diagnostics_seed0"
    folder.mkdir(exist_ok=False)
    (folder / "evidence.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown = f"""# BHDD 实现核查

范围：使用 seed=0 的合成输入定位实现行为；不修改官方模型，不训练或选择新模型。

## 1. 位置编码按批次位置而不是时间位置变化：已确认

模型将输入转为 (T,B,D)，但 PositionalEncoding 使用 x.size(1) 截取形状为 (1,max_len,D) 的编码。
对于零输入 (915,2,512)，相邻时间步差为 {time_delta:.6g}，相邻批次位置差为 {batch_delta:.6g}。
这意味着一个视频内所有时间步收到相同编码，不同批次位置却收到不同编码。
eval 模式下，同一输入复制到同一批次两个位置，分类输出最大差为 {output_delta:.6g}。
后续修正应使位置编码沿 T 轴展开，例如 self.pe[:, :x.size(0)].transpose(0,1)，同时明确函数的输入布局。
这是可复现的实现问题；它对真实数据准确率的影响尚未评估。

## 2. 三个声明的参数未参与计算：已确认

本次组合损失反向计算后无梯度的参数：{', '.join(missing)}。
代码中 token_selector.head_weights 声明后没有用于注意力分数，decoder.norm 声明后未在解码器 forward 使用。
缺少梯度由未使用造成，不是 CUDA 安装失败。是否本应使用这些层，需要论文或作者说明。
仅删除它们不会实现缺失的加权或归一化，还会改变 checkpoint 键名，暂保留。

## 3. 重建损失缩放：存在量级风险，不能断言作者写错

原函数使用 clamp(MSE,1e-10,1e10)/1e10，组合权重 0.7。
本次原始 MSE={raw_mse.item():.6g}，加权重建项={scaled_rec_value:.6g}，分类损失={loss_value:.6g}。
在共享 feature_proj.weight 上，重建项/分类项梯度范数之比为 {ratio:.6g}。
组合损失与分类损失的 float32 数值相同：{combined_value == loss_value}。
数值相同不等于自动微分中梯度为零；本次测量仅说明合成输入下辅助项极弱。
真实特征可能具有不同量纲，不能据此直接取消 1e10 缩放。

## 4. 其他静态观察

- AttentionDecoder 在创建子层之前调用 self.apply(weight_init)，因此该次调用未遍历随后创建的子层。
- weight_init 中 LayerNorm 判断嵌套在 Linear 判断内部，该分支不可达；框架默认初始化仍然存在。
- 原训练按文件做 StratifiedKFold；真实视频若共享被试 ID，需要额外核对被试级划分。

## 结论与边界

能执行一次训练流程，不能证明数学语义正确。此前合成数据流程 PASS 的结论仍成立，但需附上以上实现限制。
本次只定位问题，不更改官方代码，也不把合成结果作为论文复现。
源文件 SHA256：{before}。
建议先在单独副本中修正位置编码，并验证同一输入在不同批次位置的输出一致性；原版保留作为参考。
"""
    (folder / "实现核查.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
