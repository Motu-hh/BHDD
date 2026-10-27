"""One synthetic training step; does not save trained weights or measure accuracy."""
import sys
from pathlib import Path
import torch

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / "model" / "BHDD"))
from model import Net
from train import build_adjacency_matrix, reconstruction_loss


def main():
    torch.manual_seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    net = Net(build_adjacency_matrix().to(device), K=186).to(device)
    net.train()
    optimizer = torch.optim.Adam(
        net.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=1e-4,
    )
    lm = torch.randn(1, 915, 136, device=device)
    au = torch.randn(1, 915, 29, device=device)
    audio = torch.randn(1, 186, 128, device=device)
    labels = torch.tensor([1], device=device)
    tracked = net.Regress.fc[-1].weight
    before = tracked.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    print("Forward and loss...", flush=True)
    logits, _, _, visual, reconstructed = net(lm, au, audio)
    ce = torch.nn.functional.cross_entropy(logits, labels)
    rec = reconstruction_loss(visual, reconstructed)
    loss = ce + 0.7 * rec
    if not torch.isfinite(loss).item():
        raise RuntimeError("Non-finite loss")
    print("Backward...", flush=True)
    loss.backward()
    with_grad = []
    without_grad = []
    for name, parameter in net.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            without_grad.append(name)
        else:
            if not torch.isfinite(parameter.grad).all().item():
                raise RuntimeError(f"Non-finite gradient: {name}")
            with_grad.append(name)
    if not with_grad:
        raise RuntimeError("No gradients computed")
    grad_norm = torch.nn.utils.clip_grad_norm_(
        net.parameters(), max_norm=1.0, error_if_nonfinite=True
    )
    optimizer.step()
    torch.cuda.synchronize()
    for name, parameter in net.named_parameters():
        if not torch.isfinite(parameter).all().item():
            raise RuntimeError(f"Non-finite updated parameter: {name}")
    delta = (tracked.detach() - before).abs().max().item()
    if delta == 0:
        raise RuntimeError("Classification head did not update")
    print("classification loss:", ce.item())
    print("reconstruction loss:", rec.item())
    print("combined loss:", loss.item())
    print("gradient norm before clipping:", grad_norm.item())
    print("parameter tensors with gradients:", len(with_grad))
    print("parameter tensors without gradients:", without_grad)
    print("classification head max update:", delta)
    print("peak GPU memory (MiB):", torch.cuda.max_memory_allocated() / 1024**2)
    print("PASS: synthetic backward and optimizer step")


if __name__ == "__main__":
    main()
