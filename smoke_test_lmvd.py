import sys
from pathlib import Path

import torch

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / "model" / "BHDD"))

from model import Net
from train import build_adjacency_matrix


def main():
    torch.manual_seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

    # The adjacency tensor must be on the GPU before model construction.
    adj = build_adjacency_matrix().to(device)
    net = Net(adj_matrix=adj, K=186).to(device)
    net.eval()

    lm = torch.randn(1, 915, 136, device=device)
    au = torch.randn(1, 915, 29, device=device)
    audio = torch.randn(1, 186, 128, device=device)

    print("Running forward pass...", flush=True)
    with torch.no_grad():
        logits, attention, indices, visual, reconstructed = net(lm, au, audio)
    torch.cuda.synchronize()

    for name, value in [
        ("logits", logits),
        ("attention", attention),
        ("visual", visual),
        ("reconstructed", reconstructed),
    ]:
        print(name, tuple(value.shape))
        if not torch.isfinite(value).all().item():
            raise RuntimeError(f"{name} contains NaN or Inf")

    print("selected indices:", indices.shape)
    assert logits.shape == (1, 2)
    assert indices.shape == (1, 186)
    assert visual.shape == reconstructed.shape == (1, 915, 165)
    print("PASS: random-input forward check")


if __name__ == "__main__":
    main()
