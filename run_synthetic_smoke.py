"""Synthetic LMVD-format pipeline check. No real people or clinical labels."""
import os
import sys
import json
import hashlib
import subprocess
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "synthetic_data" / "lmvd_smoke_seed0"
OUT = ROOT / "outputs" / "synthetic_smoke_seed0"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if DATA.exists() or OUT.exists():
        raise RuntimeError("Synthetic data or run already exists; refusing overwrite.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    rng = np.random.default_rng(0)
    folders = {key: DATA / key for key in ("landmarks", "au_pose_gaze", "audio", "labels")}
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    manifest = {
        "synthetic": True, "purpose": "Pipeline debugging only; not LMVD or clinical evidence",
        "seed": 0, "samples": 20, "class_counts": {"0": 10, "1": 10},
        "generator": "Independent Gaussian noise plus random-phase temporal sine and artificial class offset; no real participants",
        "shapes": {"landmarks": [915,136], "au_pose_gaze": [915,29], "audio": [186,128]},
        "dtype": "float32", "samples_manifest": [],
        "source_hashes": {name: digest(ROOT / "model" / "BHDD" / name) for name in ("model.py", "train.py", "dataloader.py")},
    }
    for i in range(20):
        label = i % 2
        stem = f"{i+1:04d}"
        records = {}
        for key, (timesteps, channels) in manifest["shapes"].items():
            time = np.linspace(0, 4*np.pi, timesteps)[:, None]
            phase = rng.uniform(0, 2*np.pi, (1, channels))
            signal = (0.15 * rng.normal(size=(timesteps, channels))
                      + 0.2 * np.sin(time + phase)
                      + (2*label-1)*0.1).astype(np.float32)
            path = folders[key] / f"{stem}.npy"
            np.save(path, signal)
            loaded = np.load(path, allow_pickle=False)
            assert loaded.shape == (timesteps, channels) and np.isfinite(loaded).all()
            records[key] = {"file": str(path.relative_to(DATA)), "sha256": digest(path)}
        label_file = folders["labels"] / f"{stem}_Depression.csv"
        label_file.write_text(f"label\n{label}\n", encoding="utf-8")
        manifest["samples_manifest"].append({"id": stem, "label": label, "features": records})
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (DATA / "README.md").write_text(
        "# SYNTHETIC DATA ONLY\n\n20 generated samples, 10 per artificial class. "
        "No LMVD records, real participants, or clinical meaning. "
        "Use only to check software execution; all resulting metrics are synthetic.\n",
        encoding="utf-8",
    )
    OUT.mkdir(parents=True)
    command = [sys.executable, "-B", "-u", str(ROOT / "model" / "BHDD" / "train.py"),
        "--lm-dir", str(folders["landmarks"]), "--au-pose-gaze-dir", str(folders["au_pose_gaze"]),
        "--audio-dir", str(folders["audio"]), "--label-dir", str(folders["labels"]),
        "--output-dir", str(OUT.parent), "--run-name", OUT.name,
        "--num-folds", "2", "--num-epochs", "2", "--warmup-epochs", "1",
        "--batch-size", "2", "--num-workers", "0", "--seed", "0", "--schedule", "linear"]
    (OUT / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
    with (OUT / "environment.txt").open("w", encoding="utf-8") as handle:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=handle, check=True)
    print("Generated and checked 20 synthetic samples. Starting 2-fold x 2-epoch smoke run.", flush=True)
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    with (OUT / "console.log").open("w", encoding="utf-8") as handle:
        subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
    sys.path.insert(0, str(ROOT / "model" / "BHDD"))
    from model import Net
    from train import build_adjacency_matrix
    from dataloader import BHDDDataset
    checkpoints = sorted((OUT / "checkpoints").rglob("*.pth"))
    assert len({p.parent.name for p in checkpoints}) == 2, "Missing fold checkpoint"
    predictions = np.load(OUT / "total_pre.npy", allow_pickle=False)
    labels = np.load(OUT / "total_label.npy", allow_pickle=False)
    assert predictions.shape == labels.shape == (20,)
    assert set(np.unique(predictions)).issubset({0, 1})
    dataset = BHDDDataset(folders["landmarks"], folders["au_pose_gaze"], folders["audio"], folders["labels"], ["0001.npy"])
    device = torch.device("cuda")
    reloads = []
    for fold in ("fold_1", "fold_2"):
        path = next(p for p in checkpoints if p.parent.name == fold)
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        net = Net(build_adjacency_matrix().to(device), K=186).to(device)
        net.load_state_dict(checkpoint["net"], strict=True)
        net.eval()
        lm, au, audio, _ = dataset[0]
        with torch.no_grad():
            output = net(lm.unsqueeze(0).to(device), au.unsqueeze(0).to(device), audio.unsqueeze(0).to(device))[0]
        assert output.shape == (1, 2) and torch.isfinite(output).all().item()
        reloads.append(str(path.relative_to(OUT)))
        del net, checkpoint
    for file in ("config.json", "metrics_summary.json", "training.log", "confusion_matrix.png"):
        assert (OUT / file).stat().st_size > 0
    report = {"status": "PASS", "synthetic_only": True, "samples": 20,
        "folds": 2, "epochs_per_fold": 2, "checkpoint_reload_passed": reloads,
        "scope": "Generation, loading, training, validation, output saving and checkpoint reloading; no paper reproduction claim"}
    (OUT / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text("# Synthetic pipeline check\n\nAll metrics are from artificial labels and must not be reported as LMVD or depression detection results.\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
