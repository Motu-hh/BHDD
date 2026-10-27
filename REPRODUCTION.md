# Local BHDD reproduction workflow

These additions preserve the original `model/BHDD` implementation and provide a separate local positional-encoding correction, data preparation, diagnostics, and training entry points. They are not an exact reproduction of paper results.

## Environment

Use Python 3.10. `requirements.reproduction.txt` records the Windows environment used for these checks, including PyTorch 2.7.1 with CUDA 12.8. Install in a separate environment:

```powershell
python -m pip install -r requirements.reproduction.txt
```

The CUDA 11.7 setting in the upstream environment file was not used for the RTX 5070 Laptop GPU checks.

## Diagnostic and synthetic checks

```powershell
python smoke_test_lmvd.py
python smoke_test_backward.py
python run_synthetic_smoke.py
python verify_synthetic_results.py
python diagnose_model.py
```

The synthetic runner generates 20 artificial samples and runs two folds for two epochs. Synthetic metrics are only software-debugging evidence. Generated data, weights and reports remain local. Generation scripts refuse to overwrite completed runs.

## Prepare authorized LMVD data

Obtain the original LMVD feature archive separately. Supply its path explicitly:

```powershell
python prepare_real_lmvd.py --archive "D:\path\to\LMVD_Feature.zip"
python audit_lmvd_windows.py
python summarize_lmvd_preparation.py
```

The summarizer additionally expects `smoke_real_lmvd.py` to have completed. Run that script after archive preparation and before the summarizer. It performs only three updates on four real samples and does not estimate performance.

The preparation script streams the archive without modifying it or fully extracting its large CSV files. It uses the archive README's label ranges, selects 136 interleaved landmark coordinates, 17 AU intensities, six pose values and six gaze-vector values. It retains the first 915 video rows and 186 audio rows with zero padding, following the public loader. Full sequences remain in the source archive.

```powershell
python smoke_real_lmvd.py
python summarize_lmvd_preparation.py
python prepare_clean_lmvd.py
python train_lmvd_short.py
```

Cleaning preserves frame positions, zeros tracker-failure frames after scaling, isolates non-finite inputs and conflicting identical-prefix labels, and removes same-label identical-prefix duplicates. This is a local reconstruction policy, not author-confirmed original preprocessing.

Fixed approximately 80/10/10 video-level train/dev/test partitions are generated with seeds 42/43. Scaling statistics use only valid training frames, never development/test frames. `development_cv10.csv` is a future development-pool partition: each cross-validation fold MUST refit scaling inside that fold's training portion. Do not reuse the fixed-holdout scaler for cross-validation.

## Full fixed-split training in VS Code

Open the repository folder and select the `bhdd` environment's Python interpreter. Run these commands from the project root:

```powershell
python train_lmvd_full.py --check-only
python train_lmvd_full.py --epochs 50 --batch-size 4 --seed 0 --patience 10 --run-name lmvd_full_seed0
```

The full runner uses every sample in the fixed train/dev partitions. It selects the best model by development macro F1, saves `best_dev.pth`, `last.pth`, `history.csv`, `training.log` and `summary.json`, and never evaluates the reserved test partition. It uses the same Adam settings and constant learning rate as the local short run; it does not reproduce the upstream warmup/cosine schedule.

After at least one completed epoch, resume with the same settings and `--resume`. An interrupted partial epoch is replayed from the preceding completed epoch. Source and data hashes guard against changed configurations; an existing run is never silently overwritten.

## Interpretation limits

- No participant-to-video mapping is supplied, so video-level partitions do not establish subject independence.
- Independent prefix crops do not establish physical audio/video time alignment.
- Zeroing failed/padded inputs does not implement attention padding masks.
- The separate model variant changes only sequence-first positional encoding. Other unused upstream parameters remain unchanged.
- The original reconstruction-loss scaling remains; after standardization this auxiliary term is very small. Numerical success does not establish its usefulness.
- Short subset scores and a single-seed holdout run are exploratory results, not paper reproduction or confirmatory comparisons.

## Files kept out of version control

Real/synthetic generated features, labels, sample manifests, masks, splits, downloaded archives, weights, reports and local machine records are ignored. Only source code, this guide and the dependency snapshot should be shared. Obtain datasets through their original authorized sources.
