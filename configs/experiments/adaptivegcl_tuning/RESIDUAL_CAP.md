# Residual cap screening

The runner now accepts `residual_cap_03`, which only overrides
`adaptive_gcl.residual_alpha_max` to 0.3. Use the bounded-residual C0 control
(alpha initialization 0.1). `mean_layers` is the separate graph-depth control.
Neither change is a validated accuracy improvement yet.

Run the existing P1 before choosing P2. For the independent P2 screen:

```powershell
.venv-adaptive/Scripts/python.exe scripts/ablate_adaptive.py --config_dir configs/experiments/adaptivegcl_tuning/C0 --variants full residual_cap_03 mean_layers --epochs 30 --seeds 42 --sparsities 1.0
```

Add `--run` to execute. Do not reuse an existing output directory. Keep the
control in the same round because the Python/PyTorch environment may differ
from historical runs. Runtime versions are recorded in the run manifest.

Select on validation NDCG@20 and inspect Recall@20, gate saturation, residual
text/ID norm ratio and layer weights. Confirm any winner across seeds 42/43/44
and densities 1.0/0.25 before changing the default or claiming a baseline win.
