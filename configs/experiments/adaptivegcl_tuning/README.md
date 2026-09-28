# AdaptiveGCL validation screening

The C0–C5 configurations were prepared from the checkpoint identified in
`control.json`. C0 is the reference; each subsequent configuration changes
one training factor. Active model and baseline configs remain separate.

Preview the six runs from the frozen C0 configuration:

```bash
python scripts/ablate_adaptive.py --config_dir configs/experiments/adaptivegcl_tuning/C0 --epochs 30 --seeds 42 --sparsities 1.0 --variants full no_ssl ssl_0003 mlp_decay_1e4 lr_0003 no_dislikes
```

Add `--run` to execute a fresh screening round. The runner creates a timestamped
directory under `results/experiments/adaptivegcl_validation_upgrade/` and refuses
to overwrite an explicit existing output directory. It does not resume the old
round automatically. Do not launch another round while the original is running.

Refresh comparison, diagnostics, validation curves and the decision report without training:

```bash
python scripts/report_adaptive_tuning.py --output_dir results/experiments/adaptivegcl_validation_upgrade/round1
```

P1 uses validation only. Incomplete rounds defer selection. P2 and multi-seed
confirmation require review of the completed P1 results; no default model
change is justified by the implementation checks alone.

Implementation handoff: `docs/adaptivegcl-validation-upgrade-handoff.md`.
The user runs real-data training and reviews experimental acceptance separately.
