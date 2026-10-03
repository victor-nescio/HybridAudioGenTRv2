# scripts/v1 — first-submission analyses and run drivers

These scripts reproduce the figures, tables and statistics of the first submission
(git tag `v1-submission`). They read from `results/v1/`. The reusable pipeline
(data preparation, training, generation, FAD, CLAP fine-tuning, statistics) is in `scripts/`.

| Path | Purpose |
|---|---|
| `figurler/makale_fig_ana.py` | Paper figures 1–5 and 7–9 → `outputs/makale_figurleri/` |
| `figurler/makale_fig_spek_tablo.py` | Spectrogram figure (Fig. 6) and Tables III–V as images |
| `mcnemar_coherent.py` | Segment-level McNemar for the Phase-2 targets (Table III) |
| `k6_specaug.py` | SpecAugment vs generative comparison plot |
| `12b_mcnemar_lgbm_tur4.py`, `12c_mcnemar_lgbm_tum_turler.py`, `12d_mcnemar_cnn_vs_lgbm.py` | LightGBM / CNN-vs-LightGBM McNemar variants |
| `sonuclari_topla.py` | Phase-1 summary → `results/v1/results_summary_new.json` |
| `coherence_figurleri.py`, `sema_pipeline.py` | Exploratory acoustic-coherence figures and early pipeline diagram (not in the paper) |
| `10_training_report.py`, `11_phd_report_new_v2.py` | PhD progress reports (Turkish) |
| `kosu/` | PowerShell/BAT drivers used for the actual overnight runs (run from any folder; they switch to the repo root) |
