# results/v1 — outcomes behind the first submission

Every JSON file here was produced by the code tagged `v1-submission`
(IEEE Access, July 2026). The files are frozen: the revised evaluation writes to
`results/v2/`. "Phase 1" and "Phase 2" are the two target-class sets tested after the
two pipeline fixes (unique seed per clip, deterministic FAD) described in the paper.

Configurations: **H1** = pretrained AudioLDM2/CLAP, English prompts · **C2** = CLAP
fine-tuning cycle 1, English · **C3** = cycle 1, Turkish · **C4** = cycle 2, English.

## Baselines

| File | Content |
|---|---|
| `cnn_20260427_0312_sonuclar.json` | CNN (ResNet-18) baseline — val/test macro-F1, per-class report |
| `lstm_20260427_0732_sonuclar.json` | BiLSTM + attention baseline |
| `ast_20260427_1430_sonuclar.json` | AST baseline |
| `fusion_20260428_1907_sonuclar.json`, `fusion_20260428_1924_sonuclar.json` | Late-fusion ensembles |
| `mcnemar_cnn_vs_lgbm.json` | CNN vs LightGBM, file-level McNemar |
| `sinif_dagilim_test.json` | Test-set class distribution (paper Fig. 2) |

## CLAP domain fine-tuning

| File | Content |
|---|---|
| `clap_finetune_20260510_1712.json` | Cycle 1 (BBC audio–caption pairs) |
| `clap_finetune_20260523_0532.json`, `clap_finetune_20260523_0643.json` | Cycle 2 intermediate runs |
| `clap_finetune_20260523_0658.json` | Cycle 2 final (BBC + synthetic audio) |

## Phase 1 — first target set (31 May – 2 Jun 2026)

Targets: construction, religious place, emergency service and other animals
(all four augmented with 50 clips each).

| File | Content |
|---|---|
| `sentetik_uretim_ozet_tur1.json` … `_tur4.json` | Generation summaries (H1, C2, C3, C4) |
| `fad_sonuclar.json`, `fad_sonuclar_tur1_det.json` | FAD, H1 |
| `fad_sonuclar_tur2.json`, `fad_sonuclar_tur3.json`, `fad_sonuclar_4.json` | FAD, C2 / C3 / C4 |
| `iteratif_loop_1_20260601_1725.json` | Augmented training, H1 |
| `iteratif_loop_tur2_20260601_2132.json`, `iteratif_loop_tur2_re_20260602_1255.json` | Augmented training, C2 (run + determinism re-run) |
| `iteratif_loop_tur3_20260602_0303.json` | Augmented training, C3 |
| `iteratif_loop_tur4_20260602_0701.json` | Augmented training, C4 |
| `mcnemar_sonuclar.json`, `mcnemar_tur2_new.json`, `mcnemar_tur4_new.json` | Segment-level McNemar |
| `results_summary_new.json` | Phase 1 summary (FAD + classification + McNemar) |

## Phase 2 — targets dog, bird, weather (4–6 Jun 2026)

| File | Content |
|---|---|
| `sentetik_uretim_ozet_turcoh.json` … `_turcoh4.json` | Generation summaries (H1, C2, C3, C4) |
| `fad_sonuclar_cohH1.json` … `fad_sonuclar_cohC4.json` | Deterministic FAD (paper Table IV) |
| `iteratif_loop_coherent_20260604_0320.json` | Augmented training, H1 |
| `iteratif_loop_coherent2_Dongu1_EN_20260604_0942.json`, `iteratif_loop_coherent2_Dongu1_EN_rerun_20260605_1105.json` | C2 (run + determinism re-run) |
| `iteratif_loop_coherent3_Dongu1_TR_20260604_2237.json` | C3 |
| `iteratif_loop_coherent4_Dongu2_EN_20260605_0726.json` | C4 |
| `mcnemar_coherent.json`, `mcnemar_coh2.json`, `mcnemar_coh3.json`, `mcnemar_coh4.json` | Segment-level McNemar (paper Table III) |
| `mcnemar_dosya_duzeyi.json` | Recording-level majority-vote McNemar |
| `kume_direncli_test.json` | Recording-level cluster bootstrap (`preds_test_seg.npz`, untracked, holds the segment predictions) |
| `sentetik_dogrulama.json` | Baseline-classifier labels assigned to the synthetic clips |

## Conventional-augmentation comparison

| File | Content |
|---|---|
| `iteratif_loop_specaug_20260606_0129.json` | `baseline` = SpecAugment only; `augmented` = SpecAugment + synthetic clips |
| `mcnemar_specA_vs_gen.json`, `mcnemar_specA_vs_stacked.json` | McNemar comparisons (paper Table V) |

## Exploratory (not reported in the paper)

| File | Content |
|---|---|
| `korelasyon_ozet.json`, `sinif_tutarlilik_clap.json`, `sinif_tutarlilik_vggish.json` | Acoustic-coherence analysis |
| `mcnemar_lgbm_tur4.json`, `mcnemar_lgbm_tum_turler.json` | LightGBM augmentation tests |

## Superseded — produced before the pipeline fixes (not valid evidence)

`iteratif_loop_20260504_0016.json`, `iteratif_loop_20260504_0029.json`,
`iteratif_loop_20260506_0328.json`, `iteratif_loop_20260506_0348.json`,
`iteratif_loop_tur2_20260515_0847.json`, `iteratif_loop_tur2_20260516_0313.json`,
`iteratif_loop_tur3_20260515_0852.json`, `iteratif_loop_tur4_20260524_0925.json`,
`sentetik_uretim_ozet.json`

## Known issues found during revision

These issues are corrected in the revised (v2) evaluation:

1. **Control class.** In Phase 2 the control class (other animals) received synthetic
   clips only in H1. In C2, C3 and C4 the control was not augmented (`scripts/v1/kosu/gece_diger_turlar.ps1`).
2. **SpecAugment + generative run.** This run used the H1 clips (`data/synthetic_turcoh`),
   not the C4 clips.
3. **CLAP split.** CLAP fine-tuning used its own 80/10/10 split of the full archive.
   As a result, 683 of the 2,304 classifier test files (29.6%) were in the CLAP training partition.
